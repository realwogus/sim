#!/usr/bin/env python3
"""Persistent independent 7-DOF rail-PiPER cuRobo planner."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria

from rail_piper_config import JOINT_NAMES, build_rail_robot_config
from rail_motion_planner import RailMotionPlanner


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORLD_CONFIG = PROJECT_DIR / "worlds" / "piper_rail.yml"
GOAL_QUATERNION = [0.53240967, -0.53241307, -0.46533632, -0.46533674]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rail-mounted PiPER planner server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5564)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    return parser.parse_args()


def send_message(stream, payload: dict[str, Any]) -> None:
    stream.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
    stream.flush()


def failure_stage(diagnostics: list[dict[str, Any]]) -> str:
    """Select the most actionable stage from all planning attempts."""
    stages = [item.get("stage") for item in diagnostics]
    for stage in ("INTERPOLATION", "CONSTRAINT", "GOAL_ERROR", "TRAJOPT"):
        if stage in stages:
            return stage
    if "GRAPH_SEARCH" in stages:
        return "GRAPH_SEARCH"
    if "IK_SEED" in stages:
        return "IK_SEED"
    return "NO_RESULT"


def print_diagnostics(request_id: Any, diagnostics: list[dict[str, Any]]) -> None:
    print(f"rail diagnostics id={request_id}", flush=True)
    for item in diagnostics:
        graph = (
            "not-used"
            if not item.get("graph_used")
            else "success"
            if item.get("graph_success")
            else "failed"
        )
        line = (
            f"  attempt={item.get('attempt')} stage={item.get('stage')} "
            f"ik={item.get('ik_success', 0)}/{item.get('ik_seeds', 0)} "
            f"graph={graph} trajopt={item.get('trajopt_success', 0)}"
        )
        if "feasible_seeds" in item:
            line += (
                f" feasible={item['feasible_seeds']}/"
                f"{item.get('evaluated_seeds', 0)}"
            )
        if "interpolated_feasible_seeds" in item:
            line += (
                f" interpolated={item['interpolated_feasible_seeds']}/"
                f"{item.get('interpolated_evaluated_seeds', 0)}"
            )
        if "min_position_error" in item:
            line += (
                f" pos_err={item['min_position_error']:.6f}m/"
                f"{item.get('position_tolerance', 0):.6f}m"
            )
        print(line, flush=True)
        for name, values in item.get("constraints", {}).items():
            if values.get("positive_count", 0) > 0:
                print(
                    f"    violation={name} max={values['max']:.6f} "
                    f"positive={values['positive_count']}",
                    flush=True,
                )


def plan_request(planner: MotionPlanner, request: dict[str, Any]) -> dict[str, Any]:
    request_id = request.get("id")
    started = time.perf_counter()
    start_values = request["start"]
    target = request["target_position_world"]
    if len(start_values) != len(JOINT_NAMES):
        raise ValueError("rail start must contain 7 values")
    if len(target) != 3:
        raise ValueError("rail target must contain XYZ")

    start = JointState.from_position(
        torch.tensor([start_values], device="cuda", dtype=torch.float32),
        joint_names=planner.joint_names,
    )
    goal = GoalToolPose.from_poses(
        {
            "gripper_center": Pose(
                position=torch.tensor([target], device="cuda", dtype=torch.float32),
                quaternion=torch.tensor(
                    [GOAL_QUATERNION], device="cuda", dtype=torch.float32
                ),
            )
        },
        num_goalset=1,
    )
    result = planner.plan_pose(goal, start)
    elapsed = time.perf_counter() - started
    if result is None or not bool(result.success.any().item()):
        diagnostics = list(getattr(planner, "last_plan_diagnostics", []))
        stage = failure_stage(diagnostics)
        print_diagnostics(request_id, diagnostics)
        ik_result = planner.ik_solver.solve_pose(goal, current_state=start)
        ik_ok = bool(ik_result.success.any().item())
        ik_positions = None
        if ik_ok:
            solutions = ik_result.js_solution.position.reshape(-1, len(JOINT_NAMES))
            success_mask = ik_result.success.reshape(-1)
            if success_mask.numel() == solutions.shape[0]:
                solutions = solutions[success_mask]
            ik_positions = solutions[0].detach().cpu().tolist()
        return {
            "id": request_id,
            "ok": False,
            "ik_ok": ik_ok,
            "trajopt_ok": False,
            "failure_stage": stage,
            "diagnostics": diagnostics,
            "ik_positions": ik_positions,
            "error": (
                "Rail-PiPER IK found no collision-free goal configuration"
                if not ik_ok
                else "Rail-PiPER IK succeeded but trajectory optimization failed"
            ),
            "wall_time": elapsed,
            "target_position_world": target,
        }

    trajectory = result.get_interpolated_plan()
    positions = trajectory.position.detach().cpu().reshape(-1, len(JOINT_NAMES))
    return {
        "id": request_id,
        "ok": True,
        "ik_ok": True,
        "trajopt_ok": True,
        "joint_names": planner.joint_names,
        "dt": float(planner.trajopt_solver.config.interpolation_dt),
        "positions": positions.tolist(),
        "target_position_world": target,
        "solver_time": float(result.solve_time),
        "wall_time": elapsed,
    }


def serve(connection: socket.socket, planner: MotionPlanner) -> None:
    with connection, connection.makefile("rwb") as stream:
        while line := stream.readline():
            try:
                request = json.loads(line)
                if request.get("type") == "ping":
                    response = {"id": request.get("id"), "ok": True, "type": "pong"}
                elif request.get("type") == "plan":
                    response = plan_request(planner, request)
                    print(
                        f"rail plan id={request.get('id')} ok={response['ok']} "
                        f"wall={response.get('wall_time', 0):.3f}s "
                        f"solver={response.get('solver_time', 0):.3f}s",
                        flush=True,
                    )
                else:
                    raise ValueError(f"Unknown request type: {request.get('type')}")
            except Exception as exc:
                response = {
                    "id": request.get("id") if "request" in locals() else None,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            send_message(stream, response)


def main() -> None:
    args = parse_args()
    robot = build_rail_robot_config()
    world = yaml.safe_load(WORLD_CONFIG.read_text(encoding="utf-8"))
    cfg = MotionPlannerCfg.create(
        robot=robot,
        scene_model=world,
        collision_cache={"cuboid": 4},
        interpolation_dt=0.02,
        position_tolerance=0.002,
        orientation_tolerance=0.05,
    )
    planner = RailMotionPlanner(cfg)
    try:
        planner.update_tool_pose_criteria(
            {"gripper_center": ToolPoseCriteria.track_position()}
        )
        print("warming up independent rail-PiPER 7-DOF planner...", flush=True)
        planner.warmup(
            enable_graph=True,
            num_warmup_iterations=args.warmup_iterations,
        )
        print(f"rail planner ready: tcp://{args.host}:{args.port}", flush=True)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(2)
            while True:
                connection, _ = server.accept()
                serve(connection, planner)
    except KeyboardInterrupt:
        print("rail planner stopping", flush=True)
    finally:
        planner.destroy()


if __name__ == "__main__":
    main()
