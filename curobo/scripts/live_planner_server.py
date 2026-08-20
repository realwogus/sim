#!/usr/bin/env python3
"""Persistent cuRobo planner serving newline-delimited JSON over TCP."""

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
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_DIR / "robots" / "piper" / "piper.yml"
WORLD_CONFIG = PROJECT_DIR / "worlds" / "piper_red_gate.yml"
ENDPOINT_WORLD_CONFIG = PROJECT_DIR / "worlds" / "piper_gate.yml"
GOAL_QUATERNION = [0.53240967, -0.53241307, -0.46533632, -0.46533674]
ROBOT_BASE_WORLD_Z = 0.20
PARTNER_BASE_WORLD_X = 0.66
RED_HALF_HEIGHT = 0.036
APPROACH_CLEARANCE = 0.030


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent PiPER cuRobo planning server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5561)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--mode", choices=("red", "endpoint"), default="red")
    return parser.parse_args()


def load_configs(mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    robot = yaml.safe_load(ROBOT_CONFIG.read_text(encoding="utf-8"))
    robot["kinematics"]["cspace"]["default_joint_position"] = [0.0] * 6
    world_path = ENDPOINT_WORLD_CONFIG if mode == "endpoint" else WORLD_CONFIG
    world = yaml.safe_load(world_path.read_text(encoding="utf-8"))
    return robot, world


def world_for_red(template: dict[str, Any], red_pose_world: list[float]) -> dict[str, Any]:
    world = json.loads(json.dumps(template))
    x, y, z, qw, qx, qy, qz = red_pose_world
    world["cuboid"]["red_block"]["pose"] = [
        x,
        y,
        z - ROBOT_BASE_WORLD_Z,
        qw,
        qx,
        qy,
        qz,
    ]
    return world


def world_for_endpoint(
    template: dict[str, Any],
    obstacles_enabled: bool,
    other_robot_enabled: bool,
    other_robot_spheres: dict[str, Any],
    arm: str,
) -> dict[str, Any]:
    world = json.loads(json.dumps(template))
    if not obstacles_enabled:
        # Keep the supporting table, but remove the four gate pieces.
        world["cuboid"] = {"tabletop": world["cuboid"]["tabletop"]}
    if arm == "partner":
        # Express the primary-base world in the opposing base frame. All fixed
        # cuboids are axis-aligned, so a pi yaw only changes their positions.
        for obstacle in world.get("cuboid", {}).values():
            obstacle["pose"][0] = PARTNER_BASE_WORLD_X - obstacle["pose"][0]
            obstacle["pose"][1] = -obstacle["pose"][1]
    if other_robot_enabled:
        world["sphere"] = other_robot_spheres
    return world


def create_other_robot_spheres(
    planner: MotionPlanner, joint_values: list[float]
) -> dict[str, Any]:
    """Represent the opposing arm's current pose in the planning arm's frame."""
    if len(joint_values) != len(planner.joint_names):
        raise ValueError("other_start must contain 6 values")
    q = torch.tensor(
        [joint_values], device="cuda", dtype=torch.float32
    )
    spheres = planner.kinematics.get_robot_as_spheres(q, filter_valid=True)[0]
    result: dict[str, Any] = {}
    for index, sphere in enumerate(spheres):
        x, y, z = sphere.pose[:3]
        # The two bases are related by the same self-inverse transform:
        # translation x=0.66 m followed by yaw=pi.
        result[f"other_arm_{index:03d}"] = {
            "pose": [
                PARTNER_BASE_WORLD_X - float(x),
                -float(y),
                float(z),
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            "radius": float(sphere.radius),
        }
    return result


def send_message(stream, payload: dict[str, Any]) -> None:
    stream.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
    stream.flush()


def plan_request(
    planner: MotionPlanner,
    world_template: dict[str, Any],
    request: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    request_id = request.get("id")
    started = time.perf_counter()
    start_values = request["start"]
    if len(start_values) != 6:
        raise ValueError("start must contain 6 values")
    if mode == "red":
        red_pose_world = request["red_pose_world"]
        if len(red_pose_world) != 7:
            raise ValueError("red_pose_world must contain 7 values")
        planner.update_world(Scene.create(world_for_red(world_template, red_pose_world)))
        target_position_world = [
            red_pose_world[0],
            red_pose_world[1],
            red_pose_world[2] + RED_HALF_HEIGHT + APPROACH_CLEARANCE,
        ]
    else:
        arm = request.get("arm", "primary")
        if arm not in ("primary", "partner"):
            raise ValueError("arm must be 'primary' or 'partner'")
        target_position_world = request["target_position_world"]
        if len(target_position_world) != 3:
            raise ValueError("target_position_world must contain 3 values")
        obstacles_enabled = bool(request.get("obstacles_enabled", True))
        partner_robot_enabled = bool(
            request.get("partner_robot_enabled", False)
        )
        other_robot_enabled = bool(
            request.get("other_robot_enabled", partner_robot_enabled)
        )
        other_robot_spheres = (
            create_other_robot_spheres(planner, request["other_start"])
            if other_robot_enabled
            else {}
        )
        planner.update_world(
            Scene.create(
                world_for_endpoint(
                    world_template,
                    obstacles_enabled,
                    other_robot_enabled,
                    other_robot_spheres,
                    arm,
                )
            )
        )
    start = JointState.from_position(
        torch.tensor([start_values], device="cuda", dtype=torch.float32),
        joint_names=planner.joint_names,
    )
    if mode == "endpoint" and arm == "partner":
        goal_position_values = [
            PARTNER_BASE_WORLD_X - target_position_world[0],
            -target_position_world[1],
            target_position_world[2] - ROBOT_BASE_WORLD_Z,
        ]
    else:
        goal_position_values = [
            target_position_world[0],
            target_position_world[1],
            target_position_world[2] - ROBOT_BASE_WORLD_Z,
        ]
    goal_position = torch.tensor(
        [goal_position_values],
        device="cuda",
        dtype=torch.float32,
    )
    goal_quaternion = torch.tensor([GOAL_QUATERNION], device="cuda", dtype=torch.float32)
    goal = GoalToolPose.from_poses(
        {"gripper_center": Pose(position=goal_position, quaternion=goal_quaternion)},
        num_goalset=1,
    )
    result = None
    max_attempts = 3 if mode == "endpoint" else 1
    attempts = 0
    for attempts in range(1, max_attempts + 1):
        result = planner.plan_pose(goal, start)
        if result is not None and bool(result.success.any().item()):
            break
    elapsed = time.perf_counter() - started
    if result is None or not bool(result.success.any().item()):
        response = {
            "id": request_id,
            "ok": False,
            "error": "No collision-free trajectory was found",
            "wall_time": elapsed,
            "target_position_world": target_position_world,
            "scene_revision": request.get("scene_revision"),
            "attempts": attempts,
        }
        if mode == "endpoint":
            response["arm"] = arm
            response["obstacles_enabled"] = obstacles_enabled
            response["partner_robot_enabled"] = partner_robot_enabled
        if mode == "red":
            response["red_pose_world"] = red_pose_world
        return response

    trajectory = result.get_interpolated_plan()
    positions = trajectory.position.detach().cpu().reshape(-1, len(planner.joint_names))
    response = {
        "id": request_id,
        "ok": True,
        "joint_names": planner.joint_names,
        "dt": float(planner.trajopt_solver.config.interpolation_dt),
        "positions": positions.tolist(),
        "goal_position_base": goal_position[0].detach().cpu().tolist(),
        "target_position_world": target_position_world,
        "solver_time": float(result.solve_time),
        "wall_time": elapsed,
        "scene_revision": request.get("scene_revision"),
        "attempts": attempts,
    }
    if mode == "endpoint":
        response["arm"] = arm
        response["obstacles_enabled"] = obstacles_enabled
        response["partner_robot_enabled"] = partner_robot_enabled
    if mode == "red":
        response["red_pose_world"] = red_pose_world
    return response


def serve_client(
    connection: socket.socket,
    planner: MotionPlanner,
    world_template: dict[str, Any],
    mode: str,
) -> None:
    peer = connection.getpeername()
    print(f"client connected: {peer}", flush=True)
    with connection, connection.makefile("rwb") as stream:
        while line := stream.readline():
            try:
                request = json.loads(line)
                if request.get("type") == "ping":
                    response = {"id": request.get("id"), "ok": True, "type": "pong"}
                elif request.get("type") == "plan":
                    response = plan_request(
                        planner,
                        world_template,
                        request,
                        mode,
                    )
                    print(
                        f"plan id={request.get('id')} ok={response['ok']} "
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
    print(f"client disconnected: {peer}", flush=True)


def main() -> None:
    args = parse_args()
    robot, world = load_configs(args.mode)
    cfg = MotionPlannerCfg.create(
        robot=robot,
        scene_model=world,
        collision_cache={"cuboid": 8, "sphere": 80},
        interpolation_dt=0.02,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
    )
    planner = MotionPlanner(cfg)
    try:
        if args.mode == "endpoint":
            planner.update_tool_pose_criteria(
                {"gripper_center": ToolPoseCriteria.track_position()}
            )
        print("warming up cuRobo...", flush=True)
        planner.warmup(
            enable_graph=True,
            num_warmup_iterations=args.warmup_iterations,
        )
        print("warmup complete", flush=True)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(2)
            print(f"planner ready: tcp://{args.host}:{args.port}", flush=True)
            while True:
                connection, _ = server.accept()
                serve_client(
                    connection,
                    planner,
                    world,
                    args.mode,
                )
    except KeyboardInterrupt:
        print("planner server stopping", flush=True)
    finally:
        planner.destroy()


if __name__ == "__main__":
    main()
