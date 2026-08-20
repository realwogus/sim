#!/usr/bin/env python3
"""Plan a collision-free PiPER TCP motion and save it for MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT = PROJECT_DIR / "robots" / "piper" / "piper.yml"
DEFAULT_WORLD = PROJECT_DIR / "worlds" / "piper_tabletop.yml"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "piper_trajectory.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan a PiPER arm trajectory. Coordinates are relative to the robot base."
    )
    parser.add_argument("--robot", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--no-world", action="store_true")
    parser.add_argument("--start", type=float, nargs=6, metavar=("J1", "J2", "J3", "J4", "J5", "J6"))
    parser.add_argument("--goal-position", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--goal-quaternion",
        type=float,
        nargs=4,
        metavar=("QW", "QX", "QY", "QZ"),
        help="TCP quaternion in wxyz order; defaults to the start orientation",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", action="store_true", help="Run planner warmup before solving")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.robot.is_file():
        raise SystemExit("PiPER config is missing; run: bash scripts/build_piper_config.sh")
    if not args.no_world and not args.world.is_file():
        raise SystemExit(f"World config does not exist: {args.world}")

    scene = None if args.no_world else str(args.world.resolve())
    cfg = MotionPlannerCfg.create(
        robot=str(args.robot.resolve()),
        scene_model=scene,
        collision_cache={"cuboid": 8},
        interpolation_dt=0.02,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
    )
    planner = MotionPlanner(cfg)
    try:
        if args.warmup:
            print("Warming up cuRobo CUDA graphs...")
            planner.warmup(enable_graph=True, num_warmup_iterations=3)

        if args.start is None:
            start_position = planner.default_joint_state.position.clone()
        else:
            start_position = torch.tensor(args.start, device="cuda", dtype=torch.float32)
        start = JointState.from_position(
            start_position.reshape(1, -1), joint_names=planner.joint_names
        )

        start_tcp = planner.compute_kinematics(start).tool_poses.get_link_pose("gripper_center")
        if args.goal_position is None:
            goal_position = start_tcp.position.clone()
            goal_position[:, 2] += 0.05
        else:
            goal_position = torch.tensor(
                [args.goal_position], device="cuda", dtype=torch.float32
            )
        if args.goal_quaternion is None:
            goal_quaternion = start_tcp.quaternion.clone()
        else:
            goal_quaternion = torch.tensor(
                [args.goal_quaternion], device="cuda", dtype=torch.float32
            )

        goal_pose = Pose(position=goal_position, quaternion=goal_quaternion)
        goal = GoalToolPose.from_poses({"gripper_center": goal_pose}, num_goalset=1)
        print(f"start={start.position[0].detach().cpu().tolist()}")
        print(f"goal_position={goal_position[0].detach().cpu().tolist()}")
        print(f"goal_quaternion_wxyz={goal_quaternion[0].detach().cpu().tolist()}")
        result = planner.plan_pose(goal, start)
        if result is None or not bool(result.success.any().item()):
            raise SystemExit("Planning failed. Change the target pose or use --no-world for diagnosis.")

        trajectory = result.get_interpolated_plan()
        # V2 currently returns [batch, seed, time, dof] for this call. Flatten
        # only the leading singleton result dimensions into the time axis.
        positions = trajectory.position.detach().cpu().reshape(-1, len(planner.joint_names))
        payload = {
            "format": "piper-curobo-trajectory-v1",
            "joint_names": planner.joint_names,
            "dt": float(planner.trajopt_solver.config.interpolation_dt),
            "positions": positions.tolist(),
            "start_tcp_position": start_tcp.position[0].detach().cpu().tolist(),
            "goal_tcp_position": goal_position[0].detach().cpu().tolist(),
            "goal_tcp_quaternion_wxyz": goal_quaternion[0].detach().cpu().tolist(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"success=True waypoints={len(payload['positions'])} dt={payload['dt']}")
        print(f"trajectory={args.output.resolve()}")
    finally:
        planner.destroy()


if __name__ == "__main__":
    main()
