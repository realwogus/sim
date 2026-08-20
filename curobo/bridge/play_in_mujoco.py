#!/usr/bin/env python3
"""Replay a cuRobo PiPER trajectory in the existing MuJoCo task scene."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
MUJOCO_DIR = WORKSPACE_DIR / "mujoco"
DEFAULT_SCENE = MUJOCO_DIR / "models" / "scenes" / "piper_gr00t.xml"
DEFAULT_TRAJECTORY = PROJECT_DIR / "outputs" / "piper_trajectory.json"
ARM_JOINTS = tuple(f"piper_joint{i}" for i in range(1, 7))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a cuRobo trajectory in MuJoCo")
    parser.add_argument("trajectory", type=Path, nargs="?", default=DEFAULT_TRAJECTORY)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--gripper-opening", type=float, default=0.04)
    parser.add_argument("--hold", type=float, default=1.0, help="Seconds to hold the final pose")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--report-target",
        help="MuJoCo body name used to report final gripper-center distance",
    )
    return parser.parse_args()


def load_trajectory(path: Path) -> tuple[np.ndarray, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "piper-curobo-trajectory-v1":
        raise ValueError(f"Unsupported trajectory format in {path}")
    if payload.get("joint_names") != [f"joint{i}" for i in range(1, 7)]:
        raise ValueError(f"Unexpected joint order: {payload.get('joint_names')}")
    positions = np.asarray(payload["positions"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 6 or len(positions) < 2:
        raise ValueError(f"Trajectory must have shape (N, 6), got {positions.shape}")
    return positions, float(payload["dt"])


def main() -> None:
    args = parse_args()
    positions, trajectory_dt = load_trajectory(args.trajectory)
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)

    for name, value in zip(ARM_JOINTS, positions[0]):
        data.joint(name).qpos[0] = value
    data.ctrl[:6] = positions[0]
    data.ctrl[6] = np.clip(
        0.5 * args.gripper_opening,
        model.actuator_ctrlrange[6, 0],
        model.actuator_ctrlrange[6, 1],
    )
    mujoco.mj_forward(model, data)

    duration = (len(positions) - 1) * trajectory_dt
    stop = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)

    def step(elapsed: float) -> None:
        index = min(elapsed / trajectory_dt, len(positions) - 1)
        lo = int(np.floor(index))
        hi = min(lo + 1, len(positions) - 1)
        alpha = index - lo
        data.ctrl[:6] = (1.0 - alpha) * positions[lo] + alpha * positions[hi]
        mujoco.mj_step(model, data)

    print(f"scene={args.scene.resolve()}")
    print(f"trajectory={args.trajectory.resolve()} waypoints={len(positions)} duration={duration:.3f}s")

    def report_target() -> None:
        if not args.report_target:
            return
        target_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, args.report_target
        )
        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "piper_link7")
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "piper_link8")
        if min(target_id, left_id, right_id) < 0:
            raise ValueError("Target or PiPER finger body is missing from the scene")
        gripper_center = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        target_position = data.xpos[target_id].copy()
        distance = np.linalg.norm(gripper_center - target_position)
        print(f"gripper_center_world={gripper_center.tolist()}")
        print(f"{args.report_target}_world={target_position.tolist()}")
        print(f"center_distance={distance:.4f}m")

    if args.headless:
        elapsed = 0.0
        while elapsed < duration + args.hold:
            step(elapsed)
            elapsed += model.opt.timestep
        print(f"final_arm_q={np.array([data.joint(n).qpos[0] for n in ARM_JOINTS]).tolist()}")
        report_target()
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        started = time.monotonic()
        while viewer.is_running() and not stop:
            frame_started = time.monotonic()
            elapsed = min(time.monotonic() - started, duration)
            step(elapsed)
            viewer.sync()
            if time.monotonic() - started >= duration + args.hold:
                break
            remaining = model.opt.timestep - (time.monotonic() - frame_started)
            if remaining > 0:
                time.sleep(remaining)
    report_target()


if __name__ == "__main__":
    main()
