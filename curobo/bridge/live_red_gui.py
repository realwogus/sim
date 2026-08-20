#!/usr/bin/env python3
"""Interactive MuJoCo GUI driven by a persistent cuRobo planning server."""

from __future__ import annotations

import argparse
import json
import queue
import signal
import socket
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PROJECT_DIR.parent / "mujoco" / "models" / "scenes" / "piper_red_gate.xml"
ARM_JOINTS = tuple(f"piper_joint{i}" for i in range(1, 7))
ROBOT_BASE_WORLD_Z = 0.20
RED_HALF_HEIGHT = 0.036
APPROACH_CLEARANCE = 0.030


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drag red block and replan PiPER with cuRobo")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5561)
    parser.add_argument("--settle-time", type=float, default=0.6)
    parser.add_argument("--move-threshold", type=float, default=0.004)
    return parser.parse_args()


class PlannerWorker(threading.Thread):
    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.requests: queue.Queue[dict | None] = queue.Queue(maxsize=1)
        self.results: queue.Queue[dict] = queue.Queue()

    def submit(self, request: dict) -> None:
        self.requests.put_nowait(request)

    def stop(self) -> None:
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass

    def run(self) -> None:
        try:
            with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                sock.settimeout(None)
                with sock.makefile("rwb") as stream:
                    while request := self.requests.get():
                        stream.write((json.dumps(request) + "\n").encode("utf-8"))
                        stream.flush()
                        line = stream.readline()
                        if not line:
                            raise ConnectionError("cuRobo server closed the connection")
                        self.results.put(json.loads(line))
        except Exception as exc:
            self.results.put({"ok": False, "fatal": True, "error": str(exc)})


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    red_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_freejoint")
    red_qpos_address = model.jnt_qposadr[red_joint_id]
    marker_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "planner_target")
    marker_mocap_id = model.body_mocapid[marker_body_id]
    if min(red_joint_id, marker_body_id, marker_mocap_id) < 0:
        raise ValueError("Scene is missing red_block_freejoint or planner_target")

    worker = PlannerWorker(args.host, args.port)
    worker.start()
    stop = False
    planning = False
    request_id = 0
    active_positions: np.ndarray | None = None
    active_dt = 0.02
    trajectory_started = 0.0
    observed_red = data.qpos[red_qpos_address : red_qpos_address + 3].copy()
    last_change = time.monotonic()
    last_planned_red: np.ndarray | None = None
    requested_red: np.ndarray | None = None

    def request_stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    print("cuRobo live GUI connected")
    print("1) Double-click the red block to select it")
    print("2) Ctrl + right-drag: move vertically")
    print("3) Ctrl + Shift + right-drag: move horizontally")
    print("Release it and wait 0.6 s; planning and playback start automatically.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "overview"
        )
        while viewer.is_running() and not stop:
            frame_started = time.monotonic()
            now = frame_started

            red_position = data.qpos[red_qpos_address : red_qpos_address + 3].copy()
            red_quaternion = data.qpos[
                red_qpos_address + 3 : red_qpos_address + 7
            ].copy()
            if np.linalg.norm(red_position - observed_red) > args.move_threshold:
                observed_red = red_position.copy()
                last_change = now
                active_positions = None
                data.ctrl[:6] = np.array(
                    [data.joint(name).qpos[0] for name in ARM_JOINTS]
                )

            target_world = red_position.copy()
            target_world[2] += RED_HALF_HEIGHT + APPROACH_CLEARANCE
            data.mocap_pos[marker_mocap_id] = target_world

            while not worker.results.empty():
                response = worker.results.get_nowait()
                planning = False
                if response.get("fatal"):
                    raise ConnectionError(response.get("error"))
                response_red = np.asarray(response.get("red_pose_world", [])[:3])
                if response_red.shape != (3,) or np.linalg.norm(response_red - red_position) > 0.01:
                    print("Discarded a stale trajectory because the red block moved again")
                    continue
                if not response.get("ok"):
                    print(
                        f"Planning failed in {response.get('wall_time', 0):.3f}s: "
                        f"{response.get('error')}"
                    )
                    last_planned_red = red_position.copy()
                    continue
                active_positions = np.asarray(response["positions"], dtype=np.float64)
                active_dt = float(response["dt"])
                trajectory_started = now
                last_planned_red = red_position.copy()
                print(
                    f"Trajectory ready: {len(active_positions)} waypoints, "
                    f"solver={response.get('solver_time', 0):.3f}s, "
                    f"wall={response.get('wall_time', 0):.3f}s"
                )

            moved_since_plan = (
                last_planned_red is None
                or np.linalg.norm(red_position - last_planned_red) > args.move_threshold
            )
            if not planning and moved_since_plan and now - last_change >= args.settle_time:
                request_id += 1
                current_arm = [float(data.joint(name).qpos[0]) for name in ARM_JOINTS]
                request_pose = np.concatenate([red_position, red_quaternion])
                worker.submit(
                    {
                        "type": "plan",
                        "id": request_id,
                        "start": current_arm,
                        "red_pose_world": request_pose.tolist(),
                    }
                )
                planning = True
                requested_red = red_position.copy()
                active_positions = None
                data.ctrl[:6] = current_arm
                print(f"Planning to red block at {red_position.tolist()}...")

            if active_positions is not None:
                index = min(
                    (now - trajectory_started) / active_dt,
                    len(active_positions) - 1,
                )
                lo = int(np.floor(index))
                hi = min(lo + 1, len(active_positions) - 1)
                alpha = index - lo
                data.ctrl[:6] = (
                    (1.0 - alpha) * active_positions[lo] + alpha * active_positions[hi]
                )

            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - frame_started)
            if remaining > 0:
                time.sleep(remaining)

    worker.stop()


if __name__ == "__main__":
    main()
