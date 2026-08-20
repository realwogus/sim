#!/usr/bin/env python3
import argparse
import os
import signal
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from piper_mujoco.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run any MJCF scene in the passive MuJoCo viewer.")
    parser.add_argument(
        "scene",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "models" / "scenes" / "dual_piper.xml",
    )
    parser.add_argument("--keyframe", default="home")
    args = parser.parse_args()

    scene_path = args.scene.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    else:
        mujoco.mj_resetData(model, data)
        print(f"warning: keyframe {args.keyframe!r} not found; using default state")

    print(f"scene={scene_path}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} keyframe={args.keyframe}")

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        if not stop_requested:
            print("\nclosing MuJoCo viewer...")
            # Some NVIDIA GLX stacks can hang in GLFW teardown after the X
            # drawable disappears. Give normal cleanup a chance, then end only
            # this viewer process so the terminal cannot remain wedged.
            force_exit = threading.Timer(2.0, lambda: os._exit(130))
            force_exit.daemon = True
            force_exit.start()
        stop_requested = True

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and not stop_requested:
                step_start = time.monotonic()
                mujoco.mj_step(model, data)
                viewer.sync()
                remaining = model.opt.timestep - (time.monotonic() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    main()
