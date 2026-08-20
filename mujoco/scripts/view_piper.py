#!/usr/bin/env python3
import os
import signal
import threading
import time

import mujoco
import mujoco.viewer

from piper_mujoco.paths import PIPER_SCENE_XML


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(PIPER_SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        if not stop_requested:
            print("\nclosing MuJoCo viewer...")
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
