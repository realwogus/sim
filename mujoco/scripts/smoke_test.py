#!/usr/bin/env python3
import math

import mujoco
import numpy as np

from piper_mujoco.paths import PIPER_SCENE_XML


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(PIPER_SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    for _ in range(1000):
        mujoco.mj_step(model, data)

    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("Simulation produced non-finite state")

    print(f"MuJoCo {mujoco.__version__}")
    print(f"model=piper_scene nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"sim_time={data.time:.3f}s")
    print(f"qpos_norm={math.sqrt(float(np.square(data.qpos).sum())):.6f}")


if __name__ == "__main__":
    main()
