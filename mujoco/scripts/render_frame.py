#!/usr/bin/env python3
import argparse
from pathlib import Path

import mujoco

from piper_mujoco.paths import PIPER_SCENE_XML


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=PIPER_SCENE_XML)
    parser.add_argument("--output", type=Path, default=Path("outputs/piper.png"))
    parser.add_argument("--distance", type=float, default=1.1)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene.expanduser().resolve()))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = (0.0, 0.0, 0.25)
    camera.distance = args.distance
    camera.azimuth = 135
    camera.elevation = -20
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    renderer.close()

    from PIL import Image

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
