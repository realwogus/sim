#!/usr/bin/env python3
"""Render the three policy input cameras for calibration."""

import argparse
from pathlib import Path

import mujoco
from PIL import Image

from piper_mujoco.gr00t_task import CAMERA_NAMES, reset_home
from piper_mujoco.paths import GR00T_SCENE_XML


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/gr00t_preview"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(GR00T_SCENE_XML))
    data = mujoco.MjData(model)
    reset_home(model, data)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for camera in CAMERA_NAMES:
            renderer.update_scene(data, camera=camera)
            path = args.output / f"{camera}.png"
            Image.fromarray(renderer.render()).save(path)
            print(path)


if __name__ == "__main__":
    main()
