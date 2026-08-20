#!/usr/bin/env python3
"""Roll out the fine-tuned GR00T PiPER policy in the MuJoCo task scene."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from piper_mujoco.gr00t_client import Gr00tClient
from piper_mujoco.gr00t_task import (
    CAMERA_NAMES,
    TASKS,
    apply_policy_action,
    reset_home,
    robot_state,
    update_wrist_camera,
)
from piper_mujoco.paths import GR00T_SCENE_XML


def observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    instruction: str,
) -> dict:
    images = {}
    for camera in CAMERA_NAMES:
        renderer.update_scene(data, camera=camera)
        images[camera] = renderer.render().copy()[None, None]
    arm, gripper = robot_state(model, data)
    return {
        "video": images,
        "state": {
            "arm": arm[None, None],
            "gripper": gripper[None, None],
        },
        "language": {
            "annotation.human.action.task_description": [[instruction]],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="red")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--execution-horizon", type=int, default=4)
    parser.add_argument("--policy-steps", type=int, default=100)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.execution_horizon <= 16:
        parser.error("--execution-horizon must be between 1 and the trained horizon 16")

    model = mujoco.MjModel.from_xml_path(str(GR00T_SCENE_XML))
    data = mujoco.MjData(model)
    reset_home(model, data)
    instruction = TASKS[args.task]
    control_steps = round((1.0 / 20.0) / model.opt.timestep)

    viewer_context = (
        mujoco.viewer.launch_passive(model, data)
        if not args.headless
        else None
    )
    viewer = viewer_context.__enter__() if viewer_context is not None else None
    try:
        with mujoco.Renderer(model, height=480, width=640) as renderer, Gr00tClient(
            args.host, args.port
        ) as client:
            if not client.ping():
                raise RuntimeError("GR00T policy server did not answer ping")
            client.reset()
            print(f"task={instruction!r}")
            print(f"server=tcp://{args.host}:{args.port} execution_horizon={args.execution_horizon}")

            for policy_step in range(args.policy_steps):
                update_wrist_camera(model, data)
                obs = observation(model, data, renderer, instruction)
                action_chunk, _ = client.get_action(obs)
                arm_chunk = np.asarray(action_chunk["arm"])[0]
                gripper_chunk = np.asarray(action_chunk["gripper"])[0]

                for action_index in range(min(args.execution_horizon, len(arm_chunk))):
                    # Bound abrupt sim targets even if an out-of-distribution frame
                    # causes a poor policy prediction.
                    current = data.ctrl[:6].copy()
                    bounded_arm = np.clip(arm_chunk[action_index], current - 0.15, current + 0.15)
                    apply_policy_action(
                        model,
                        data,
                        bounded_arm,
                        gripper_chunk[action_index],
                    )
                    for _ in range(control_steps):
                        started = time.monotonic()
                        mujoco.mj_step(model, data)
                        update_wrist_camera(model, data)
                        if viewer is not None:
                            if not viewer.is_running():
                                return
                            viewer.sync()
                            remaining = model.opt.timestep - (time.monotonic() - started)
                            if remaining > 0:
                                time.sleep(remaining)

                arm, gripper = robot_state(model, data)
                print(
                    f"policy_step={policy_step:03d} "
                    f"arm={np.round(arm, 3).tolist()} gripper={gripper[0]:.3f}"
                )
    finally:
        if viewer_context is not None:
            viewer_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
