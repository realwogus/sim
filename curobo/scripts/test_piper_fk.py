#!/usr/bin/env python3
"""Load the generated PiPER model and print its TCP pose."""

from pathlib import Path

import torch

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_DIR / "robots" / "piper" / "piper.yml"


def main() -> None:
    if not ROBOT_CONFIG.is_file():
        raise SystemExit("PiPER config is missing; run: bash scripts/build_piper_config.sh")
    robot = Kinematics(KinematicsCfg.from_robot_yaml_file(str(ROBOT_CONFIG)))
    q = robot.default_joint_state.position.reshape(1, -1)
    state = robot.compute_kinematics(
        JointState.from_position(q, joint_names=robot.joint_names)
    )
    tcp = state.tool_poses.get_link_pose("gripper_center")
    print(f"joints={robot.joint_names}")
    print(f"q={q[0].detach().cpu().tolist()}")
    print(f"tcp_position_m={tcp.position[0].detach().cpu().tolist()}")
    print(f"tcp_quaternion_wxyz={tcp.quaternion[0].detach().cpu().tolist()}")
    if not torch.isfinite(tcp.position).all():
        raise SystemExit("FK returned a non-finite pose")


if __name__ == "__main__":
    main()
