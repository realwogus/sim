"""Utilities shared by the PiPER GR00T viewer and policy runner."""

from __future__ import annotations

import mujoco
import numpy as np


CAMERA_NAMES = ("wrist", "right", "left")
ARM_JOINT_NAMES = tuple(f"piper_joint{i}" for i in range(1, 7))
GRIPPER_JOINT_NAME = "piper_joint7"
TASKS = {
    "blue": "Pick up the blue object from the table and place it on the white plate",
    "green": "Pick up the green object from the table and place it on the white plate",
    "red": "Pick up the red object from the table and place it on the white plate",
}

# Rigid camera mount expressed in the piper_link6 coordinate frame. These
# values preserve the calibrated initial view while making both translation
# and rotation follow the wrist exactly.
WRIST_CAMERA_POS_LOCAL = np.array(
    [-0.1033043788, 0.0185832172, 0.0422064735], dtype=np.float64
)
WRIST_CAMERA_QUAT_LOCAL = np.array(
    [0.2690481483, 0.6534896737, -0.6506855338, -0.2777997054], dtype=np.float64
)


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset robot and task objects to the scene's reproducible home keyframe."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise ValueError("piper_gr00t.xml must define the 'home' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    update_wrist_camera(model, data)


def update_wrist_camera(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Apply a rigid wrist-to-camera transform to the mocap camera rig."""
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "piper_link6")
    rig_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_camera_rig")
    if wrist_id < 0 or rig_id < 0:
        raise ValueError("GR00T scene is missing piper_link6 or wrist_camera_rig")
    mocap_id = model.body_mocapid[rig_id]
    if mocap_id < 0:
        raise ValueError("wrist_camera_rig must be a mocap body")

    wrist_rotation = data.xmat[wrist_id].reshape(3, 3)
    data.mocap_pos[mocap_id] = (
        data.xpos[wrist_id] + wrist_rotation @ WRIST_CAMERA_POS_LOCAL
    )
    mujoco.mju_mulQuat(
        data.mocap_quat[mocap_id],
        data.xquat[wrist_id],
        WRIST_CAMERA_QUAT_LOCAL,
    )
    # Refresh body and camera transforms immediately; mocap writes otherwise
    # become visible only on the next physics step.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)


def robot_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Return the 6 arm joints and total gripper opening in training units."""
    arm = np.array([data.joint(name).qpos[0] for name in ARM_JOINT_NAMES], dtype=np.float32)
    # The demonstrations store total jaw opening; the MJCF actuator controls one
    # jaw and mirrors it to the other with an equality constraint.
    gripper = np.array([2.0 * data.joint(GRIPPER_JOINT_NAME).qpos[0]], dtype=np.float32)
    return arm, gripper


def apply_policy_action(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: np.ndarray,
    gripper: np.ndarray,
) -> None:
    """Convert decoded GR00T joint targets to MuJoCo position controls."""
    arm_target = np.asarray(arm, dtype=np.float64).reshape(-1)
    gripper_target = np.asarray(gripper, dtype=np.float64).reshape(-1)
    if arm_target.size != 6 or gripper_target.size != 1:
        raise ValueError("GR00T action must contain arm(6) and gripper(1)")

    data.ctrl[:6] = np.clip(
        arm_target,
        model.actuator_ctrlrange[:6, 0],
        model.actuator_ctrlrange[:6, 1],
    )
    data.ctrl[6] = np.clip(
        0.5 * gripper_target[0],
        model.actuator_ctrlrange[6, 0],
        model.actuator_ctrlrange[6, 1],
    )
