import mujoco
import numpy as np

from piper_mujoco.gr00t_task import (
    CAMERA_NAMES,
    reset_home,
    robot_state,
    update_wrist_camera,
)
from piper_mujoco.paths import GR00T_SCENE_XML


def test_gr00t_scene_has_expected_policy_interface() -> None:
    model = mujoco.MjModel.from_xml_path(str(GR00T_SCENE_XML))
    data = mujoco.MjData(model)
    reset_home(model, data)

    assert model.nu == 7
    assert model.nq == 36
    assert model.nmocap == 1
    assert all(model.camera(name).id >= 0 for name in CAMERA_NAMES)
    assert all(model.body(f"{color}_block").id >= 0 for color in ("blue", "green", "red", "yellow"))
    assert model.body("white_plate").id >= 0

    arm, gripper = robot_state(model, data)
    np.testing.assert_allclose(arm, 0.0, atol=1e-7)
    np.testing.assert_allclose(gripper, [0.04], atol=1e-7)


def test_gr00t_task_steps_with_finite_state() -> None:
    model = mujoco.MjModel.from_xml_path(str(GR00T_SCENE_XML))
    data = mujoco.MjData(model)
    reset_home(model, data)
    mujoco.mj_step(model, data, nstep=100)

    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()


def test_wrist_camera_is_rigidly_attached() -> None:
    model = mujoco.MjModel.from_xml_path(str(GR00T_SCENE_XML))
    data = mujoco.MjData(model)
    reset_home(model, data)

    def relative_pose() -> tuple[np.ndarray, np.ndarray]:
        wrist_rotation = data.body("piper_link6").xmat.reshape(3, 3)
        camera_rotation = data.camera("wrist").xmat.reshape(3, 3)
        relative_position = wrist_rotation.T @ (
            data.camera("wrist").xpos - data.body("piper_link6").xpos
        )
        relative_rotation = wrist_rotation.T @ camera_rotation
        return relative_position.copy(), relative_rotation.copy()

    initial_position, initial_rotation = relative_pose()
    np.testing.assert_allclose(
        data.camera("wrist").xpos,
        [0.088, 0.020, 0.510],
        atol=1e-7,
    )
    data.joint("piper_joint1").qpos[0] = 0.5
    data.joint("piper_joint2").qpos[0] = 0.8
    data.joint("piper_joint3").qpos[0] = -0.7
    mujoco.mj_forward(model, data)
    update_wrist_camera(model, data)
    moved_position, moved_rotation = relative_pose()

    np.testing.assert_allclose(moved_position, initial_position, atol=1e-7)
    np.testing.assert_allclose(moved_rotation, initial_rotation, atol=1e-7)
