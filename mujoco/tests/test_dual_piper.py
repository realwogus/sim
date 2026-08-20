import mujoco
import numpy as np

from piper_mujoco.paths import PROJECT_ROOT


DUAL_PIPER_XML = PROJECT_ROOT / "models" / "scenes" / "dual_piper.xml"
ENDPOINT_SCENE_XML = PROJECT_ROOT / "models" / "scenes" / "piper_endpoint_gate.xml"


def test_dual_piper_layout_and_home_keyframe() -> None:
    model = mujoco.MjModel.from_xml_path(str(DUAL_PIPER_XML))
    data = mujoco.MjData(model)

    assert model.nq == 16
    assert model.nv == 16
    assert model.nu == 14
    assert np.allclose(model.body("left_base_link").pos, (0.0, -0.45, 0.0))
    assert np.allclose(model.body("right_base_link").pos, (0.0, 0.45, 0.0))
    assert np.allclose(model.body("left_base_link").quat, (1.0, 0.0, 0.0, 0.0))
    assert np.allclose(
        np.abs(model.body("right_base_link").quat), (0.0, 0.0, 0.0, 1.0), atol=1e-7
    )

    for prefix in ("left_", "right_"):
        for joint_number in range(1, 9):
            assert model.joint(f"{prefix}joint{joint_number}").id >= 0
        assert model.actuator(f"{prefix}gripper").id >= 0

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert home_id >= 0
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    expected_arm = np.array((0.0, 1.57, -1.3485, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert np.allclose(data.qpos[:8], expected_arm)
    assert np.allclose(data.qpos[8:], expected_arm)

    mujoco.mj_step(model, data, nstep=100)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()


def test_endpoint_scene_starts_and_holds_both_arms_at_zero() -> None:
    model = mujoco.MjModel.from_xml_path(str(ENDPOINT_SCENE_XML))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)

    for prefix in ("piper", "partner"):
        arm_qpos = np.array(
            [data.joint(f"{prefix}_joint{i}").qpos[0] for i in range(1, 7)]
        )
        assert np.allclose(arm_qpos, 0.0)

    mujoco.mj_forward(model, data)
    assert data.ncon == 0

    tcp_offset = np.array((0.0, 0.0, 0.13503))
    tcp_positions = []
    for prefix in ("piper", "partner"):
        link6 = data.body(f"{prefix}_link6")
        tcp_positions.append(
            link6.xpos + link6.xmat.reshape(3, 3) @ tcp_offset
        )
    target_body = model.body("endpoint_target")
    target_position = data.mocap_pos[int(target_body.mocapid[0])]
    np.testing.assert_allclose(
        target_position, np.mean(tcp_positions, axis=0), atol=1e-9
    )

    mujoco.mj_step(model, data, nstep=500)

    for prefix in ("piper", "partner"):
        arm_qpos = np.array(
            [data.joint(f"{prefix}_joint{i}").qpos[0] for i in range(1, 7)]
        )
        assert np.allclose(arm_qpos, 0.0, atol=1e-6)
