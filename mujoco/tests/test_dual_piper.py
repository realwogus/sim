import mujoco
import numpy as np

from piper_mujoco.paths import PROJECT_ROOT


DUAL_PIPER_XML = PROJECT_ROOT / "models" / "scenes" / "dual_piper.xml"


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
