import mujoco
import numpy as np

from piper_mujoco.paths import PIPER_SCENE_XML


def test_piper_model_steps_with_finite_state() -> None:
    model = mujoco.MjModel.from_xml_path(str(PIPER_SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_step(model, data, nstep=100)

    assert model.nq == 8
    assert model.nu == 7
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()

