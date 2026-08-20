import numpy as np
from gymnasium.utils.env_checker import check_env

from piper_mujoco import PiperReachEnv


def test_reach_env_contract() -> None:
    env = PiperReachEnv()
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=7)
    next_observation, reward, terminated, truncated, next_info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert env.observation_space.contains(observation)
    assert env.observation_space.contains(next_observation)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance" in info and "distance" in next_info
    env.close()

