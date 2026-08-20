#!/usr/bin/env python3
import argparse

import numpy as np

from piper_mujoco import PiperReachEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = PiperReachEnv()
    observation, info = env.reset(seed=args.seed)
    steps = max(1, int(args.seconds / (env.model.opt.timestep * env.frame_skip)))
    rng = np.random.default_rng(args.seed)
    total_reward = 0.0

    for _ in range(steps):
        action = rng.uniform(-0.25, 0.25, size=env.action_space.shape)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            observation, info = env.reset()

    print(f"steps={steps} observation_shape={observation.shape}")
    print(f"distance={info['distance']:.6f} total_reward={total_reward:.6f}")
    env.close()


if __name__ == "__main__":
    main()

