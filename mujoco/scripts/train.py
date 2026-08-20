#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml

from piper_mujoco import PiperReachEnv
from piper_mujoco.paths import TRAINING_CONFIG


def main() -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: python -m pip install -e '.[train]'") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_piper_reach"))
    args = parser.parse_args()

    with TRAINING_CONFIG.open() as config_file:
        config = yaml.safe_load(config_file)
    total_timesteps = args.steps or int(config["total_timesteps"])

    env = PiperReachEnv()
    model = PPO(
        config["policy"],
        env,
        learning_rate=float(config["learning_rate"]),
        n_steps=int(config["n_steps"]),
        batch_size=int(config["batch_size"]),
        gamma=float(config["gamma"]),
        seed=int(config["seed"]),
        device=config["device"],
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    env.close()


if __name__ == "__main__":
    main()
