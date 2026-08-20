from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from piper_mujoco.paths import PIPER_SCENE_XML, SIMULATION_CONFIG


class PiperReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """Position-control reaching task for the AgileX PiPER arm."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None) -> None:
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        with SIMULATION_CONFIG.open() as config_file:
            config = yaml.safe_load(config_file)

        sim_cfg = config["simulation"]
        task_cfg = config["task"]
        self.frame_skip = int(sim_cfg["frame_skip"])
        self.max_episode_steps = int(sim_cfg["episode_steps"])
        self.action_scale = np.array(
            [float(sim_cfg["arm_action_scale"])] * 6
            + [float(sim_cfg["gripper_action_scale"])],
            dtype=np.float64,
        )
        self.target_low = np.asarray(task_cfg["target_low"], dtype=np.float64)
        self.target_high = np.asarray(task_cfg["target_high"], dtype=np.float64)
        self.success_distance = float(task_cfg["success_distance"])

        self.model = mujoco.MjModel.from_xml_path(str(PIPER_SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer: mujoco.Renderer | None = None
        self._step_count = 0
        self.target = np.zeros(3, dtype=np.float64)
        self._ee_body_id = self.model.body("link6").id

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        # Constraint solvers can exceed a hard joint limit by tiny numerical
        # tolerances, so keep the declared observation bounds slightly wider.
        qpos_low = self.model.jnt_range[:, 0] - 1e-3
        qpos_high = self.model.jnt_range[:, 1] + 1e-3
        velocity_limit = np.full(self.model.nv, 100.0)
        workspace_limit = np.full(3, 10.0)
        obs_low = np.concatenate(
            (qpos_low, -velocity_limit, -workspace_limit, self.target_low)
        )
        obs_high = np.concatenate(
            (qpos_high, velocity_limit, workspace_limit, self.target_high)
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float64
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            (
                self.data.qpos,
                self.data.qvel,
                self.data.xpos[self._ee_body_id],
                self.target,
            )
        ).astype(np.float64, copy=True)

    def _distance(self) -> float:
        return float(np.linalg.norm(self.data.xpos[self._ee_body_id] - self.target))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.target = self.np_random.uniform(self.target_low, self.target_high)
        self._step_count = 0
        mujoco.mj_forward(self.model, self.data)
        distance = self._distance()
        return self._get_obs(), {"distance": distance, "is_success": distance < self.success_distance}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}")

        desired = self.data.ctrl + np.clip(action, -1.0, 1.0) * self.action_scale
        self.data.ctrl[:] = np.clip(
            desired,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._step_count += 1

        distance = self._distance()
        success = distance < self.success_distance
        reward = -distance + (1.0 if success else 0.0) - 1e-3 * float(np.square(action).sum())
        terminated = bool(success)
        truncated = self._step_count >= self.max_episode_steps
        info = {"distance": distance, "is_success": success}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = (0.0, 0.0, 0.25)
        camera.distance = 1.1
        camera.azimuth = 135
        camera.elevation = -20
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
