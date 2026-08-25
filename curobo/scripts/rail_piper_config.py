"""Construct the independent 7-DOF rail-mounted PiPER cuRobo config."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
SINGLE_CONFIG = PROJECT_DIR / "robots" / "piper" / "piper.yml"
RAIL_URDF = PROJECT_DIR / "robots" / "piper_rail" / "piper_rail.urdf"
JOINT_NAMES = ("rail_slide",) + tuple(f"joint{i}" for i in range(1, 7))


def build_rail_robot_config() -> dict[str, Any]:
    """Add one prismatic rail DOF in front of the six PiPER arm joints."""
    config = yaml.safe_load(SINGLE_CONFIG.read_text(encoding="utf-8"))
    kinematics = config["kinematics"]
    kinematics["urdf_path"] = "/workspace/curobo/robots/piper_rail/piper_rail.urdf"
    kinematics["base_link"] = "world_base"
    kinematics["collision_link_names"] = [
        "rail_carriage",
        *kinematics["collision_link_names"],
    ]
    kinematics["mesh_link_names"] = list(kinematics["collision_link_names"])
    kinematics["collision_spheres"]["rail_carriage"] = [
        {
            "center": [x, y, 0.0],
            "radius": 0.04,
        }
        for x in (-0.07, 0.0, 0.07)
        for y in (-0.09, -0.03, 0.03, 0.09)
    ]
    kinematics["self_collision_buffer"]["rail_carriage"] = 0.0
    kinematics["self_collision_ignore"]["rail_carriage"] = ["arm_base", "link1"]
    kinematics["self_collision_ignore"].setdefault("arm_base", []).append(
        "rail_carriage"
    )
    kinematics["self_collision_ignore"].setdefault("link1", []).append(
        "rail_carriage"
    )

    source_cspace = copy.deepcopy(kinematics["cspace"])
    cspace = kinematics["cspace"]
    cspace["joint_names"] = list(JOINT_NAMES)
    cspace["default_joint_position"] = [0.0] * len(JOINT_NAMES)
    for key in (
        "acceleration_scale",
        "cspace_distance_weight",
        "jerk_scale",
        "max_acceleration",
        "null_space_maximum_distance",
        "null_space_weight",
        "velocity_scale",
    ):
        cspace[key] = [1.0, *source_cspace[key]]
    cspace["max_acceleration"][0] = 1.0
    cspace["max_jerk"] = [5.0, *source_cspace["max_jerk"]]
    return config
