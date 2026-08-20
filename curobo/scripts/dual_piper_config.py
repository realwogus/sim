"""Construct the cuRobo configuration for two fixed-base PiPER arms."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
SINGLE_CONFIG = PROJECT_DIR / "robots" / "piper" / "piper.yml"
DUAL_URDF = PROJECT_DIR / "robots" / "piper" / "piper_dual_arm.urdf"
ARM_PREFIXES = ("primary_", "partner_")
TOOL_FRAMES = ("primary_gripper_center", "partner_gripper_center")
JOINT_NAMES = tuple(
    f"{prefix}joint{joint_index}"
    for prefix in ARM_PREFIXES
    for joint_index in range(1, 7)
)


def _prefixed_mapping(source: dict[str, Any]) -> dict[str, Any]:
    return {
        prefix + name: copy.deepcopy(value)
        for prefix in ARM_PREFIXES
        for name, value in source.items()
    }


def build_dual_robot_config() -> dict[str, Any]:
    """Duplicate the verified 6-DOF collision model into one 12-DOF robot."""
    single = yaml.safe_load(SINGLE_CONFIG.read_text(encoding="utf-8"))
    source = single["kinematics"]
    dual = copy.deepcopy(single)
    kinematics = dual["kinematics"]

    kinematics["urdf_path"] = "/workspace/curobo/robots/piper/piper_dual_arm.urdf"
    kinematics["base_link"] = "world_base"
    kinematics["tool_frames"] = list(TOOL_FRAMES)
    kinematics["collision_link_names"] = [
        prefix + link
        for prefix in ARM_PREFIXES
        for link in source["collision_link_names"]
    ]
    kinematics["collision_spheres"] = _prefixed_mapping(
        source["collision_spheres"]
    )
    kinematics["mesh_link_names"] = [
        prefix + link
        for prefix in ARM_PREFIXES
        for link in source["mesh_link_names"]
    ]
    kinematics["self_collision_buffer"] = _prefixed_mapping(
        source["self_collision_buffer"]
    )
    kinematics["self_collision_ignore"] = {
        prefix + link: [prefix + ignored for ignored in ignored_links]
        for prefix in ARM_PREFIXES
        for link, ignored_links in source["self_collision_ignore"].items()
    }

    source_cspace = source["cspace"]
    cspace = kinematics["cspace"]
    cspace["joint_names"] = list(JOINT_NAMES)
    for key, value in source_cspace.items():
        if key == "joint_names":
            continue
        cspace[key] = copy.deepcopy(value + value) if isinstance(value, list) else value

    return dual
