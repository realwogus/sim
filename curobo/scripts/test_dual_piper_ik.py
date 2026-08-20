#!/usr/bin/env python3
"""GPU smoke test for simultaneous 12-DOF, two-tool PiPER IK."""

from __future__ import annotations

import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria

from dual_piper_config import JOINT_NAMES, TOOL_FRAMES, build_dual_robot_config


def main() -> None:
    robot = build_dual_robot_config()
    solver = InverseKinematics(
        InverseKinematicsCfg.create(
            robot=robot,
            num_seeds=64,
            seed_solver_num_seeds=64,
            use_cuda_graph=False,
        )
    )
    solver.update_tool_pose_criteria(
        {frame: ToolPoseCriteria.track_position() for frame in TOOL_FRAMES}
    )
    device = torch.device("cuda")
    primary_position = torch.tensor([[0.36, 0.08, 0.12]], device=device)
    partner_position = torch.tensor([[0.48, 0.08, 0.12]], device=device)
    primary_quaternion = torch.tensor(
        [[0.53240967, -0.53241307, -0.46533632, -0.46533674]], device=device
    )
    partner_quaternion = torch.tensor(
        [[0.46533674, 0.46533632, -0.53241307, 0.53240967]], device=device
    )
    goal = GoalToolPose.from_poses(
        {
            TOOL_FRAMES[0]: Pose(primary_position, primary_quaternion),
            TOOL_FRAMES[1]: Pose(partner_position, partner_quaternion),
        },
        ordered_tool_frames=list(TOOL_FRAMES),
        num_goalset=1,
    )
    result = solver.solve_pose(goal)
    if not bool(result.success.any().item()):
        raise SystemExit("joint 12-DOF IK failed")
    solution = result.js_solution.position.reshape(-1, len(JOINT_NAMES))[0]
    state = solver.compute_kinematics(
        JointState.from_position(
            result.js_solution.position.reshape(-1, len(JOINT_NAMES))[0:1],
            joint_names=list(JOINT_NAMES),
        )
    )
    print(f"joint_count={len(JOINT_NAMES)}")
    print(f"joint_names={list(JOINT_NAMES)}")
    print(f"solution={solution.detach().cpu().tolist()}")
    for frame in TOOL_FRAMES:
        pose = state.tool_poses.get_link_pose(frame)
        print(f"{frame}_position={pose.position[0].detach().cpu().tolist()}")


if __name__ == "__main__":
    main()
