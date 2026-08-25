"""Runtime cuRobo fixes and diagnostics used by the rail planner.

This module intentionally lives outside the cuRobo git submodule so a clone of
the parent ``sim`` repository gets the behavior without requiring a cuRobo
fork.  It fixes failed-seed replacement and records compact per-attempt data.
"""

from __future__ import annotations

from typing import Any

import torch

from curobo.motion_planner import MotionPlanner
from curobo.types import GoalToolPose, JointState


def _capture_metrics(result: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if result.metrics is not None:
        costs = result.metrics.costs_and_constraints
        feasible = costs.get_feasible(include_all_hybrid=False, sum_horizon=True)
        if isinstance(feasible, torch.Tensor):
            metrics["feasible_seeds"] = int(torch.count_nonzero(feasible).item())
            metrics["evaluated_seeds"] = int(feasible.numel())

        constraints = {}
        for collection_name in ("constraints", "hybrid_costs_constraints"):
            collection = getattr(costs, collection_name)
            for name, value in zip(collection.names, collection.values):
                if value is None or value.numel() == 0:
                    continue
                constraints[f"{collection_name}/{name}"] = {
                    "max": float(value.max().item()),
                    "positive_count": int(torch.count_nonzero(value > 0.0).item()),
                }
        if constraints:
            metrics["constraints"] = constraints

    if result.interpolated_metrics is not None:
        interpolated = result.interpolated_metrics.costs_and_constraints.get_feasible(
            include_all_hybrid=False,
            sum_horizon=True,
        )
        if isinstance(interpolated, torch.Tensor):
            metrics["interpolated_feasible_seeds"] = int(
                torch.count_nonzero(interpolated).item()
            )
            metrics["interpolated_evaluated_seeds"] = int(interpolated.numel())
    return metrics


class RailMotionPlanner(MotionPlanner):
    """MotionPlanner with JSON-safe diagnostics and repaired failed IK seeds."""

    def __init__(self, config):
        super().__init__(config)
        self.last_plan_diagnostics: list[dict[str, Any]] = []

    @staticmethod
    def _summarize_result(result: Any) -> dict[str, Any]:
        diagnostic: dict[str, Any] = {
            "trajopt_success": int(torch.count_nonzero(result.success).item()),
            "position_tolerance": float(result.position_tolerance),
            "orientation_tolerance": float(result.orientation_tolerance),
        }
        if result.position_error is not None and result.position_error.numel() > 0:
            diagnostic["min_position_error"] = float(result.position_error.min().item())
        if result.rotation_error is not None and result.rotation_error.numel() > 0:
            diagnostic["min_rotation_error"] = float(result.rotation_error.min().item())
        if result.cspace_error is not None and result.cspace_error.numel() > 0:
            diagnostic["min_cspace_error"] = float(result.cspace_error.min().item())

        diagnostic.update(result.debug_info.get("rail_metrics", {}))
        if diagnostic["trajopt_success"] > 0:
            diagnostic["stage"] = "SUCCESS"
        elif diagnostic.get("feasible_seeds", 0) == 0:
            diagnostic["stage"] = "CONSTRAINT"
        elif diagnostic.get("interpolated_feasible_seeds", 1) == 0:
            diagnostic["stage"] = "INTERPOLATION"
        elif diagnostic.get("min_position_error", float("inf")) >= float(
            result.position_tolerance
        ):
            diagnostic["stage"] = "GOAL_ERROR"
        else:
            diagnostic["stage"] = "TRAJOPT"
        return diagnostic

    def _plan_pose_single(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        max_attempts: int,
        enable_graph_attempt: int,
    ):
        trajopt_result = None
        self.last_plan_diagnostics = []
        total_time = 0.0
        solve_time = 0.0
        original_current_state = current_state.clone()
        num_seeds = self.trajopt_solver.config.num_seeds

        for current_attempt in range(max_attempts):
            attempt = {
                "attempt": current_attempt + 1,
                "ik_success": 0,
                "ik_seeds": int(num_seeds),
                "graph_used": False,
                "graph_success": None,
                "trajopt_success": 0,
            }
            current_state = original_current_state.clone()
            ik_result = self.ik_solver.solve_pose(
                goal_tool_poses,
                return_seeds=num_seeds,
                current_state=current_state,
            )
            total_time += ik_result.total_time
            solve_time += ik_result.solve_time

            success_count = torch.count_nonzero(ik_result.success)
            attempt["ik_success"] = int(success_count.item())
            if success_count == 0:
                attempt["stage"] = "IK_SEED"
                self.last_plan_diagnostics.append(attempt)
                continue

            # Boolean chained indexing writes to a temporary tensor.  Clone
            # and assign directly so colliding failed seeds cannot poison the
            # graph planner's complete start/goal batch validation.
            seed_config = ik_result.solution.clone()
            if success_count < num_seeds:
                good_solution = seed_config[ik_result.success][0:1, :].clone()
                seed_config[~ik_result.success] = good_solution

            seed_traj = None
            finetune_attempts = 1
            finetune_dt_scale = 0.55
            if current_attempt >= enable_graph_attempt and self.graph_planner is not None:
                attempt["graph_used"] = True
                graph_seed = self._get_graph_seed_trajectories(current_state, seed_config)
                if graph_seed is None:
                    attempt["graph_success"] = False
                    attempt["stage"] = "GRAPH_SEARCH"
                    self.last_plan_diagnostics.append(attempt)
                    continue
                attempt["graph_success"] = True
                seed_traj = graph_seed
                finetune_attempts = 3
                finetune_dt_scale = 0.75

            trajopt_result = self.trajopt_solver.solve_pose(
                goal_tool_poses,
                current_state,
                seed_config=seed_config,
                seed_traj=seed_traj,
                use_implicit_goal=True,
                finetune_attempts=finetune_attempts,
                finetune_dt_scale=finetune_dt_scale,
            )
            total_time += trajopt_result.total_time
            solve_time += trajopt_result.solve_time
            attempt.update(self._summarize_result(trajopt_result))
            self.last_plan_diagnostics.append(attempt)
            if torch.count_nonzero(trajopt_result.success) > 0:
                break

        if trajopt_result is not None:
            trajopt_result.total_time = total_time
            trajopt_result.solve_time = solve_time
        return trajopt_result


def install_metric_capture() -> None:
    """Preserve feasibility data before cuRobo drops full seed metrics."""
    from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult

    if getattr(TrajOptSolverResult, "_rail_metrics_installed", False):
        return
    original_process_metrics = TrajOptSolverResult._process_metrics

    def process_metrics_with_capture(result):
        original_process_metrics(result)
        result.debug_info["rail_metrics"] = _capture_metrics(result)

    TrajOptSolverResult._process_metrics = process_metrics_with_capture
    TrajOptSolverResult._rail_metrics_installed = True


install_metric_capture()
