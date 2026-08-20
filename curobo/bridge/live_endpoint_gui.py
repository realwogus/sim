#!/usr/bin/env python3
"""Drag a non-physical TCP marker and execute cuRobo plans in MuJoCo."""

from __future__ import annotations

import argparse
import json
import queue
import signal
import socket
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from mujoco.glfw import glfw


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PROJECT_DIR.parent / "mujoco" / "models" / "scenes" / "piper_endpoint_gate.xml"
ARM_JOINTS = tuple(f"piper_joint{i}" for i in range(1, 7))
TCP_OFFSET_LINK6 = np.array([0.0, 0.0, 0.13503])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive PiPER TCP target")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5562)
    parser.add_argument(
        "--single-port",
        type=int,
        help="also connect to a single-arm planner and enable F8 mode switching",
    )
    parser.add_argument("--settle-time", type=float, default=0.35)
    parser.add_argument("--move-threshold", type=float, default=0.003)
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="initial TCP target in MuJoCo world coordinates (metres)",
    )
    parser.add_argument(
        "--dual-goal-offset",
        type=float,
        default=0.06,
        help="per-arm X offset from the shared target when two arms are active",
    )
    parser.add_argument(
        "--joint-dual",
        action="store_true",
        help="request one joint 12-DOF trajectory for both arms",
    )
    obstacle_group = parser.add_mutually_exclusive_group()
    obstacle_group.add_argument(
        "--no-obstacles",
        dest="no_obstacles",
        action="store_true",
        help="start with the gate hidden and excluded from collision planning",
    )
    obstacle_group.add_argument(
        "--obstacles",
        dest="no_obstacles",
        action="store_false",
        help="start with the gate visible and included in collision planning",
    )
    parser.add_argument(
        "--cspace-samples",
        type=int,
        default=1200,
        help="number of joint configurations used for the workspace projection",
    )
    cspace_group = parser.add_mutually_exclusive_group()
    cspace_group.add_argument(
        "--hide-cspace",
        dest="hide_cspace",
        action="store_true",
        help="start with the collision-free workspace projection hidden",
    )
    cspace_group.add_argument(
        "--show-cspace",
        dest="hide_cspace",
        action="store_false",
        help="start with the collision-free workspace projection visible",
    )
    sphere_group = parser.add_mutually_exclusive_group()
    sphere_group.add_argument(
        "--hide-cspace-sphere",
        dest="hide_cspace_sphere",
        action="store_true",
        help="start with the spherical reach envelope hidden",
    )
    sphere_group.add_argument(
        "--show-cspace-sphere",
        dest="hide_cspace_sphere",
        action="store_false",
        help="start with the spherical reach envelope visible",
    )
    parser.add_argument(
        "--show-derived-targets",
        action="store_true",
        help="show the cyan and magenta per-arm target markers",
    )
    parser.set_defaults(
        no_obstacles=True,
        hide_cspace=True,
        hide_cspace_sphere=True,
    )
    return parser.parse_args()


def sample_cspace_projection(
    model: mujoco.MjModel, data: mujoco.MjData, sample_count: int
) -> np.ndarray:
    """Project sampled collision-free 6-D joint configurations into TCP XYZ."""
    if sample_count < 1:
        return np.empty((0, 3), dtype=np.float64)

    joint_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINTS
        ]
    )
    qpos_addresses = model.jnt_qposadr[joint_ids]
    limits = model.jnt_range[joint_ids]
    link6_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "piper_link6"
    )
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.mocap_pos[:] = data.mocap_pos
    scratch.mocap_quat[:] = data.mocap_quat

    # Oversample because configurations colliding with the robot, table, or gate
    # are discarded. A fixed seed keeps ON/OFF comparisons visually stable.
    rng = np.random.default_rng(7)
    candidates = rng.uniform(
        limits[:, 0], limits[:, 1], size=(sample_count * 3, len(ARM_JOINTS))
    )
    points: list[np.ndarray] = []
    for q in candidates:
        scratch.qpos[qpos_addresses] = q
        mujoco.mj_forward(model, scratch)
        if scratch.ncon:
            continue
        rotation = scratch.xmat[link6_id].reshape(3, 3)
        points.append(scratch.xpos[link6_id] + rotation @ TCP_OFFSET_LINK6)
        if len(points) == sample_count:
            break
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def draw_cspace_projection(
    viewer,
    points: np.ndarray,
    points_visible: bool,
    sphere_visible: bool,
    sphere_center: np.ndarray,
    trajectory_paths: dict[str, np.ndarray] | None = None,
) -> None:
    """Draw a planned TCP path plus optional workspace visualizations."""
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        for path in (trajectory_paths or {}).values():
            for start, end in zip(path[:-1], path[1:]):
                if scene.ngeom >= scene.maxgeom:
                    break
                if np.linalg.norm(end - start) < 1e-9:
                    continue
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).reshape(-1),
                    np.array([1.0, 0.03, 0.03, 1.0], dtype=np.float32),
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    4.0,
                    np.asarray(start, dtype=np.float64),
                    np.asarray(end, dtype=np.float64),
                )
                scene.ngeom += 1

        if sphere_visible and len(points) and scene.ngeom < scene.maxgeom:
            radius = float(np.max(np.linalg.norm(points - sphere_center, axis=1)))
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, radius, radius]),
                sphere_center,
                np.eye(3).reshape(-1),
                np.array([0.06, 0.48, 1.0, 0.055], dtype=np.float32),
            )
            scene.ngeom += 1

        point_count = (
            min(len(points), scene.maxgeom - scene.ngeom)
            if points_visible
            else 0
        )
        for point_index in range(point_count):
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.004, 0.004, 0.004]),
                points[point_index],
                np.eye(3).reshape(-1),
                np.array([0.08, 0.72, 1.0, 0.22], dtype=np.float32),
            )
            scene.ngeom += 1


def trajectory_tcp_paths(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectories: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert arm joint trajectories to world-frame TCP polylines."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.mocap_pos[:] = data.mocap_pos
    scratch.mocap_quat[:] = data.mocap_quat
    result: dict[str, np.ndarray] = {}
    for arm, positions in trajectories.items():
        prefix = "piper" if arm == "primary" else "partner"
        qpos_addresses = np.array(
            [
                model.jnt_qposadr[
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        f"{prefix}_joint{i}",
                    )
                ]
                for i in range(1, 7)
            ]
        )
        link6_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_link6"
        )
        path = []
        for joint_position in positions:
            scratch.qpos[qpos_addresses] = joint_position
            mujoco.mj_forward(model, scratch)
            rotation = scratch.xmat[link6_id].reshape(3, 3)
            path.append(
                scratch.xpos[link6_id] + rotation @ TCP_OFFSET_LINK6
            )
        result[arm] = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    return result

def first_scene_collision(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectories: dict[str, np.ndarray],
    delays: dict[str, int] | None = None,
    substeps: int = 4,
) -> str | None:
    """Return the first MuJoCo contact along interpolated arm trajectories."""
    if not trajectories:
        return None
    delays = delays or {}
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.mocap_pos[:] = data.mocap_pos
    scratch.mocap_quat[:] = data.mocap_quat
    qpos_addresses = {
        arm: np.array(
            [
                model.jnt_qposadr[
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        f"{'piper' if arm == 'primary' else 'partner'}_joint{i}",
                    )
                ]
                for i in range(1, 7)
            ]
        )
        for arm in trajectories
    }
    horizon = max(
        delays.get(arm, 0) + len(positions)
        for arm, positions in trajectories.items()
    )
    sample_count = max(1, (horizon - 1) * substeps + 1)
    for sample in range(sample_count):
        waypoint_time = sample / substeps
        for arm, positions in trajectories.items():
            index = min(
                max(waypoint_time - delays.get(arm, 0), 0.0),
                len(positions) - 1,
            )
            lo = int(np.floor(index))
            hi = min(lo + 1, len(positions) - 1)
            alpha = index - lo
            scratch.qpos[qpos_addresses[arm]] = (
                (1.0 - alpha) * positions[lo] + alpha * positions[hi]
            )
        mujoco.mj_forward(model, scratch)
        if scratch.ncon:
            contact = scratch.contact[0]
            geom1 = model.geom(contact.geom1).name or f"geom#{contact.geom1}"
            geom2 = model.geom(contact.geom2).name or f"geom#{contact.geom2}"
            return (
                f"{geom1} <-> {geom2} at waypoint {waypoint_time:.2f} "
                f"(penetration={max(0.0, -float(contact.dist)) * 1000.0:.2f} mm)"
            )
    return None


def find_safe_partner_delay_steps(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    primary_positions: np.ndarray,
    partner_positions: np.ndarray,
) -> tuple[int | None, str | None]:
    """Find the earliest stagger with no robot, table, or gate contact."""
    # Try truly simultaneous first, then add 0.1 s (5 waypoint) staggers.
    candidates = list(range(0, len(primary_positions) + 1, 5))
    if candidates[-1] != len(primary_positions):
        candidates.append(len(primary_positions))
    last_collision = None
    for delay in candidates:
        last_collision = first_scene_collision(
            model,
            data,
            {"primary": primary_positions, "partner": partner_positions},
            delays={"partner": delay},
        )
        if last_collision is None:
            return delay, None
    return None, last_collision

class PlannerWorker(threading.Thread):
    def __init__(
        self,
        host: str,
        port: int,
        mode: str,
        results: queue.Queue[dict] | None = None,
    ):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.mode = mode
        self.requests: queue.Queue[dict | None] = queue.Queue(maxsize=1)
        self.results = results if results is not None else queue.Queue()

    def submit(self, request: dict) -> None:
        self.requests.put_nowait(request)

    def stop(self) -> None:
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass

    def run(self) -> None:
        try:
            with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                sock.settimeout(None)
                with sock.makefile("rwb") as stream:
                    while request := self.requests.get():
                        generation = request.pop("_planner_generation", 0)
                        stream.write((json.dumps(request) + "\n").encode("utf-8"))
                        stream.flush()
                        line = stream.readline()
                        if not line:
                            raise ConnectionError("cuRobo server closed the connection")
                        response = json.loads(line)
                        response["_planner_mode"] = self.mode
                        response["_planner_generation"] = generation
                        self.results.put(response)
        except Exception as exc:
            self.results.put(
                {
                    "ok": False,
                    "fatal": True,
                    "error": str(exc),
                    "_planner_mode": self.mode,
                }
            )


def main() -> None:
    args = parse_args()
    if args.single_port is not None and not args.joint_dual:
        raise ValueError("--single-port requires --joint-dual for the primary planner")
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    # Both arms start at q=[0, 0, 0, 0, 0, 0] and their position
    # actuators hold that configuration until the target is edited.
    for prefix in ("piper", "partner"):
        for joint_number in range(1, 7):
            joint_name = f"{prefix}_joint{joint_number}"
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
            )
            if min(joint_id, actuator_id) < 0:
                raise ValueError(f"Scene is missing joint or actuator {joint_name}")
            data.qpos[model.jnt_qposadr[joint_id]] = 0.0
            data.ctrl[actuator_id] = 0.0
    mujoco.mj_forward(model, data)

    shared_target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "endpoint_target"
    )
    shared_target_mocap_id = (
        model.body_mocapid[shared_target_body_id]
        if shared_target_body_id >= 0
        else -1
    )
    if min(shared_target_body_id, shared_target_mocap_id) < 0:
        raise ValueError("Scene is missing the endpoint_target mocap body")

    target_mocap_ids: dict[str, int] = {}
    for arm, body_name in (
        ("primary", "primary_endpoint_goal"),
        ("partner", "partner_endpoint_target"),
    ):
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        mocap_id = model.body_mocapid[body_id] if body_id >= 0 else -1
        if min(body_id, mocap_id) < 0:
            raise ValueError(f"Scene is missing the {body_name} mocap body")
        target_mocap_ids[arm] = mocap_id
    if args.target is not None:
        data.mocap_pos[shared_target_mocap_id] = args.target
    else:
        tcp_positions = []
        for prefix in ("piper", "partner"):
            link6_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_link6"
            )
            if link6_id < 0:
                raise ValueError(f"Scene is missing {prefix}_link6")
            link6_rotation = data.xmat[link6_id].reshape(3, 3)
            tcp_positions.append(
                data.xpos[link6_id] + link6_rotation @ TCP_OFFSET_LINK6
            )
        data.mocap_pos[shared_target_mocap_id] = np.mean(
            tcp_positions, axis=0
        )
    mujoco.mj_forward(model, data)

    gate_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"gate_{part}")
        for part in ("left", "right", "bottom", "top")
    ]
    if any(geom_id < 0 for geom_id in gate_geom_ids):
        raise ValueError("Scene is missing one or more gate geoms")
    gate_contype = model.geom_contype[gate_geom_ids].copy()
    gate_conaffinity = model.geom_conaffinity[gate_geom_ids].copy()
    gate_alpha = model.geom_rgba[gate_geom_ids, 3].copy()

    partner_body_ids = [
        body_id
        for body_id in range(model.nbody)
        if (model.body(body_id).name or "").startswith("partner_")
    ]
    partner_geom_ids = np.flatnonzero(
        np.isin(model.geom_bodyid, partner_body_ids)
    )
    if not len(partner_geom_ids):
        raise ValueError("Scene is missing the dormant partner_ robot")
    partner_contype = model.geom_contype[partner_geom_ids].copy()
    partner_conaffinity = model.geom_conaffinity[partner_geom_ids].copy()
    partner_alpha = model.geom_rgba[partner_geom_ids, 3].copy()
    partner_light_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_LIGHT, "partner_spotlight"
    )
    partner_target_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "partner_endpoint_target_marker",
    )
    if partner_target_geom_id < 0:
        raise ValueError("Scene is missing partner_endpoint_target_marker")
    partner_target_alpha = float(model.geom_rgba[partner_target_geom_id, 3])
    primary_target_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "primary_endpoint_goal_marker",
    )
    if primary_target_geom_id < 0:
        raise ValueError("Scene is missing primary_endpoint_goal_marker")
    primary_target_alpha = float(model.geom_rgba[primary_target_geom_id, 3])

    obstacles_enabled = not args.no_obstacles
    # Keep the facing second arm present from startup in both legacy and
    # joint-12-DOF modes. Legacy mode can still remove it with '-'.
    partner_robot_enabled = True
    partner_motion_enabled = True

    def update_derived_targets() -> None:
        center = data.mocap_pos[shared_target_mocap_id]
        data.mocap_pos[target_mocap_ids["primary"]] = center
        data.mocap_pos[target_mocap_ids["partner"]] = center
        if partner_robot_enabled:
            data.mocap_pos[target_mocap_ids["primary"], 0] -= args.dual_goal_offset
            data.mocap_pos[target_mocap_ids["partner"], 0] += args.dual_goal_offset

    def apply_obstacle_state() -> None:
        if obstacles_enabled:
            model.geom_contype[gate_geom_ids] = gate_contype
            model.geom_conaffinity[gate_geom_ids] = gate_conaffinity
            model.geom_rgba[gate_geom_ids, 3] = gate_alpha
        else:
            model.geom_contype[gate_geom_ids] = 0
            model.geom_conaffinity[gate_geom_ids] = 0
            model.geom_rgba[gate_geom_ids, 3] = 0.0

    def apply_partner_robot_state() -> None:
        if partner_robot_enabled:
            model.geom_contype[partner_geom_ids] = partner_contype
            model.geom_conaffinity[partner_geom_ids] = partner_conaffinity
            model.geom_rgba[partner_geom_ids, 3] = partner_alpha
        else:
            model.geom_contype[partner_geom_ids] = 0
            model.geom_conaffinity[partner_geom_ids] = 0
            model.geom_rgba[partner_geom_ids, 3] = 0.0
        if partner_light_id >= 0:
            model.light_active[partner_light_id] = partner_robot_enabled
        model.geom_rgba[partner_target_geom_id, 3] = (
            partner_target_alpha
            if partner_robot_enabled and args.show_derived_targets
            else 0.0
        )
        model.geom_rgba[primary_target_geom_id, 3] = (
            primary_target_alpha
            if partner_robot_enabled and args.show_derived_targets
            else 0.0
        )

    apply_obstacle_state()
    apply_partner_robot_state()
    update_derived_targets()

    joint_dual_mode = args.joint_dual
    planner_results: queue.Queue[dict] = queue.Queue()
    unavailable_modes: set[str] = set()
    workers = {
        "joint": PlannerWorker(args.host, args.port, "joint", planner_results)
        if args.joint_dual
        else None,
        "sequential": PlannerWorker(
            args.host,
            args.single_port if args.single_port is not None else args.port,
            "sequential",
            planner_results,
        )
        if args.single_port is not None or not args.joint_dual
        else None,
    }
    workers = {mode: worker for mode, worker in workers.items() if worker is not None}
    for planner_worker in workers.values():
        planner_worker.start()
    stop = False
    request_id = 0
    planner_generation = 0
    scene_revision = 0
    planning_arm: str | None = None
    executing_arm: str | None = None
    pending_execution_arm: str | None = None
    active_positions: np.ndarray | None = None
    dual_active_positions: dict[str, np.ndarray] | None = None
    trajectory_preview_paths: dict[str, np.ndarray] = {}
    visualization_dirty = False
    dual_partner_delay_steps = 0
    pending_dual_plans: dict[str, dict] = {}
    ik_status = {
        "primary": "NOT RUN",
        "partner": "NOT RUN",
        "dual": "NOT RUN",
    }
    dual_cycle_active = partner_robot_enabled and not joint_dual_mode
    active_dt = 0.02
    trajectory_started = 0.0
    started_at = time.monotonic()
    planner_available_at = started_at
    preferred_arm: str | None = "primary" if partner_robot_enabled else None
    observed_shared_target = data.mocap_pos[shared_target_mocap_id].copy()
    # An explicit CLI target counts as an initial command; otherwise the two
    # arms remain at zero until the marker is moved in the GUI.
    target_has_been_moved = args.target is not None
    arm_states = {
        "primary": {
            "joints": tuple(f"piper_joint{i}" for i in range(1, 7)),
            "actuators": np.array(
                [
                    mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"piper_joint{i}"
                    )
                    for i in range(1, 7)
                ]
            ),
            "observed": data.mocap_pos[target_mocap_ids["primary"]].copy(),
            "last_change": started_at,
            # Suppress automatic startup planning. The first target edit sets
            # this back to None and starts planning from the zero pose.
            "last_planned": (
                None
                if target_has_been_moved
                else data.mocap_pos[target_mocap_ids["primary"]].copy()
            ),
        },
        "partner": {
            "joints": tuple(f"partner_joint{i}" for i in range(1, 7)),
            "actuators": np.array(
                [
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_ACTUATOR,
                        f"partner_joint{i}",
                    )
                    for i in range(1, 7)
                ]
            ),
            "observed": data.mocap_pos[target_mocap_ids["partner"]].copy(),
            "last_change": started_at,
            "last_planned": (
                None
                if target_has_been_moved
                else data.mocap_pos[target_mocap_ids["partner"]].copy()
            ),
        },
    }

    def joint_values(arm: str) -> np.ndarray:
        return np.array(
            [float(data.joint(name).qpos[0]) for name in arm_states[arm]["joints"]]
        )

    def hold_arm(arm: str) -> None:
        data.ctrl[arm_states[arm]["actuators"]] = joint_values(arm)

    def cancel_active_execution() -> None:
        nonlocal executing_arm, pending_execution_arm
        nonlocal active_positions, dual_active_positions
        nonlocal trajectory_preview_paths, visualization_dirty
        if executing_arm == "dual":
            hold_arm("primary")
            hold_arm("partner")
        elif executing_arm is not None:
            hold_arm(executing_arm)
        executing_arm = None
        pending_execution_arm = None
        active_positions = None
        dual_active_positions = None
        trajectory_preview_paths = {}
        visualization_dirty = True

    def mark_for_replan(arm: str, now: float) -> None:
        arm_states[arm]["last_planned"] = None
        arm_states[arm]["last_change"] = now
        if joint_dual_mode:
            ik_status["dual"] = "WAITING"
        else:
            ik_status[arm] = "WAITING"
    toggle_requested = threading.Event()
    cspace_toggle_requested = threading.Event()
    sphere_toggle_requested = threading.Event()
    robot_add_requested = threading.Event()
    robot_remove_requested = threading.Event()
    partner_motion_toggle_requested = threading.Event()
    target_undo_requested = threading.Event()
    target_redo_requested = threading.Event()
    planner_mode_toggle_requested = threading.Event()
    execute_trajectory_requested = threading.Event()
    last_toggle = 0.0
    cspace_visible = not args.hide_cspace
    sphere_visible = not args.hide_cspace_sphere
    cspace_points = np.empty((0, 3), dtype=np.float64)
    shoulder_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "piper_link1"
    )
    sphere_center = data.xpos[shoulder_id].copy()

    def request_stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)

    def key_callback(keycode: int) -> None:
        window = glfw.get_current_context()
        control_pressed = window is not None and (
            glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
        )
        shift_pressed = window is not None and (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if keycode == glfw.KEY_Z and control_pressed:
            if shift_pressed:
                target_redo_requested.set()
            else:
                target_undo_requested.set()
        elif keycode == glfw.KEY_F8:
            planner_mode_toggle_requested.set()
        elif keycode == glfw.KEY_SPACE:
            execute_trajectory_requested.set()
        elif keycode in (ord("O"), ord("o")):
            toggle_requested.set()
        elif keycode in (ord("C"), ord("c")):
            cspace_toggle_requested.set()
        elif keycode in (ord("P"), ord("p")):
            sphere_toggle_requested.set()
        elif keycode in (ord("+"), ord("="), 334):
            robot_add_requested.set()
        elif keycode in (ord("-"), ord("_"), 333):
            robot_remove_requested.set()
        elif keycode in (ord("/"), 331):
            partner_motion_toggle_requested.set()

    print("cuRobo endpoint GUI connected")
    print("Green marker=shared target for both arms.")
    print("1) Double-click the green shared target")
    print("2) Ctrl + right-drag: move vertically")
    print("3) Ctrl + Shift + right-drag: move horizontally")
    print("Release it; XYZ-only planning starts automatically after 0.35 s.")
    print("Press O to toggle the gate obstacle on/off and replan.")
    print("Press C to show/hide the sampled collision-free C-space projection.")
    print("Press P to show/hide its spherical maximum-reach envelope.")
    print("Press Ctrl+Z to move the shared target back to its previous position.")
    print("Press Ctrl+Shift+Z to restore the target position that was undone.")
    print("A red TCP path appears after planning; press Space to execute it.")
    if args.joint_dual:
        print("Joint 12-DOF mode: both TCP goals are solved in one cuRobo request.")
        print("The partner cannot be removed or paused while joint mode is active.")
        if args.single_port is not None:
            print("Press F8 to switch between joint 12-DOF and sequential 6-DOF planning.")
    else:
        print("Press + to add the facing partner arm; press - to remove it.")
        print("Press / to pause/resume only the partner arm.")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Start in an editable free camera instead of locking to the XML camera.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.fixedcamid = -1
        viewer.cam.lookat[:] = np.array([0.28, 0.08, 0.32])
        viewer.cam.distance = 1.25
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -23.0
        cspace_points = np.empty((0, 3), dtype=np.float64)

        def refresh_planner_mode_overlay() -> None:
            mode_label = (
                "JOINT 12-DOF"
                if joint_dual_mode
                else "SEQUENTIAL 6+6-DOF"
            )
            switch_hint = "F8" if len(workers) > 1 else "disabled"
            if pending_execution_arm is not None:
                trajectory_status = "READY - press SPACE"
            elif executing_arm is not None:
                trajectory_status = "EXECUTING"
            elif planning_arm is not None:
                trajectory_status = "PLANNING"
            else:
                trajectory_status = "IDLE"
            if joint_dual_mode:
                ik_headings = "IK / plan"
                ik_values = ik_status["dual"]
            else:
                ik_headings = "Primary IK / plan\nPartner IK / plan"
                ik_values = (
                    f"{ik_status['primary']}\n"
                    f"{ik_status['partner'] if partner_robot_enabled else 'DISABLED'}"
                )
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f"Planner mode\nSwitch mode\nTrajectory\n{ik_headings}",
                    (
                        f"{mode_label}\n{switch_hint}\n{trajectory_status}\n"
                        f"{ik_values}"
                    ),
                )
            )

        def refresh_cspace_projection() -> None:
            """Only sample the projection when one of its visualizations is enabled."""
            nonlocal cspace_points, visualization_dirty
            if cspace_visible or sphere_visible:
                cspace_points = sample_cspace_projection(
                    model, data, args.cspace_samples
                )
            else:
                cspace_points = np.empty((0, 3), dtype=np.float64)
            draw_cspace_projection(
                viewer,
                cspace_points,
                cspace_visible,
                sphere_visible,
                sphere_center,
                trajectory_preview_paths,
            )
            visualization_dirty = False

        refresh_cspace_projection()
        refresh_planner_mode_overlay()
        if cspace_visible or sphere_visible:
            print(
                f"C-space projection: {len(cspace_points)} collision-free TCP samples"
            )
        else:
            print("C-space visualization: OFF (sampling deferred)")
        target_history: list[np.ndarray] = []
        target_redo_history: list[np.ndarray] = []
        target_edit_origin: np.ndarray | None = None
        target_last_changed_at = started_at

        while viewer.is_running() and not stop:
            frame_started = time.monotonic()
            now = frame_started

            if planner_mode_toggle_requested.is_set():
                planner_mode_toggle_requested.clear()
                if len(workers) < 2:
                    print("Planner mode switching is unavailable in this launch mode")
                else:
                    next_mode = "sequential" if joint_dual_mode else "joint"
                    if next_mode in unavailable_modes:
                        print(f"Cannot switch: {next_mode} planner is unavailable")
                        continue
                    joint_dual_mode = next_mode == "joint"
                    ik_status["primary"] = "NOT RUN"
                    ik_status["partner"] = "NOT RUN"
                    ik_status["dual"] = "NOT RUN"
                    planner_generation += 1
                    planning_arm = None
                    cancel_active_execution()
                    pending_dual_plans.clear()
                    dual_cycle_active = False
                    preferred_arm = "primary"
                    if joint_dual_mode:
                        partner_robot_enabled = True
                        partner_motion_enabled = True
                        apply_partner_robot_state()
                    update_derived_targets()
                    for arm in ("primary", "partner"):
                        # Switching selects the algorithm for the next target edit;
                        # it does not immediately replay the already reached goal.
                        arm_states[arm]["observed"] = data.mocap_pos[
                            target_mocap_ids[arm]
                        ].copy()
                        arm_states[arm]["last_planned"] = data.mocap_pos[
                            target_mocap_ids[arm]
                        ].copy()
                        arm_states[arm]["last_change"] = now
                        hold_arm(arm)
                    mode_label = (
                        "JOINT 12-DOF" if joint_dual_mode else "SEQUENTIAL 6+6-DOF"
                    )
                    refresh_planner_mode_overlay()
                    print(
                        f"Planner mode: {mode_label}; move the target to plan with this mode"
                    )

            if execute_trajectory_requested.is_set():
                execute_trajectory_requested.clear()
                if pending_execution_arm is None:
                    print("Trajectory execution: no preview is ready")
                else:
                    executing_arm = pending_execution_arm
                    pending_execution_arm = None
                    trajectory_started = now
                    visualization_dirty = True
                    print(
                        f"Trajectory execution started: {executing_arm}"
                    )

            requested_partner_state: bool | None = None
            partner_added_this_frame = False
            if robot_add_requested.is_set():
                robot_add_requested.clear()
                requested_partner_state = True
            if robot_remove_requested.is_set():
                robot_remove_requested.clear()
                requested_partner_state = False
            if joint_dual_mode and requested_partner_state is not None:
                if not requested_partner_state:
                    print("Partner removal ignored: joint 12-DOF mode requires both arms")
                requested_partner_state = None
            if (
                requested_partner_state is not None
                and requested_partner_state != partner_robot_enabled
            ):
                partner_robot_enabled = requested_partner_state
                if partner_robot_enabled:
                    partner_added_this_frame = True
                    partner_motion_enabled = True
                    preferred_arm = "primary"
                scene_revision += 1
                apply_partner_robot_state()
                update_derived_targets()
                cancel_active_execution()
                pending_dual_plans.clear()
                dual_cycle_active = partner_robot_enabled
                mark_for_replan("primary", now)
                mark_for_replan("partner", now)
                hold_arm("primary")
                hold_arm("partner")
                refresh_cspace_projection()
                state = "ADDED" if partner_robot_enabled else "REMOVED"
                print(f"Partner robot: {state}; replanning...")

            if partner_motion_toggle_requested.is_set():
                partner_motion_toggle_requested.clear()
                if joint_dual_mode:
                    print("Partner pause ignored: joint 12-DOF mode requires synchronized motion")
                elif partner_robot_enabled:
                    partner_motion_enabled = not partner_motion_enabled
                    if not partner_motion_enabled and (
                        executing_arm in ("partner", "dual")
                        or pending_execution_arm in ("partner", "dual")
                    ):
                        was_dual_execution = (
                            executing_arm == "dual"
                            or pending_execution_arm == "dual"
                        )
                        cancel_active_execution()
                        pending_dual_plans.clear()
                        dual_cycle_active = False
                        if was_dual_execution:
                            mark_for_replan("primary", now)
                        mark_for_replan("partner", now)
                    if partner_motion_enabled:
                        mark_for_replan("partner", now)
                        preferred_arm = "partner"
                    state = "RUNNING" if partner_motion_enabled else "PAUSED"
                    print(f"Partner arm motion: {state}")
                else:
                    print("Partner arm is not present; press + first")

            if toggle_requested.is_set() and now - last_toggle >= 0.25:
                toggle_requested.clear()
                last_toggle = now
                obstacles_enabled = not obstacles_enabled
                scene_revision += 1
                apply_obstacle_state()
                cancel_active_execution()
                pending_dual_plans.clear()
                dual_cycle_active = (
                    not joint_dual_mode
                    and partner_robot_enabled
                    and partner_motion_enabled
                )
                if dual_cycle_active:
                    preferred_arm = "primary"
                mark_for_replan("primary", now)
                mark_for_replan("partner", now)
                hold_arm("primary")
                hold_arm("partner")
                state = "ON" if obstacles_enabled else "OFF"
                refresh_cspace_projection()
                print(f"Gate obstacle: {state}; replanning...")

            if cspace_toggle_requested.is_set():
                cspace_toggle_requested.clear()
                cspace_visible = not cspace_visible
                refresh_cspace_projection()
                print(
                    "C-space projection: "
                    f"{'ON' if cspace_visible else 'OFF'}"
                )

            if sphere_toggle_requested.is_set():
                sphere_toggle_requested.clear()
                sphere_visible = not sphere_visible
                refresh_cspace_projection()
                print(
                    "C-space spherical envelope: "
                    f"{'ON' if sphere_visible else 'OFF'}"
                )

            if target_undo_requested.is_set():
                target_undo_requested.clear()
                current_target = data.mocap_pos[shared_target_mocap_id].copy()
                previous_target = None
                if target_edit_origin is not None:
                    previous_target = target_edit_origin.copy()
                    target_edit_origin = None
                elif target_history:
                    previous_target = target_history.pop()
                if previous_target is None:
                    print("Target undo: no previous position")
                else:
                    target_redo_history.append(current_target)
                    if len(target_redo_history) > 100:
                        del target_redo_history[0]
                    data.mocap_pos[shared_target_mocap_id] = previous_target
                    observed_shared_target = previous_target.copy()
                    target_last_changed_at = now
                    pending_dual_plans.clear()
                    dual_cycle_active = (
                        not joint_dual_mode
                        and partner_robot_enabled
                        and partner_motion_enabled
                    )
                    cancel_active_execution()
                    mark_for_replan("primary", now)
                    mark_for_replan("partner", now)
                    preferred_arm = "primary"
                    update_derived_targets()
                    print(f"Target undo: restored {previous_target.tolist()}")

            if target_redo_requested.is_set():
                target_redo_requested.clear()
                if target_edit_origin is not None:
                    print("Target redo: finish the current target edit first")
                elif not target_redo_history:
                    print("Target redo: no undone position")
                else:
                    current_target = data.mocap_pos[shared_target_mocap_id].copy()
                    next_target = target_redo_history.pop()
                    target_history.append(current_target)
                    if len(target_history) > 100:
                        del target_history[0]
                    data.mocap_pos[shared_target_mocap_id] = next_target
                    observed_shared_target = next_target.copy()
                    target_last_changed_at = now
                    pending_dual_plans.clear()
                    dual_cycle_active = (
                        not joint_dual_mode
                        and partner_robot_enabled
                        and partner_motion_enabled
                    )
                    cancel_active_execution()
                    mark_for_replan("primary", now)
                    mark_for_replan("partner", now)
                    preferred_arm = "primary"
                    update_derived_targets()
                    print(f"Target redo: restored {next_target.tolist()}")

            shared_target = data.mocap_pos[shared_target_mocap_id].copy()
            shared_target_changed = (
                np.linalg.norm(shared_target - observed_shared_target)
                > args.move_threshold
            )
            if shared_target_changed:
                target_has_been_moved = True
                if joint_dual_mode:
                    ik_status["dual"] = "WAITING"
                else:
                    ik_status["primary"] = "WAITING"
                    if partner_robot_enabled and partner_motion_enabled:
                        ik_status["partner"] = "WAITING"
                visualization_dirty = True
                if pending_execution_arm is not None:
                    cancel_active_execution()
                if target_edit_origin is None:
                    target_edit_origin = observed_shared_target.copy()
                    # A new edit branches history, so an older redo chain is
                    # no longer valid.
                    target_redo_history.clear()
                observed_shared_target = shared_target.copy()
                target_last_changed_at = now
                pending_dual_plans.clear()
                dual_cycle_active = (
                    not joint_dual_mode
                    and partner_robot_enabled
                    and partner_motion_enabled
                )
            elif (
                target_edit_origin is not None
                and now - target_last_changed_at >= args.settle_time
            ):
                if np.linalg.norm(shared_target - target_edit_origin) > args.move_threshold:
                    target_history.append(target_edit_origin.copy())
                    if len(target_history) > 100:
                        del target_history[0]
                target_edit_origin = None
            # Derived markers cannot be moved independently: they always follow
            # the one green target and straddle it along the table X axis.
            update_derived_targets()

            for arm in ("primary", "partner"):
                if arm == "partner" and not partner_robot_enabled:
                    continue
                target = data.mocap_pos[target_mocap_ids[arm]].copy()
                state = arm_states[arm]
                if np.linalg.norm(target - state["observed"]) > args.move_threshold:
                    state["observed"] = target.copy()
                    state["last_change"] = now
                    state["last_planned"] = None
                    preferred_arm = arm
                    if executing_arm in (arm, "dual"):
                        cancel_active_execution()
            if shared_target_changed or partner_added_this_frame:
                preferred_arm = "primary"

            while not planner_results.empty():
                response = planner_results.get_nowait()
                response_mode = response.pop("_planner_mode", None)
                response_generation = response.pop("_planner_generation", None)
                active_mode = "joint" if joint_dual_mode else "sequential"
                if response.get("fatal"):
                    unavailable_modes.add(response_mode)
                    if response_mode == "joint":
                        ik_status["dual"] = "ERROR"
                    else:
                        ik_status["primary"] = "ERROR"
                        ik_status["partner"] = "ERROR"
                    visualization_dirty = True
                    if response_mode == active_mode:
                        raise ConnectionError(response.get("error"))
                    print(
                        f"Inactive {response_mode} planner became unavailable: "
                        f"{response.get('error')}"
                    )
                    continue
                if (
                    response_mode != active_mode
                    or response_generation != planner_generation
                ):
                    print(
                        f"Discarded stale {response_mode} planner result after mode switch"
                    )
                    continue
                response_arm = response.get("arm", planning_arm or "primary")
                planning_arm = None
                visualization_dirty = True
                if response_arm == "dual":
                    response_targets = response.get("target_positions_world", {})
                    targets_are_current = all(
                        arm in response_targets
                        and np.asarray(response_targets[arm]).shape == (3,)
                        and np.linalg.norm(
                            np.asarray(response_targets[arm])
                            - data.mocap_pos[target_mocap_ids[arm]]
                        )
                        <= 0.008
                        for arm in ("primary", "partner")
                    )
                    if not targets_are_current:
                        print("Discarded stale joint 12-DOF trajectory: target moved")
                        continue
                    if response.get("scene_revision") != scene_revision:
                        print("Discarded stale joint 12-DOF trajectory: scene changed")
                        continue
                    if response.get("obstacles_enabled", True) != obstacles_enabled:
                        print("Discarded stale joint 12-DOF trajectory: obstacle state changed")
                        continue
                    if not response.get("ok"):
                        ik_status["dual"] = "FAILED"
                        print(
                            "Joint 12-DOF planning failed in "
                            f"{response.get('wall_time', 0):.3f}s: "
                            f"{response.get('error')}"
                        )
                        for arm in ("primary", "partner"):
                            arm_states[arm]["last_planned"] = (
                                data.mocap_pos[target_mocap_ids[arm]].copy()
                            )
                        continue
                    ik_status["dual"] = "SUCCESS"
                    joint_names = response.get("joint_names", [])
                    expected_joint_names = [
                        f"{prefix}_joint{i}"
                        for prefix in ("primary", "partner")
                        for i in range(1, 7)
                    ]
                    if joint_names != expected_joint_names:
                        raise ValueError(
                            "Unexpected joint 12-DOF order: "
                            f"{joint_names}"
                        )
                    combined_positions = np.asarray(
                        response["positions"], dtype=np.float64
                    )
                    if combined_positions.ndim != 2 or combined_positions.shape[1] != 12:
                        raise ValueError(
                            "Joint 12-DOF trajectory must have shape [N, 12]"
                        )
                    primary_positions = combined_positions[:, :6]
                    partner_positions = combined_positions[:, 6:]
                    collision = first_scene_collision(
                        model,
                        data,
                        {
                            "primary": primary_positions,
                            "partner": partner_positions,
                        },
                    )
                    if collision is not None:
                        for arm in ("primary", "partner"):
                            arm_states[arm]["last_planned"] = (
                                data.mocap_pos[target_mocap_ids[arm]].copy()
                            )
                        print(
                            "Rejected unsafe joint 12-DOF trajectory before playback: "
                            f"{collision}"
                        )
                        continue
                    dual_partner_delay_steps = 0
                    dual_active_positions = {
                        "primary": primary_positions,
                        "partner": partner_positions,
                    }
                    active_dt = float(response["dt"])
                    pending_execution_arm = "dual"
                    executing_arm = None
                    active_positions = None
                    trajectory_preview_paths = trajectory_tcp_paths(
                        model, data, dual_active_positions
                    )
                    visualization_dirty = True
                    hold_arm("primary")
                    hold_arm("partner")
                    for arm in ("primary", "partner"):
                        arm_states[arm]["last_planned"] = (
                            data.mocap_pos[target_mocap_ids[arm]].copy()
                        )
                    print(
                        "Joint 12-DOF trajectory preview ready: "
                        f"{len(combined_positions)} waypoints, "
                        f"solver={response.get('solver_time', 0):.3f}s, "
                        f"wall={response.get('wall_time', 0):.3f}s; "
                        "press Space to execute"
                    )
                    continue
                if response_arm == "partner" and (
                    not partner_robot_enabled or not partner_motion_enabled
                ):
                    print("Discarded partner trajectory: partner is paused or removed")
                    continue
                target = data.mocap_pos[target_mocap_ids[response_arm]].copy()
                response_target = np.asarray(response.get("target_position_world", []))
                if response_target.shape != (3,) or np.linalg.norm(response_target - target) > 0.008:
                    print("Discarded stale trajectory: the target moved again")
                    continue
                if response.get("scene_revision") != scene_revision:
                    print("Discarded stale trajectory: scene changed")
                    continue
                if response.get("obstacles_enabled", True) != obstacles_enabled:
                    print("Discarded stale trajectory: obstacle state changed")
                    continue
                if response.get("partner_robot_enabled", False) != partner_robot_enabled:
                    print("Discarded stale trajectory: partner robot state changed")
                    continue
                if not response.get("ok"):
                    ik_status[response_arm] = "FAILED"
                    print(
                        f"{response_arm} planning failed in "
                        f"{response.get('wall_time', 0):.3f}s: "
                        f"{response.get('error')}"
                    )
                    arm_states[response_arm]["last_planned"] = target.copy()
                    if dual_cycle_active:
                        dual_cycle_active = False
                        pending_dual_plans.clear()
                        print("Coordinated dual-arm move cancelled")
                    continue
                ik_status[response_arm] = "SUCCESS"
                response_positions = np.asarray(
                    response["positions"], dtype=np.float64
                )
                arm_states[response_arm]["last_planned"] = target.copy()
                if dual_cycle_active:
                    pending_dual_plans[response_arm] = {
                        "positions": response_positions,
                        "dt": float(response["dt"]),
                    }
                    print(
                        f"{response_arm} coordinated plan ready: "
                        f"{len(response_positions)} waypoints"
                    )
                    if len(pending_dual_plans) < 2:
                        preferred_arm = (
                            "partner" if response_arm == "primary" else "primary"
                        )
                        continue
                    primary_positions = pending_dual_plans["primary"]["positions"]
                    partner_positions = pending_dual_plans["partner"]["positions"]
                    safe_delay, collision = find_safe_partner_delay_steps(
                        model,
                        data,
                        primary_positions,
                        partner_positions,
                    )
                    if safe_delay is None:
                        dual_cycle_active = False
                        pending_dual_plans.clear()
                        print(
                            "Dual-arm execution cancelled: no collision-free "
                            f"timing was found ({collision})"
                        )
                        continue
                    dual_partner_delay_steps = safe_delay
                    dual_active_positions = {
                        "primary": primary_positions,
                        "partner": partner_positions,
                    }
                    active_dt = float(pending_dual_plans["primary"]["dt"])
                    pending_execution_arm = "dual"
                    executing_arm = None
                    active_positions = None
                    trajectory_preview_paths = trajectory_tcp_paths(
                        model, data, dual_active_positions
                    )
                    visualization_dirty = True
                    hold_arm("primary")
                    hold_arm("partner")
                    dual_cycle_active = False
                    pending_dual_plans.clear()
                    print(
                        "Dual-arm trajectory preview ready: "
                        f"partner delay={dual_partner_delay_steps * active_dt:.2f}s; "
                        "press Space to execute"
                    )
                    continue

                collision = first_scene_collision(
                    model,
                    data,
                    {response_arm: response_positions},
                )
                if collision is not None:
                    arm_states[response_arm]["last_planned"] = target.copy()
                    print(
                        f"Rejected unsafe {response_arm} trajectory before playback: "
                        f"{collision}"
                    )
                    continue

                active_positions = response_positions
                active_dt = float(response["dt"])
                pending_execution_arm = response_arm
                executing_arm = None
                trajectory_preview_paths = trajectory_tcp_paths(
                    model, data, {response_arm: response_positions}
                )
                visualization_dirty = True
                hold_arm(response_arm)
                print(
                    f"{response_arm} trajectory preview ready: "
                    f"{len(active_positions)} waypoints, "
                    f"attempts={response.get('attempts', 1)}, "
                    f"solver={response.get('solver_time', 0):.3f}s, "
                    f"wall={response.get('wall_time', 0):.3f}s; "
                    "press Space to execute"
                )

            if executing_arm == "dual" and dual_active_positions is not None:
                elapsed_steps = (now - trajectory_started) / active_dt
                for arm, delay_steps in (
                    ("primary", 0),
                    ("partner", dual_partner_delay_steps),
                ):
                    positions = dual_active_positions[arm]
                    index = min(
                        max(elapsed_steps - delay_steps, 0.0),
                        len(positions) - 1,
                    )
                    lo = int(np.floor(index))
                    hi = min(lo + 1, len(positions) - 1)
                    alpha = index - lo
                    data.ctrl[arm_states[arm]["actuators"]] = (
                        (1.0 - alpha) * positions[lo] + alpha * positions[hi]
                    )
                dual_horizon = max(
                    len(dual_active_positions["primary"]),
                    dual_partner_delay_steps
                    + len(dual_active_positions["partner"]),
                )
                if elapsed_steps >= dual_horizon - 1:
                    print("dual trajectory complete")
                    executing_arm = None
                    dual_active_positions = None
                    active_positions = None
                    trajectory_preview_paths = {}
                    visualization_dirty = True
                    planner_available_at = now + 1.0
            elif executing_arm is not None and active_positions is not None:
                index = min(
                    (now - trajectory_started) / active_dt,
                    len(active_positions) - 1,
                )
                lo = int(np.floor(index))
                hi = min(lo + 1, len(active_positions) - 1)
                alpha = index - lo
                data.ctrl[arm_states[executing_arm]["actuators"]] = (
                    (1.0 - alpha) * active_positions[lo] + alpha * active_positions[hi]
                )
                if index >= len(active_positions) - 1:
                    print(f"{executing_arm} trajectory complete")
                    executing_arm = None
                    active_positions = None
                    trajectory_preview_paths = {}
                    visualization_dirty = True
                    # Let the position actuators settle before freezing this arm
                    # as an obstacle for the other arm's cuRobo request.
                    planner_available_at = now + 1.0

            if (
                target_has_been_moved
                and joint_dual_mode
                and planning_arm is None
                and executing_arm is None
                and pending_execution_arm is None
                and now >= planner_available_at
            ):
                targets = {
                    arm: data.mocap_pos[target_mocap_ids[arm]].copy()
                    for arm in ("primary", "partner")
                }
                needs_plan = any(
                    arm_states[arm]["last_planned"] is None
                    or np.linalg.norm(
                        targets[arm] - arm_states[arm]["last_planned"]
                    )
                    > args.move_threshold
                    for arm in ("primary", "partner")
                )
                targets_settled = all(
                    now - arm_states[arm]["last_change"] >= args.settle_time
                    for arm in ("primary", "partner")
                )
                if needs_plan and targets_settled:
                    request_id += 1
                    start = np.concatenate(
                        (joint_values("primary"), joint_values("partner"))
                    )
                    workers["joint"].submit(
                        {
                            "type": "plan",
                            "id": request_id,
                            "_planner_generation": planner_generation,
                            "start": start.tolist(),
                            "target_positions_world": {
                                arm: target.tolist() for arm, target in targets.items()
                            },
                            "obstacles_enabled": obstacles_enabled,
                            "scene_revision": scene_revision,
                        }
                    )
                    planning_arm = "dual"
                    ik_status["dual"] = "SOLVING"
                    visualization_dirty = True
                    hold_arm("primary")
                    hold_arm("partner")
                    print(
                        "Planning joint 12-DOF TCP positions: "
                        f"primary={targets['primary'].tolist()}, "
                        f"partner={targets['partner'].tolist()}..."
                    )

            if (
                target_has_been_moved
                and not joint_dual_mode
                and planning_arm is None
                and executing_arm is None
                and pending_execution_arm is None
                and now >= planner_available_at
            ):
                arm_order = (
                    (preferred_arm, "partner" if preferred_arm == "primary" else "primary")
                    if preferred_arm in ("primary", "partner")
                    else ("primary", "partner")
                )
                for arm in arm_order:
                    if arm == "partner" and (
                        not partner_robot_enabled or not partner_motion_enabled
                    ):
                        continue
                    state = arm_states[arm]
                    target = data.mocap_pos[target_mocap_ids[arm]].copy()
                    last_planned = state["last_planned"]
                    needs_plan = (
                        last_planned is None
                        or np.linalg.norm(target - last_planned) > args.move_threshold
                    )
                    if not needs_plan or now - state["last_change"] < args.settle_time:
                        continue
                    other_arm = "partner" if arm == "primary" else "primary"
                    start = joint_values(arm).tolist()
                    if other_arm in pending_dual_plans:
                        other_start = pending_dual_plans[other_arm]["positions"][-1].tolist()
                    else:
                        other_start = joint_values(other_arm).tolist()
                    request_id += 1
                    workers["sequential"].submit(
                        {
                            "type": "plan",
                            "id": request_id,
                            "_planner_generation": planner_generation,
                            "arm": arm,
                            "start": start,
                            "other_start": other_start,
                            "other_robot_enabled": (
                                partner_robot_enabled if arm == "primary" else True
                            ),
                            "target_position_world": target.tolist(),
                            "obstacles_enabled": obstacles_enabled,
                            "partner_robot_enabled": partner_robot_enabled,
                            "scene_revision": scene_revision,
                        }
                    )
                    planning_arm = arm
                    ik_status[arm] = "SOLVING"
                    visualization_dirty = True
                    preferred_arm = None
                    hold_arm(arm)
                    print(f"Planning {arm} TCP position {target.tolist()}...")
                    break

            if visualization_dirty:
                refresh_cspace_projection()
                refresh_planner_mode_overlay()

            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - frame_started)
            if remaining > 0:
                time.sleep(remaining)

    for planner_worker in workers.values():
        planner_worker.stop()


if __name__ == "__main__":
    main()
