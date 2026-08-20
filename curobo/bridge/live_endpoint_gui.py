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


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PROJECT_DIR.parent / "mujoco" / "models" / "scenes" / "piper_endpoint_gate.xml"
ARM_JOINTS = tuple(f"piper_joint{i}" for i in range(1, 7))
TCP_OFFSET_LINK6 = np.array([0.0, 0.0, 0.13503])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive PiPER TCP target")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5562)
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
        "--no-obstacles",
        action="store_true",
        help="start with the gate hidden and excluded from collision planning",
    )
    parser.add_argument(
        "--cspace-samples",
        type=int,
        default=1200,
        help="number of joint configurations used for the workspace projection",
    )
    parser.add_argument(
        "--hide-cspace",
        action="store_true",
        help="start with the collision-free workspace projection hidden",
    )
    parser.add_argument(
        "--hide-cspace-sphere",
        action="store_true",
        help="start with the spherical reach envelope hidden",
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
) -> None:
    """Draw free TCP samples and their spherical maximum-reach envelope."""
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
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


def find_safe_partner_delay_steps(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    primary_positions: np.ndarray,
    partner_positions: np.ndarray,
) -> int | None:
    """Find the earliest stagger with no MuJoCo inter-arm contact."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.mocap_pos[:] = data.mocap_pos
    scratch.mocap_quat[:] = data.mocap_quat
    qpos_addresses = {
        prefix: np.array(
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
        for prefix in ("piper", "partner")
    }

    # Try truly simultaneous first, then add 0.1 s (5 waypoint) staggers.
    candidates = list(range(0, len(primary_positions) + 1, 5))
    if candidates[-1] != len(primary_positions):
        candidates.append(len(primary_positions))
    for delay in candidates:
        collision = False
        horizon = max(len(primary_positions), delay + len(partner_positions))
        for step in range(horizon):
            p_index = min(step, len(primary_positions) - 1)
            r_index = min(max(step - delay, 0), len(partner_positions) - 1)
            scratch.qpos[qpos_addresses["piper"]] = primary_positions[p_index]
            scratch.qpos[qpos_addresses["partner"]] = partner_positions[r_index]
            mujoco.mj_forward(model, scratch)
            for contact in scratch.contact[: scratch.ncon]:
                body1 = model.body(model.geom_bodyid[contact.geom1]).name or ""
                body2 = model.body(model.geom_bodyid[contact.geom2]).name or ""
                if (
                    body1.startswith("piper_")
                    and body2.startswith("partner_")
                ) or (
                    body2.startswith("piper_")
                    and body1.startswith("partner_")
                ):
                    collision = True
                    break
            if collision:
                break
        if not collision:
            return delay
    return None

class PlannerWorker(threading.Thread):
    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.requests: queue.Queue[dict | None] = queue.Queue(maxsize=1)
        self.results: queue.Queue[dict] = queue.Queue()

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
                        stream.write((json.dumps(request) + "\n").encode("utf-8"))
                        stream.flush()
                        line = stream.readline()
                        if not line:
                            raise ConnectionError("cuRobo server closed the connection")
                        self.results.put(json.loads(line))
        except Exception as exc:
            self.results.put({"ok": False, "fatal": True, "error": str(exc)})


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
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
    partner_robot_enabled = False
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
            partner_target_alpha if partner_robot_enabled else 0.0
        )
        model.geom_rgba[primary_target_geom_id, 3] = (
            primary_target_alpha if partner_robot_enabled else 0.0
        )

    apply_obstacle_state()
    apply_partner_robot_state()
    update_derived_targets()

    worker = PlannerWorker(args.host, args.port)
    worker.start()
    stop = False
    request_id = 0
    scene_revision = 0
    planning_arm: str | None = None
    executing_arm: str | None = None
    active_positions: np.ndarray | None = None
    dual_active_positions: dict[str, np.ndarray] | None = None
    dual_partner_delay_steps = 0
    pending_dual_plans: dict[str, dict] = {}
    dual_cycle_active = False
    active_dt = 0.02
    trajectory_started = 0.0
    started_at = time.monotonic()
    planner_available_at = started_at
    preferred_arm: str | None = None
    observed_shared_target = data.mocap_pos[shared_target_mocap_id].copy()
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
            "last_planned": None,
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
            "last_planned": None,
        },
    }

    def joint_values(arm: str) -> np.ndarray:
        return np.array(
            [float(data.joint(name).qpos[0]) for name in arm_states[arm]["joints"]]
        )

    def hold_arm(arm: str) -> None:
        data.ctrl[arm_states[arm]["actuators"]] = joint_values(arm)

    def cancel_active_execution() -> None:
        nonlocal executing_arm, active_positions, dual_active_positions
        if executing_arm == "dual":
            hold_arm("primary")
            hold_arm("partner")
        elif executing_arm is not None:
            hold_arm(executing_arm)
        executing_arm = None
        active_positions = None
        dual_active_positions = None

    def mark_for_replan(arm: str, now: float) -> None:
        arm_states[arm]["last_planned"] = None
        arm_states[arm]["last_change"] = now
    toggle_requested = threading.Event()
    cspace_toggle_requested = threading.Event()
    sphere_toggle_requested = threading.Event()
    robot_add_requested = threading.Event()
    robot_remove_requested = threading.Event()
    partner_motion_toggle_requested = threading.Event()
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
        if keycode in (ord("O"), ord("o")):
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
    print("Green=shared target; cyan/magenta=derived collision-safe goals.")
    print("1) Double-click the green shared target")
    print("2) Ctrl + right-drag: move vertically")
    print("3) Ctrl + Shift + right-drag: move horizontally")
    print("Release it; XYZ-only planning starts automatically after 0.35 s.")
    print("Press O to toggle the gate obstacle on/off and replan.")
    print("Press C to show/hide the sampled collision-free C-space projection.")
    print("Press P to show/hide its spherical maximum-reach envelope.")
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
        cspace_points = sample_cspace_projection(model, data, args.cspace_samples)
        draw_cspace_projection(
            viewer,
            cspace_points,
            cspace_visible,
            sphere_visible,
            sphere_center,
        )
        print(
            f"C-space projection: {len(cspace_points)} collision-free TCP samples "
            f"({'visible' if cspace_visible else 'hidden'})"
        )
        while viewer.is_running() and not stop:
            frame_started = time.monotonic()
            now = frame_started

            requested_partner_state: bool | None = None
            partner_added_this_frame = False
            if robot_add_requested.is_set():
                robot_add_requested.clear()
                requested_partner_state = True
            if robot_remove_requested.is_set():
                robot_remove_requested.clear()
                requested_partner_state = False
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
                cspace_points = sample_cspace_projection(
                    model, data, args.cspace_samples
                )
                draw_cspace_projection(
                    viewer,
                    cspace_points,
                    cspace_visible,
                    sphere_visible,
                    sphere_center,
                )
                state = "ADDED" if partner_robot_enabled else "REMOVED"
                print(
                    f"Partner robot: {state}; {len(cspace_points)} free samples; "
                    "replanning..."
                )

            if partner_motion_toggle_requested.is_set():
                partner_motion_toggle_requested.clear()
                if partner_robot_enabled:
                    partner_motion_enabled = not partner_motion_enabled
                    if not partner_motion_enabled and executing_arm in (
                        "partner",
                        "dual",
                    ):
                        was_dual_execution = executing_arm == "dual"
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
                    partner_robot_enabled and partner_motion_enabled
                )
                if dual_cycle_active:
                    preferred_arm = "primary"
                mark_for_replan("primary", now)
                mark_for_replan("partner", now)
                hold_arm("primary")
                hold_arm("partner")
                state = "ON" if obstacles_enabled else "OFF"
                cspace_points = sample_cspace_projection(
                    model, data, args.cspace_samples
                )
                draw_cspace_projection(
                    viewer,
                    cspace_points,
                    cspace_visible,
                    sphere_visible,
                    sphere_center,
                )
                print(
                    f"Gate obstacle: {state}; {len(cspace_points)} free samples; "
                    "replanning..."
                )

            if cspace_toggle_requested.is_set():
                cspace_toggle_requested.clear()
                cspace_visible = not cspace_visible
                draw_cspace_projection(
                    viewer,
                    cspace_points,
                    cspace_visible,
                    sphere_visible,
                    sphere_center,
                )
                print(
                    "C-space projection: "
                    f"{'ON' if cspace_visible else 'OFF'}"
                )

            if sphere_toggle_requested.is_set():
                sphere_toggle_requested.clear()
                sphere_visible = not sphere_visible
                draw_cspace_projection(
                    viewer,
                    cspace_points,
                    cspace_visible,
                    sphere_visible,
                    sphere_center,
                )
                print(
                    "C-space spherical envelope: "
                    f"{'ON' if sphere_visible else 'OFF'}"
                )

            shared_target = data.mocap_pos[shared_target_mocap_id].copy()
            shared_target_changed = (
                np.linalg.norm(shared_target - observed_shared_target)
                > args.move_threshold
            )
            if shared_target_changed:
                observed_shared_target = shared_target.copy()
                pending_dual_plans.clear()
                dual_cycle_active = (
                    partner_robot_enabled and partner_motion_enabled
                )
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

            while not worker.results.empty():
                response = worker.results.get_nowait()
                response_arm = response.get("arm", planning_arm or "primary")
                planning_arm = None
                if response.get("fatal"):
                    raise ConnectionError(response.get("error"))
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
                    safe_delay = find_safe_partner_delay_steps(
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
                            "timing was found"
                        )
                        continue
                    dual_partner_delay_steps = safe_delay
                    dual_active_positions = {
                        "primary": primary_positions,
                        "partner": partner_positions,
                    }
                    active_dt = float(pending_dual_plans["primary"]["dt"])
                    trajectory_started = now
                    executing_arm = "dual"
                    active_positions = None
                    dual_cycle_active = False
                    pending_dual_plans.clear()
                    print(
                        "Dual-arm trajectory ready: "
                        f"partner delay={dual_partner_delay_steps * active_dt:.2f}s"
                    )
                    continue

                active_positions = response_positions
                active_dt = float(response["dt"])
                trajectory_started = now
                executing_arm = response_arm
                print(
                    f"{response_arm} trajectory ready: "
                    f"{len(active_positions)} waypoints, "
                    f"attempts={response.get('attempts', 1)}, "
                    f"solver={response.get('solver_time', 0):.3f}s, "
                    f"wall={response.get('wall_time', 0):.3f}s"
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
                    # Let the position actuators settle before freezing this arm
                    # as an obstacle for the other arm's cuRobo request.
                    planner_available_at = now + 1.0

            if (
                planning_arm is None
                and executing_arm is None
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
                    worker.submit(
                        {
                            "type": "plan",
                            "id": request_id,
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
                    preferred_arm = None
                    hold_arm(arm)
                    print(f"Planning {arm} TCP position {target.tolist()}...")
                    break

            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - frame_started)
            if remaining > 0:
                time.sleep(remaining)

    worker.stop()


if __name__ == "__main__":
    main()
