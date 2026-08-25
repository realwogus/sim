#!/usr/bin/env python3
"""Interactive GUI for the independent 7-DOF rail-mounted PiPER."""

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
DEFAULT_SCENE = (
    PROJECT_DIR.parent / "mujoco" / "models" / "scenes" / "rail_piper.xml"
)
JOINT_NAMES = ("rail_slide",) + tuple(f"rail_joint{i}" for i in range(1, 7))
TCP_OFFSET_LINK6 = np.array([0.0, 0.0, 0.13503])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent rail-PiPER endpoint GUI")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5564)
    parser.add_argument("--settle-time", type=float, default=0.35)
    parser.add_argument("--move-threshold", type=float, default=0.001)
    return parser.parse_args()


class PlannerWorker(threading.Thread):
    def __init__(self, host: str, port: int, results: queue.Queue[dict]):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.results = results
        self.requests: queue.Queue[dict | None] = queue.Queue(maxsize=1)

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
                        stream.write((json.dumps(request) + "\n").encode())
                        stream.flush()
                        line = stream.readline()
                        if not line:
                            raise ConnectionError("rail planner closed the connection")
                        self.results.put(json.loads(line))
        except Exception as exc:
            self.results.put({"ok": False, "fatal": True, "error": str(exc)})


def tcp_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    link6 = model.body("rail_link6").id
    return data.xpos[link6] + data.xmat[link6].reshape(3, 3) @ TCP_OFFSET_LINK6


def trajectory_path(
    model: mujoco.MjModel, data: mujoco.MjData, positions: np.ndarray
) -> np.ndarray:
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    qpos_addresses = np.array(
        [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    )
    result = []
    for position in positions:
        scratch.qpos[qpos_addresses] = position
        mujoco.mj_forward(model, scratch)
        result.append(tcp_position(model, scratch))
    return np.asarray(result)


def first_collision(
    model: mujoco.MjModel, data: mujoco.MjData, positions: np.ndarray
) -> str | None:
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    qpos_addresses = np.array(
        [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    )
    for waypoint, position in enumerate(positions):
        scratch.qpos[qpos_addresses] = position
        mujoco.mj_forward(model, scratch)
        if scratch.ncon:
            contact = scratch.contact[0]
            return (
                f"{model.geom(contact.geom1).name} <-> "
                f"{model.geom(contact.geom2).name} at waypoint {waypoint}"
            )
    return None


def draw_preview(
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    path: np.ndarray | None,
    goal_positions: np.ndarray | None,
) -> None:
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        # Add the goal robot first so a long TCP polyline cannot exhaust the
        # user-scene geometry buffer and leave only part of the ghost visible.
        if goal_positions is not None:
            scratch = mujoco.MjData(model)
            scratch.qpos[:] = data.qpos
            for name, value in zip(JOINT_NAMES, goal_positions):
                scratch.joint(name).qpos[0] = value
            mujoco.mj_forward(model, scratch)

            # Let MuJoCo build the complete render geoms first. Copying model
            # mesh IDs into mjvGeom by hand misses material/render metadata and
            # can make multi-mesh links appear fragmented.
            source_scene = mujoco.MjvScene(model, maxgeom=max(256, model.ngeom * 2))
            source_option = mujoco.MjvOption()
            source_perturb = mujoco.MjvPerturb()
            mujoco.mjv_updateScene(
                model,
                scratch,
                source_option,
                source_perturb,
                viewer.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                source_scene,
            )
            wanted_geom_ids = {
                geom_id
                for geom_id in range(model.ngeom)
                if model.geom(geom_id).name == "rail_carriage_geom"
                or (
                    model.body(int(model.geom_bodyid[geom_id])).name.startswith(
                        "rail_"
                    )
                    and model.geom_group[geom_id] == 2
                )
            }
            for source_id in range(source_scene.ngeom):
                source = source_scene.geoms[source_id]
                if (
                    source.objtype != int(mujoco.mjtObj.mjOBJ_GEOM)
                    or source.objid not in wanted_geom_ids
                    or scene.ngeom >= scene.maxgeom
                ):
                    continue
                ghost = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    ghost,
                    source.type,
                    np.asarray(source.size, dtype=np.float64),
                    np.asarray(source.pos, dtype=np.float64),
                    np.asarray(source.mat, dtype=np.float64).reshape(-1),
                    np.array([0.10, 0.90, 1.0, 0.42], dtype=np.float32),
                )
                for scalar_field in (
                    "camdist",
                    "dataid",
                    "modelrbound",
                    "objid",
                    "objtype",
                    "reflectance",
                    "segid",
                    "shininess",
                    "specular",
                    "texcoord",
                    "texid",
                    "texuniform",
                ):
                    setattr(ghost, scalar_field, getattr(source, scalar_field))
                ghost.texrepeat[:] = source.texrepeat
                ghost.matid = -1
                ghost.category = int(mujoco.mjtCatBit.mjCAT_DYNAMIC)
                ghost.transparent = 1
                ghost.emission = 0.25
                scene.ngeom += 1

        if path is not None:
            for start, end in zip(path[:-1], path[1:]):
                if scene.ngeom >= scene.maxgeom:
                    break
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).reshape(-1),
                    np.array([1.0, 0.08, 0.03, 1.0], dtype=np.float32),
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    4.0,
                    np.asarray(start),
                    np.asarray(end),
                )
                scene.ngeom += 1


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    target_body = model.body("rail_endpoint_target")
    target_mocap_id = int(target_body.mocapid[0])
    target_geom_id = model.geom("rail_endpoint_target_marker").id
    data.mocap_pos[target_mocap_id] = tcp_position(model, data)

    rail_actuator = model.actuator("rail_slide").id
    arm_actuators = np.array(
        [model.actuator(name).id for name in JOINT_NAMES[1:]]
    )
    rail_joint = data.joint("rail_slide")
    rail_command = float(rail_joint.qpos[0])
    results: queue.Queue[dict] = queue.Queue()
    worker = PlannerWorker(args.host, args.port, results)
    worker.start()
    execute_requested = threading.Event()
    target_undo_requested = threading.Event()
    target_redo_requested = threading.Event()
    stop = False

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
        elif keycode == glfw.KEY_SPACE:
            execute_requested.set()

    def request_stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    observed_target = data.mocap_pos[target_mocap_id].copy()
    last_change = time.monotonic()
    last_planned = observed_target.copy()
    planning = False
    ik_status = "NOT RUN"
    plan_status = "NOT RUN"
    failure_stage = "-"
    preview: np.ndarray | None = None
    preview_path: np.ndarray | None = None
    preview_goal: np.ndarray | None = None
    executing: np.ndarray | None = None
    active_dt = 0.02
    trajectory_started = 0.0
    request_id = 0
    discard_response_through = 0

    def joint_values() -> np.ndarray:
        return np.array([float(data.joint(name).qpos[0]) for name in JOINT_NAMES])

    def command_joints(values: np.ndarray) -> None:
        """Command the arm dynamically but pin the rail as a kinematic axis."""
        nonlocal rail_command
        rail_range = model.joint("rail_slide").range
        rail_command = float(
            np.clip(values[0], rail_range[0], rail_range[1])
        )
        data.ctrl[rail_actuator] = rail_command
        data.ctrl[arm_actuators] = values[1:]
        rail_joint.qpos[0] = rail_command
        rail_joint.qvel[0] = 0.0

    def hold() -> None:
        command_joints(joint_values())

    print("Orange marker = independent rail-PiPER TCP target")
    print("Move the marker, wait for the red preview, then press Space")
    print("Ctrl+Z undoes a target edit; Ctrl+Shift+Z redoes it")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.33, 1.55, 0.68])
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -25.0

        def refresh_overlay() -> None:
            trajectory_status = (
                "EXECUTING"
                if executing is not None
                else "READY - press SPACE"
                if preview is not None
                else "PLANNING"
                if planning
                else "IK GHOST ONLY"
                if preview_goal is not None
                else "IDLE"
            )
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Planner\nRail + arm DOF\nIK\nMotion plan\nFailure stage\nTrajectory",
                    f"INDEPENDENT\n1 + 6 = 7\n{ik_status}\n{plan_status}\n{failure_stage}\n{trajectory_status}",
                )
            )

        refresh_overlay()
        target_history: list[np.ndarray] = []
        target_redo_history: list[np.ndarray] = []
        target_edit_origin: np.ndarray | None = None
        while viewer.is_running() and not stop:
            frame_started = time.monotonic()
            now = frame_started

            # MuJoCo's Controls panel writes data.ctrl. Treat a user change as
            # a direct kinematic rail command instead of overwriting it with
            # the previous pinned value on the next frame.
            manual_rail_command = float(
                np.clip(
                    data.ctrl[rail_actuator],
                    model.joint("rail_slide").range[0],
                    model.joint("rail_slide").range[1],
                )
            )
            if (
                executing is None
                and abs(manual_rail_command - rail_command) > 1e-7
            ):
                rail_command = manual_rail_command
                rail_joint.qpos[0] = rail_command
                rail_joint.qvel[0] = 0.0
                mujoco.mj_forward(model, data)
                preview = None
                preview_path = None
                preview_goal = None
                planning = False
                discard_response_through = request_id
                last_planned = data.mocap_pos[target_mocap_id].copy()
                ik_status = "MANUAL RAIL"
                plan_status = "NOT RUN"
                failure_stage = "-"
                draw_preview(viewer, model, data, None, None)
                refresh_overlay()

            if target_undo_requested.is_set():
                target_undo_requested.clear()
                current_target = data.mocap_pos[target_mocap_id].copy()
                previous_target = None
                if target_edit_origin is not None:
                    previous_target = target_edit_origin.copy()
                    target_edit_origin = None
                elif target_history:
                    previous_target = target_history.pop()
                if previous_target is None:
                    print("Rail target undo: no previous position")
                else:
                    target_redo_history.append(current_target)
                    target_redo_history[:] = target_redo_history[-100:]
                    data.mocap_pos[target_mocap_id] = previous_target
                    observed_target = previous_target.copy()
                    last_change = now
                    last_planned = None
                    preview = None
                    preview_path = None
                    preview_goal = None
                    if executing is not None:
                        hold()
                        executing = None
                    draw_preview(viewer, model, data, None, None)
                    refresh_overlay()
                    print(f"Rail target undo: restored {previous_target.tolist()}")

            if target_redo_requested.is_set():
                target_redo_requested.clear()
                if target_edit_origin is not None:
                    print("Rail target redo: finish the current target edit first")
                elif not target_redo_history:
                    print("Rail target redo: no undone position")
                else:
                    current_target = data.mocap_pos[target_mocap_id].copy()
                    next_target = target_redo_history.pop()
                    target_history.append(current_target)
                    target_history[:] = target_history[-100:]
                    data.mocap_pos[target_mocap_id] = next_target
                    observed_target = next_target.copy()
                    last_change = now
                    last_planned = None
                    preview = None
                    preview_path = None
                    preview_goal = None
                    if executing is not None:
                        hold()
                        executing = None
                    draw_preview(viewer, model, data, None, None)
                    refresh_overlay()
                    print(f"Rail target redo: restored {next_target.tolist()}")

            target = data.mocap_pos[target_mocap_id].copy()
            if np.linalg.norm(target - observed_target) > args.move_threshold:
                if target_edit_origin is None:
                    target_edit_origin = observed_target.copy()
                    target_redo_history.clear()
                observed_target = target
                last_change = now
                preview = None
                preview_path = None
                preview_goal = None
                if executing is not None:
                    hold()
                    executing = None
                ik_status = "WAITING"
                plan_status = "WAITING"
                failure_stage = "-"
                draw_preview(viewer, model, data, None, None)
                refresh_overlay()
            elif target_edit_origin is not None and now - last_change >= args.settle_time:
                if np.linalg.norm(target - target_edit_origin) > args.move_threshold:
                    target_history.append(target_edit_origin.copy())
                    target_history[:] = target_history[-100:]
                target_edit_origin = None

            while not results.empty():
                response = results.get_nowait()
                planning = False
                if response.get("fatal"):
                    raise ConnectionError(response.get("error"))
                if int(response.get("id", -1)) <= discard_response_through:
                    print("Discarded stale plan after manual rail movement")
                    continue
                response_target = np.asarray(response.get("target_position_world", []))
                if response_target.shape != (3,) or np.linalg.norm(
                    response_target - data.mocap_pos[target_mocap_id]
                ) > 0.008:
                    continue
                last_planned = response_target.copy()
                ik_status = "SUCCESS" if response.get("ik_ok") else "FAILED"
                if not response.get("ok"):
                    plan_status = "FAILED"
                    failure_stage = response.get("failure_stage", "UNKNOWN")
                    ik_pose = np.asarray(response.get("ik_positions", []), dtype=np.float64)
                    preview_goal = ik_pose if ik_pose.shape == (len(JOINT_NAMES),) else None
                    draw_preview(viewer, model, data, None, preview_goal)
                    print(f"Rail planning failed: {response.get('error')}")
                    for diagnostic in response.get("diagnostics", []):
                        print(
                            "  "
                            f"attempt={diagnostic.get('attempt')} "
                            f"stage={diagnostic.get('stage')} "
                            f"IK={diagnostic.get('ik_success', 0)}/"
                            f"{diagnostic.get('ik_seeds', 0)} "
                            f"graph={diagnostic.get('graph_success')} "
                            f"feasible={diagnostic.get('feasible_seeds', '-')} "
                            f"interpolated={diagnostic.get('interpolated_feasible_seeds', '-')} "
                            f"position_error={diagnostic.get('min_position_error', '-')}",
                        )
                    refresh_overlay()
                    continue
                candidate = np.asarray(response["positions"], dtype=np.float64)
                collision = first_collision(model, data, candidate)
                if collision is not None:
                    plan_status = "REJECTED"
                    failure_stage = "MUJOCO_COLLISION"
                    preview_goal = candidate[-1].copy()
                    draw_preview(viewer, model, data, None, preview_goal)
                    print(f"Rail trajectory rejected by MuJoCo: {collision}")
                    refresh_overlay()
                    continue
                preview = candidate
                preview_path = trajectory_path(model, data, candidate)
                preview_goal = candidate[-1].copy()
                active_dt = float(response["dt"])
                plan_status = "SUCCESS"
                failure_stage = "-"
                draw_preview(viewer, model, data, preview_path, preview_goal)
                hold()
                refresh_overlay()
                print(
                    f"Rail trajectory ready: {len(candidate)} waypoints, "
                    f"solver={response.get('solver_time', 0):.3f}s"
                )

            if execute_requested.is_set():
                execute_requested.clear()
                if preview is None:
                    print("No rail trajectory preview is ready")
                else:
                    executing = preview
                    preview = None
                    trajectory_started = now
                    refresh_overlay()

            if executing is not None:
                index = min((now - trajectory_started) / active_dt, len(executing) - 1)
                lo = int(np.floor(index))
                hi = min(lo + 1, len(executing) - 1)
                alpha = index - lo
                command_joints(
                    (1.0 - alpha) * executing[lo] + alpha * executing[hi]
                )
                if index >= len(executing) - 1:
                    executing = None
                    preview_path = None
                    preview_goal = None
                    draw_preview(viewer, model, data, None, None)
                    refresh_overlay()

            if (
                not planning
                and preview is None
                and executing is None
                and now - last_change >= args.settle_time
                and (
                    last_planned is None
                    or np.linalg.norm(target - last_planned) > args.move_threshold
                )
            ):
                request_id += 1
                worker.submit(
                    {
                        "type": "plan",
                        "id": request_id,
                        "start": joint_values().tolist(),
                        "target_position_world": target.tolist(),
                    }
                )
                planning = True
                ik_status = "SOLVING"
                plan_status = "SOLVING"
                hold()
                refresh_overlay()

            # The slide is an ideal commanded axis: it has no coast, overshoot,
            # or gravity-driven motion. Re-pin it after the dynamic arm step.
            rail_joint.qpos[0] = rail_command
            rail_joint.qvel[0] = 0.0
            data.ctrl[rail_actuator] = rail_command
            mujoco.mj_step(model, data)
            rail_joint.qpos[0] = rail_command
            rail_joint.qvel[0] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - frame_started)
            if remaining > 0:
                time.sleep(remaining)

    worker.stop()


if __name__ == "__main__":
    main()
