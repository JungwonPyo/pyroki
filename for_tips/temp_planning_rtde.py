#!/usr/bin/env python3
import os
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from yourdfpy import URDF

import pyroki as pk
import pyroki_snippets as pks
from pyroki.collision import HalfSpace, RobotCollision, Sphere, Box
from scene_understanding_msgs.msg import SceneContext

from gripper_handler_rtde import GripperHandler

try:
    import rtde_control
    import rtde_receive
    RTDE_AVAILABLE = True
except Exception:
    RTDE_AVAILABLE = False

try:
    import viser
    from viser.extras import ViserUrdf
    VISER_AVAILABLE = True
except Exception:
    VISER_AVAILABLE = False


@dataclass
class DetectedObjectState:
    object_id: str
    class_name: str
    center: np.ndarray
    size: np.ndarray
    frame_id: str
    score: float
    stamp_sec: float


@dataclass
class PoseWaypoint:
    pos: np.ndarray
    wxyz: np.ndarray


@dataclass
class TaskStep:
    name: str
    kind: str
    waypoints: List[PoseWaypoint] = field(default_factory=list)
    gripper_action: Optional[str] = None
    dwell_sec: float = 0.0


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=float)
    n = np.linalg.norm([x, y, z, w])
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=float)


def make_transform(translation, quaternion_xyzw) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = quat_xyzw_to_rot(np.asarray(quaternion_xyzw, dtype=float))
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p_h = np.ones(4, dtype=float)
    p_h[:3] = p
    return (T @ p_h)[:3]


class PyrokiRtdeTaskPlanner(Node):
    def __init__(self):
        super().__init__("pyroki_rtde_task_planner")
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # self.declare_parameter("scene_topic", "/scene_context")
        self.declare_parameter("scene_topic", "/scene_graph/planner_context")
        self.declare_parameter("robot_ip", "192.168.200.11")
        self.declare_parameter("enable_rtde", True)
        self.declare_parameter("use_visualizer", True)
        self.declare_parameter("task_config_path", os.path.join(base_dir, "task_config.yaml"))
        self.declare_parameter("transform_config_path", os.path.join(base_dir, "camera_to_robot.yaml"))

        self.robot_ip = self.get_parameter("robot_ip").value
        self.enable_rtde = bool(self.get_parameter("enable_rtde").value) and RTDE_AVAILABLE
        self.use_visualizer = bool(self.get_parameter("use_visualizer").value) and VISER_AVAILABLE

        self.task_cfg = load_yaml(self.get_parameter("task_config_path").value)
        self.tf_cfg = load_yaml(self.get_parameter("transform_config_path").value)

        perception = self.task_cfg.get("perception", {})

        self.scene_topic = perception.get(
            "scene_topic",
            self.get_parameter("scene_topic").value,
        )
        self.min_score = float(perception.get("min_score", 0.5))
        self.collision_class_names = set(perception.get("collision_class_names", []))
        self.default_target_class_name = perception.get("target_class_name", "")

        motion = self.task_cfg["motion"]
        servo = self.task_cfg["servo"]
        grip = self.task_cfg["gripper"]

        self.planner_frame = self.task_cfg.get("planner_frame", "base_link")
        self.target_link_name = self.task_cfg.get("target_link_name", "tool0")
        self.default_tool_wxyz = normalize_quat_wxyz(
            np.array(self.task_cfg.get("default_tool_wxyz", [0.0, 0.0, 1.0, 0.0]), dtype=float)
        )

        self.dt = float(motion["control_dt"])
        self.traj_len = int(motion["traj_len"])
        self.waypoint_reached_pos_tol = float(motion["waypoint_reached_pos_tol"])
        self.bbox_inflation_m = float(motion["bbox_inflation_m"])
        self.obstacle_timeout_sec = float(motion["obstacle_timeout_sec"])
        self.use_box_collision = bool(motion["use_box_collision"])
        self.keep_obstacle_slot = bool(motion["keep_obstacle_slot"])
        self.enable_stall_advance = bool(motion["enable_stall_advance"])
        self.stall_cycles_to_advance = int(motion["stall_cycles_to_advance"])

        self.servo_max_joint_step = float(servo["max_joint_step"])
        self.servo_speed = float(servo["speed"])
        self.servo_acceleration = float(servo["acceleration"])
        self.servo_lookahead_time = float(servo["lookahead_time"])
        self.servo_gain = float(servo["gain"])

        self.gripper_enabled = bool(grip["enabled"])
        self.gripper_cfg = grip

        tf_block = self.tf_cfg["transform"]
        self.camera_parent_frame = tf_block["parent_frame"]
        self.camera_frame = tf_block["child_frame"]
        self.T_parent_camera = make_transform(tf_block["translation"], tf_block["quaternion_xyzw"])

        urdf_path = os.path.join(base_dir, "ur5e_with_robotiq.urdf")
        self.urdf_model = URDF.load(urdf_path)
        self.robot = pk.Robot.from_urdf(self.urdf_model)
        self.robot_coll = RobotCollision.from_urdf(self.urdf_model)

        self.plane_coll = HalfSpace.from_point_and_normal(
            np.array([0.0, 0.0, -0.05], dtype=float),   # lower plane by 2 cm
            np.array([0.0, 0.0, 1.0], dtype=float),
        )

        q0 = np.array(self.robot.joint_var_cls.default_factory(), dtype=float)
        self.current_q = q0.copy()
        self.actual_q = q0.copy()
        self.actual_tcp_pose = np.zeros(6, dtype=float)
        self.sol_traj = np.repeat(q0[None], self.traj_len, axis=0)

        self._lock = threading.Lock()
        self.perceived_objects: List[DetectedObjectState] = []
        self.selected_target: Optional[DetectedObjectState] = None

        self.task_steps: List[TaskStep] = []
        self.task_ready = False
        self.task_finished = False
        self.current_step_idx = 0
        self.current_wp_idx = 0
        self.step_started_at = None

        self.prev_wp_dist = float("inf")
        self.wp_stall_count = 0
        self.box_supported = True

        self.rtde_c = None
        self.rtde_r = None
        self.rtde_ready = False
        self.gripper = None
        self.last_elapsed_ms = 0.0

        self._connect_rtde()
        self._connect_gripper()

        self.subscription = self.create_subscription(
            SceneContext,
            self.scene_topic,
            self.scene_callback,
            10,
        )

        if self.use_visualizer:
            self.setup_visualizer(self.urdf_model)

        self.feedback_timer = self.create_timer(0.02, self.feedback_loop)
        self.control_timer = self.create_timer(self.dt, self.control_loop)

        self.build_static_task()
        self.get_logger().info(f"Subscribed to {self.scene_topic}")

    def _connect_rtde(self):
        if not self.enable_rtde:
            self.get_logger().warning("RTDE disabled.")
            return
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            self.actual_q = np.array(self.rtde_r.getActualQ(), dtype=float)
            self.current_q = self.actual_q.copy()
            self.actual_tcp_pose = np.array(self.rtde_r.getActualTCPPose(), dtype=float)
            self.rtde_ready = True
            self.get_logger().info(f"Connected RTDE to {self.robot_ip}")
        except Exception as exc:
            self.get_logger().error(f"Failed RTDE connection: {exc}")
            self.rtde_ready = False

    def _connect_gripper(self):
        if not self.gripper_enabled:
            self.get_logger().warning("Gripper disabled.")
            return

        port = self.gripper_cfg.get("port", "/tmp/ttyUR")
        if not os.path.exists(port):
            self.get_logger().warning(f"Gripper device does not exist: {port}")
            return

        try:
            self.gripper = GripperHandler(
                port=port,
                baudrate=int(self.gripper_cfg["baudrate"]),
                timeout=float(self.gripper_cfg["timeout"]),
                slaveID=int(self.gripper_cfg["slave_id"]),
            )
            self.get_logger().info(f"Opened gripper port: {port}")

            status = self.gripper.read_status()
            self.get_logger().info(f"Gripper status response: {status.hex()}")

            act = self.gripper.activate()
            self.get_logger().info(f"Gripper activate response: {act.hex()}")

            time.sleep(0.5)

            opn = self.gripper.open(
                speed=int(self.gripper_cfg["speed"]),
                force=int(self.gripper_cfg["force"]),
                position=int(self.gripper_cfg["open_position"]),
            )
            self.get_logger().info(f"Gripper open response: {opn.hex()}")

            self.get_logger().info("Gripper connected and initialized.")

        except Exception as exc:
            self.gripper = None
            self.get_logger().error(f"Failed gripper connection: {exc}")

    def setup_visualizer(self, urdf):
        self.server = viser.ViserServer()
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self.urdf_vis = ViserUrdf(self.server, urdf, root_node_name="/robot")
        self.target_handle = self.server.scene.add_transform_controls(
            "/ik_target", scale=0.15, position=(0.0, 0.0, 0.2), wxyz=(1.0, 0.0, 0.0, 0.0)
        )
        self.obstacle_handle = self.server.scene.add_transform_controls(
            "/obstacle", scale=0.15, position=(10.0, 10.0, 10.0), wxyz=(1.0, 0.0, 0.0, 0.0)
        )
        self.target_frame_handle = self.server.scene.add_batched_axes(
            "/planned_frames",
            axes_length=0.05,
            axes_radius=0.005,
            batched_positions=np.zeros((max(self.traj_len, 1), 3)),
            batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * max(self.traj_len, 1)),
        )
        self.status_handle = self.server.gui.add_text("Status", "idle")
        self.timing_handle = self.server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        self.step_handle = self.server.gui.add_number("Task step", 0.0, disabled=True)
        self.wp_handle = self.server.gui.add_number("Waypoint", 0.0, disabled=True)

    def transform_center_to_planner(self, center: np.ndarray, frame_id: str) -> Optional[np.ndarray]:
        if frame_id in ("", None, self.planner_frame, self.camera_parent_frame):
            return center
        if frame_id == self.camera_frame:
            return transform_point(self.T_parent_camera, center)
        self.get_logger().warning(f"Unknown bbox frame '{frame_id}', dropping object.")
        return None

    def scene_callback(self, msg: SceneContext):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        objs = []

        planner_frame_msg = msg.planner_frame if hasattr(msg, "planner_frame") and msg.planner_frame else self.planner_frame

        for obj in msg.objects:
            if obj.score < self.min_score:
                continue
            if not obj.bbox_3d.valid:
                continue

            raw_frame = obj.bbox_3d.frame_id if obj.bbox_3d.frame_id else planner_frame_msg
            center_raw = np.array(
                [obj.bbox_3d.center.x, obj.bbox_3d.center.y, obj.bbox_3d.center.z],
                dtype=float,
            )
            center = self.transform_center_to_planner(center_raw, raw_frame)
            if center is None:
                continue

            size = np.maximum(
                np.array(
                    [obj.bbox_3d.size.x, obj.bbox_3d.size.y, obj.bbox_3d.size.z],
                    dtype=float,
                ),
                1e-4,
            )

            objs.append(
                DetectedObjectState(
                    object_id=obj.id,
                    class_name=obj.class_name,
                    center=center,
                    size=size,
                    frame_id=self.planner_frame,
                    score=float(obj.score),
                    stamp_sec=now_sec,
                )
            )

        with self._lock:
            # self.get_logger().info(
            #     "scene objects: " +
            #     ", ".join([f"{o.class_name}@{np.round(o.center, 3)} score={o.score:.2f}" for o in objs])
            # )
            self.perceived_objects = objs

    def get_object_by_class(self, class_name: str) -> Optional[DetectedObjectState]:
        with self._lock:
            matches = [obj for obj in self.perceived_objects if obj.class_name == class_name]
        if not matches:
            return None
        matches.sort(key=lambda o: o.score, reverse=True)
        return matches[0]

    def resolve_anchor(self, anchor_name: str) -> PoseWaypoint:
        anchor_cfg = self.task_cfg["anchors"][anchor_name]
        return PoseWaypoint(
            pos=np.array(anchor_cfg["pos"], dtype=float),
            wxyz=normalize_quat_wxyz(np.array(anchor_cfg.get("wxyz", self.default_tool_wxyz), dtype=float)),
        )

    def resolve_path(self, path_name: str) -> List[PoseWaypoint]:
        out = []
        for item in self.task_cfg["paths"][path_name]:
            out.append(self.resolve_anchor(item["anchor"]))
        return out

    def resolve_target_base(self, step_cfg: Dict[str, Any]) -> PoseWaypoint:
        source = step_cfg.get("target_source", "config")

        if source == "config":
            anchor_name = step_cfg.get("target_anchor", "debug_pick_target")
            return self.resolve_anchor(anchor_name)

        if source == "perception":
            class_name = step_cfg.get("target_class_name", self.default_target_class_name)
            obj = self.get_object_by_class(class_name)
            if obj is None:
                raise RuntimeError(f"No perceived object found for class '{class_name}'")
            return PoseWaypoint(pos=obj.center.copy(), wxyz=self.default_tool_wxyz.copy())

        raise ValueError(f"Unknown target_source '{source}'")

    # def resolve_relative_waypoints(self, step_cfg: Dict[str, Any]) -> List[PoseWaypoint]:
    #     base = self.resolve_target_base(step_cfg)
    #     out = []
    #     for wp_cfg in step_cfg.get("waypoints", []):
    #         offset = np.array(wp_cfg.get("offset", [0.0, 0.0, 0.0]), dtype=float)
    #         out.append(PoseWaypoint(pos=base.pos + offset, wxyz=base.wxyz.copy()))
    #     return out

    def resolve_relative_waypoints(self, step_cfg: Dict[str, Any]) -> List[PoseWaypoint]:
        out = []

        base = None
        if "target_source" in step_cfg:
            base = self.resolve_target_base(step_cfg)

        for wp_cfg in step_cfg.get("waypoints", []):
            if "anchor" in wp_cfg:
                out.append(self.resolve_anchor(wp_cfg["anchor"]))
                continue

            if wp_cfg.get("relative_to", None) == "target":
                if base is None:
                    raise RuntimeError(
                        f"Step '{step_cfg.get('name', 'unknown')}' uses relative target waypoint "
                        "but has no target_source."
                    )
                offset = np.array(wp_cfg.get("offset", [0.0, 0.0, 0.0]), dtype=float)
                out.append(PoseWaypoint(pos=base.pos + offset, wxyz=base.wxyz.copy()))
                continue

            raise RuntimeError(
                f"Unsupported waypoint spec in step '{step_cfg.get('name', 'unknown')}': {wp_cfg}"
            )

        return out

    def build_static_task(self):
        self.task_steps = []
        for step_cfg in self.task_cfg["scenario"]:
            kind = step_cfg["kind"]
            name = step_cfg["name"]

            if kind == "move":
                self.task_steps.append(TaskStep(name=name, kind=kind))
            elif kind == "gripper":
                self.task_steps.append(TaskStep(
                    name=name,
                    kind=kind,
                    gripper_action=step_cfg["action"],
                    dwell_sec=float(step_cfg.get("dwell_sec", 1.0)),
                ))
            elif kind == "wait":
                self.task_steps.append(TaskStep(
                    name=name,
                    kind=kind,
                    dwell_sec=float(step_cfg.get("dwell_sec", 1.0)),
                ))

        self.task_ready = True
        self.task_finished = False
        self.current_step_idx = 0
        self.current_wp_idx = 0
        self.step_started_at = None
        self.prev_wp_dist = float("inf")
        self.wp_stall_count = 0

    def current_step_cfg(self) -> Dict[str, Any]:
        return self.task_cfg["scenario"][self.current_step_idx]

    def current_step(self) -> Optional[TaskStep]:
        if self.task_finished or self.current_step_idx >= len(self.task_steps):
            return None
        return self.task_steps[self.current_step_idx]

    def current_waypoints_for_step(self) -> List[PoseWaypoint]:
        step_cfg = self.current_step_cfg()
        if "use_path" in step_cfg:
            return self.resolve_path(step_cfg["use_path"])
        return self.resolve_relative_waypoints(step_cfg)

    def current_waypoint(self) -> Optional[PoseWaypoint]:
        step = self.current_step()
        if step is None or step.kind != "move":
            return None
        wps = self.current_waypoints_for_step()
        if self.current_wp_idx >= len(wps):
            return None
        return wps[self.current_wp_idx]

    def advance_step(self):
        self.current_step_idx += 1
        self.current_wp_idx = 0
        self.step_started_at = None
        self.prev_wp_dist = float("inf")
        self.wp_stall_count = 0
        if self.current_step_idx >= len(self.task_steps):
            self.task_finished = True
            self.get_logger().info("Task finished.")
        else:
            self.get_logger().info(f"Advance to step {self.current_step_idx}: {self.task_steps[self.current_step_idx].name}")

    def advance_waypoint_or_step(self):
        wps = self.current_waypoints_for_step()
        self.current_wp_idx += 1
        self.prev_wp_dist = float("inf")
        self.wp_stall_count = 0
        if self.current_wp_idx >= len(wps):
            self.advance_step()
        else:
            self.get_logger().info(
                f"Advance to waypoint {self.current_wp_idx} of step '{self.current_step().name}'"
            )

    def run_gripper_action(self, action: Optional[str]):
        self.get_logger().info(
            f"run_gripper_action(action={action}, enabled={self.gripper_enabled}, "
            f"gripper_is_none={self.gripper is None})"
        )

        if action is None:
            return

        if not self.gripper_enabled:
            self.get_logger().warning(
                f"Skipping gripper action '{action}' because gripper is disabled in config."
            )
            return

        if self.gripper is None:
            self.get_logger().warning(
                f"Gripper object is None before '{action}', trying reconnect."
            )
            self._connect_gripper()

        if self.gripper is None:
            self.get_logger().warning(
                f"Skipping gripper action '{action}' because gripper is not connected."
            )
            return

        speed = int(self.gripper_cfg["speed"])
        force = int(self.gripper_cfg["force"])

        try:
            if action == "open":
                self.gripper.open(
                    speed=speed,
                    force=force,
                    position=int(self.gripper_cfg["open_position"]),
                )
            elif action == "close":
                self.gripper.close(
                    speed=speed,
                    force=force,
                    position=int(self.gripper_cfg["close_position"]),
                )
            else:
                self.get_logger().warning(f"Unknown gripper action '{action}'")
                return

            self.get_logger().info(f"Gripper action '{action}' completed.")

        except Exception as exc:
            self.get_logger().error(f"Gripper action '{action}' failed: {exc}")
            try:
                if self.gripper is not None:
                    self.gripper.close_serial()
            except Exception:
                pass
            self.gripper = None
        
    def feedback_loop(self):
        if not self.rtde_ready:
            return
        try:
            self.actual_q = np.array(self.rtde_r.getActualQ(), dtype=float)
            self.actual_tcp_pose = np.array(self.rtde_r.getActualTCPPose(), dtype=float)
            self.current_q = self.actual_q.copy()
            if self.use_visualizer:
                self.urdf_vis.update_cfg(self.current_q)
        except Exception as exc:
            self.get_logger().warning(f"RTDE feedback failed: {exc}")
            self.rtde_ready = False

    def obstacle_is_fresh(self, obs: DetectedObjectState, now_sec: float) -> bool:
        return (now_sec - obs.stamp_sec) <= self.obstacle_timeout_sec

    def bbox_to_sphere(self, obs: DetectedObjectState) -> Sphere:
        radius = 0.5 * float(np.max(obs.size)) + self.bbox_inflation_m
        sphere = Sphere.from_center_and_radius(np.array([0.0, 0.0, 0.0]), np.array([radius]))
        return sphere.transform_from_wxyz_position(
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=obs.center,
        )

    def bbox_to_box(self, obs: DetectedObjectState) -> Box:
        # size_scale = np.array([0.90, 0.90, 0.75], dtype=float)
        size_scale = np.array([1.0, 1.0, 1.0], dtype=float)
        extent = obs.size * size_scale + 2.0 * self.bbox_inflation_m
        extent = np.maximum(extent, np.array([0.03, 0.03, 0.03], dtype=float))

        # self.get_logger().info(
        #     f"collision box for {obs.class_name}: "
        #     f"center={np.round(obs.center, 3)}, raw_size={np.round(obs.size, 3)}, "
        #     f"extent={np.round(extent, 3)}"
        # )

        return Box.from_extent(
            extent=extent,
            position=obs.center,
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        
    def obstacle_geom(self, obs: DetectedObjectState):
        if self.use_box_collision and self.box_supported:
            try:
                return self.bbox_to_box(obs)
            except Exception:
                self.box_supported = False
        return self.bbox_to_sphere(obs)

    def inactive_obstacle_geom(self):
        fake = DetectedObjectState(
            object_id="inactive",
            class_name="inactive",
            center=np.array([10.0, 10.0, 10.0]),
            size=np.array([0.02, 0.02, 0.02]),
            frame_id=self.planner_frame,
            score=0.0,
            stamp_sec=0.0,
        )
        return self.obstacle_geom(fake)

    def build_world_collisions(self) -> Tuple[List[Any], List[DetectedObjectState]]:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            objs = list(self.perceived_objects)

        step_cfg = self.current_step_cfg()
        target_class_name = step_cfg.get("target_class_name", self.default_target_class_name)

        # self.get_logger().info(
        #     f"collision whitelist={list(self.collision_class_names)}"
        # )

        # self.get_logger().info(
        #     "collision candidates: " +
        #     ", ".join([f"{o.class_name}@{np.round(o.center, 3)}" for o in objs])
        # )

        active = []
        for obj in objs:
            if not self.obstacle_is_fresh(obj, now_sec):
                continue

            if obj.class_name not in self.collision_class_names:
                continue

            if obj.class_name == target_class_name:
                continue

            active.append(obj)
            
        # self.get_logger().info(
        #     "active obstacles: " +
        #     ", ".join([f"{o.class_name}@{np.round(o.center, 3)}" for o in active])
        # )

        world = [self.plane_coll]
        if len(active) == 0 and self.keep_obstacle_slot:
            world.append(self.inactive_obstacle_geom())
        else:
            for obs in active:
                world.append(self.obstacle_geom(obs))

        return world, active

    def publish_or_send_joint_command(self, q_cmd: np.ndarray):
        dq = q_cmd - self.current_q
        dq = np.clip(dq, -self.servo_max_joint_step, self.servo_max_joint_step)
        q_safe = self.current_q + dq

        if self.rtde_ready:
            try:
                self.rtde_c.servoJ(
                    q_safe.tolist(),
                    self.servo_speed,
                    self.servo_acceleration,
                    self.dt,
                    self.servo_lookahead_time,
                    self.servo_gain,
                )
                return
            except Exception as exc:
                self.get_logger().warning(f"RTDE servoJ failed: {exc}")
                self.rtde_ready = False

    def advance_move_progress(self, ee_pos: np.ndarray):
        wp = self.current_waypoint()
        if wp is None:
            return

        dist = np.linalg.norm(ee_pos - wp.pos)
        self.get_logger().info(f"Step={self.current_step().name}, wp={self.current_wp_idx}, dist={dist:.4f}")

        if dist < self.waypoint_reached_pos_tol:
            self.advance_waypoint_or_step()
            return

        if dist < self.prev_wp_dist - 1e-3:
            self.wp_stall_count = 0
        else:
            self.wp_stall_count += 1

        self.prev_wp_dist = dist

        if self.enable_stall_advance and self.wp_stall_count >= self.stall_cycles_to_advance:
            self.get_logger().warning(
                f"Waypoint stall in step '{self.current_step().name}', forcing next waypoint."
            )
            self.advance_waypoint_or_step()

    def run_non_motion_step(self):
        step = self.current_step()
        now = time.time()

        if self.step_started_at is None:
            self.step_started_at = now
            self.get_logger().info(f"Executing non-motion step: {step.name}")

            if step.kind == "gripper":
                try:
                    self.run_gripper_action(step.gripper_action)
                except Exception as exc:
                    self.get_logger().error(
                        f"Non-motion step '{step.name}' failed: {exc}"
                    )

        if (now - self.step_started_at) >= step.dwell_sec:
            self.advance_step()

    def control_loop(self):
        start_time = time.time()

        try:
            if not self.task_ready or self.task_finished:
                if self.use_visualizer:
                    self.update_visualizer(None, None, None, [])
                return

            step = self.current_step()
            if step is None:
                return

            if step.kind in ("gripper", "wait"):
                self.run_non_motion_step()
                if self.use_visualizer:
                    self.update_visualizer(None, None, None, [])
                return

            try:
                wp = self.current_waypoint()
            except Exception as exc:
                self.get_logger().warning(f"Waiting for target resolution: {exc}")
                return

            if wp is None:
                self.advance_step()
                return

            start_cfg = self.actual_q.copy() if self.rtde_ready else self.current_q.copy()
            self.current_q = start_cfg.copy()

            world_coll, active_obs = self.build_world_collisions()

            try:
                sol_traj, sol_pos, sol_wxyz = pks.solve_online_planning(
                    robot=self.robot,
                    robot_coll=self.robot_coll,
                    world_coll=world_coll,
                    target_link_name=self.target_link_name,
                    target_position=wp.pos,
                    target_wxyz=wp.wxyz,
                    timesteps=self.traj_len,
                    dt=self.dt,
                    start_cfg=start_cfg,
                    prev_sols=self.sol_traj,
                )
                self.sol_traj = np.array(sol_traj)
            except Exception as exc:
                self.get_logger().warning(f"Planner failed: {exc}")
                return

            cmd_idx = 1 if len(self.sol_traj) > 1 else 0
            q_cmd = np.array(self.sol_traj[cmd_idx], dtype=float)
            self.publish_or_send_joint_command(q_cmd)

            ee_pos_now = None
            if sol_pos is not None and len(sol_pos) > cmd_idx:
                ee_pos_now = np.array(sol_pos[cmd_idx], dtype=float)

            if ee_pos_now is not None:
                self.advance_move_progress(ee_pos_now)

            self.last_elapsed_ms = (time.time() - start_time) * 1000.0
            if self.use_visualizer:
                self.update_visualizer(wp, sol_pos, sol_wxyz, active_obs)

        except Exception as exc:
            self.get_logger().error(f"Unhandled control_loop error: {exc}")

    def update_visualizer(self, wp: Optional[PoseWaypoint], sol_pos, sol_wxyz, active_obstacles):
        step = self.current_step()
        status = "finished" if self.task_finished else (step.name if step is not None else "idle")
        self.status_handle.value = status
        self.timing_handle.value = 0.99 * self.timing_handle.value + 0.01 * self.last_elapsed_ms
        self.step_handle.value = float(self.current_step_idx)
        self.wp_handle.value = float(self.current_wp_idx)

        if wp is not None:
            self.target_handle.position = tuple(wp.pos.tolist())
            self.target_handle.wxyz = tuple(wp.wxyz.tolist())

        if len(active_obstacles) > 0:
            self.obstacle_handle.position = tuple(active_obstacles[0].center.tolist())
            # self.get_logger().info(
            #     f"visualizing obstacle at {np.round(active_obstacles[0].center, 3)}"
            # )
        else:
            self.obstacle_handle.position = (10.0, 10.0, 10.0)
            # self.get_logger().info("no active obstacle for visualizer")

        if sol_pos is not None and sol_wxyz is not None:
            pos_arr = np.array(sol_pos)
            wxyz_arr = np.array(sol_wxyz)
            if hasattr(self.target_frame_handle, "batched_positions"):
                self.target_frame_handle.batched_positions = pos_arr
                self.target_frame_handle.batched_wxyzs = wxyz_arr
            else:
                self.target_frame_handle.positions_batched = pos_arr
                self.target_frame_handle.wxyzs_batched = wxyz_arr

    def destroy_node(self):
        try:
            if self.rtde_c is not None:
                self.rtde_c.servoStop()
                self.rtde_c.stopScript()
        except Exception:
            pass
        try:
            if self.gripper is not None:
                self.gripper.close_serial()
        except Exception:
            pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = PyrokiRtdeTaskPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()