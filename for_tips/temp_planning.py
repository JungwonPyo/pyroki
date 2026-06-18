import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import os
import numpy as np
from yourdfpy import URDF, urdf
import pyroki as pk
import rclpy
import viser
from rclpy.node import Node

from pyroki.collision import HalfSpace, RobotCollision, Sphere, Box
from robot_descriptions.loaders.yourdfpy import load_robot_description
from scene_understanding_msgs.msg import SceneContext
from viser.extras import ViserUrdf

import pyroki_snippets as pks


@dataclass
class ObstacleState:
    object_id: str
    class_name: str
    center: np.ndarray
    size: np.ndarray
    score: float
    stamp_sec: float


class PyrokiPathFollower(Node):
    def __init__(self):
        super().__init__("pyroki_path_follower")

        self.declare_parameter("scene_topic", "/scene_context")
        self.declare_parameter("target_object_id", "")
        self.declare_parameter("target_class_name", "person")
        self.declare_parameter("min_score", 0.5)
        self.declare_parameter("obstacle_timeout_sec", 0.5)
        self.declare_parameter("bbox_inflation_m", 0.001)
        self.declare_parameter("waypoint_reached_pos_tol", 0.05)
        self.declare_parameter("control_dt", 0.1)
        self.declare_parameter("traj_len", 8)
        self.declare_parameter("stall_cycles_to_advance", 8)
        self.declare_parameter("use_box_collision", True)
        self.declare_parameter("keep_obstacle_slot", True)

        scene_topic = self.get_parameter("scene_topic").value
        self.target_object_id = self.get_parameter("target_object_id").value
        self.target_class_name = self.get_parameter("target_class_name").value
        self.min_score = float(self.get_parameter("min_score").value)
        self.obstacle_timeout_sec = float(self.get_parameter("obstacle_timeout_sec").value)
        self.bbox_inflation_m = float(self.get_parameter("bbox_inflation_m").value)
        self.waypoint_reached_pos_tol = float(self.get_parameter("waypoint_reached_pos_tol").value)
        self.dt = float(self.get_parameter("control_dt").value)
        self.traj_len = int(self.get_parameter("traj_len").value)
        self.stall_cycles_to_advance = int(self.get_parameter("stall_cycles_to_advance").value)
        self.use_box_collision = bool(self.get_parameter("use_box_collision").value)
        self.keep_obstacle_slot = bool(self.get_parameter("keep_obstacle_slot").value)

        # urdf = load_robot_description("ur5e_description")
        # self.robot = pk.Robot.from_urdf(urdf)
        # Use absolute path to avoid working-directory issues
        urdf_path = os.path.join(os.path.dirname(__file__), "ur5e_with_robotiq.urdf")

        # Load with urdfpy
        urdf = URDF.load(urdf_path)

        # Pass the URDF object to PyRoki
        self.robot = pk.Robot.from_urdf(urdf)
        
        self.target_link_name = "tool0"  
        self.robot_coll = RobotCollision.from_urdf(urdf)

        self.plane_coll = HalfSpace.from_point_and_normal(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

        q0 = self.robot.joint_var_cls.default_factory()
        self.current_q = np.array(q0)
        self.sol_traj = np.array(q0[None].repeat(self.traj_len, axis=0))

        # self.global_waypoints: List[Tuple[np.ndarray, np.ndarray]] = [
        #     (np.array([0.40, -0.20, 0.45]), np.array([0.0, 0.0, 1.0, 0.0])),
        #     (np.array([0.45, -0.05, 0.40]), np.array([0.0, 0.0, 1.0, 0.0])),
        #     (np.array([0.50,  0.10, 0.35]), np.array([0.0, 0.0, 1.0, 0.0])),
        #     (np.array([0.55,  0.20, 0.30]), np.array([0.0, 0.0, 1.0, 0.0])),
        # ]
        self.global_waypoints = [
            (np.array([-0.35, -0.20, 0.30]), np.array([0.0, 0.0, 1.0, 0.0])),
            (np.array([-0.40, -0.05, 0.30]), np.array([0.0, 0.0, 1.0, 0.0])),
            (np.array([-0.45,  0.10, 0.30]), np.array([0.0, 0.0, 1.0, 0.0])),
            (np.array([-0.40,  0.20, 0.35]), np.array([0.0, 0.0, 1.0, 0.0])),
        ]
        self.current_wp_idx = 0

        self._lock = threading.Lock()
        self._planner_frame: Optional[str] = None
        self._active_obstacle: Optional[ObstacleState] = None

        self.prev_wp_dist = float("inf")
        self.wp_stall_count = 0
        self.box_supported = True

        self.subscription = self.create_subscription(
            SceneContext,
            scene_topic,
            self.scene_callback,
            10,
        )

        self.setup_visualizer(urdf)
        self.prewarm_planner()
        self.control_timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(f"Subscribed to {scene_topic}")

    def setup_visualizer(self, urdf):
        self.server = viser.ViserServer()
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self.urdf_vis = ViserUrdf(self.server, urdf, root_node_name="/robot")

        self.base_obstacle_sphere = Sphere.from_center_and_radius(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.05]),
        )
        self.server.scene.add_mesh_trimesh(
            "/obstacle/mesh", mesh=self.base_obstacle_sphere.to_trimesh()
        )

        wp_pos, wp_wxyz = self.current_waypoint()
        self.target_handle = self.server.scene.add_transform_controls(
            "/ik_target",
            scale=0.2,
            position=tuple(wp_pos.tolist()),
            wxyz=tuple(wp_wxyz.tolist()),
        )

        self.obstacle_handle = self.server.scene.add_transform_controls(
            "/obstacle",
            scale=0.2,
            position=(10.0, 10.0, 10.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )

        self.target_frame_handle = self.server.scene.add_batched_axes(
            "/planned_frames",
            axes_length=0.05,
            axes_radius=0.005,
            batched_positions=np.zeros((max(self.traj_len, 1), 3)),
            batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * max(self.traj_len, 1)),
        )

        self.timing_handle = self.server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        self.waypoint_handle = self.server.gui.add_number("Current waypoint", float(self.current_wp_idx), disabled=True)

    def scene_callback(self, msg: SceneContext):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        selected = None

        for obj in msg.objects:
            if obj.score < self.min_score:
                continue
            if not obj.bbox_3d.valid:
                continue

            id_ok = bool(self.target_object_id) and (obj.id == self.target_object_id)
            class_ok = bool(self.target_class_name) and (obj.class_name == self.target_class_name)

            if id_ok or class_ok:
                center = np.array([
                    obj.bbox_3d.center.x,
                    obj.bbox_3d.center.y,
                    obj.bbox_3d.center.z,
                ], dtype=float)
                size = np.array([
                    obj.bbox_3d.size.x,
                    obj.bbox_3d.size.y,
                    obj.bbox_3d.size.z,
                ], dtype=float)

                selected = ObstacleState(
                    object_id=obj.id,
                    class_name=obj.class_name,
                    center=center,
                    size=size,
                    score=float(obj.score),
                    stamp_sec=now_sec,
                )
                break

        with self._lock:
            self._planner_frame = msg.planner_frame
            self._active_obstacle = selected

    def current_waypoint(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.global_waypoints[self.current_wp_idx % len(self.global_waypoints)]

    def obstacle_is_fresh(self, obs: Optional[ObstacleState], now_sec: float) -> bool:
        return obs is not None and (now_sec - obs.stamp_sec) <= self.obstacle_timeout_sec

    def bbox_to_sphere(self, obs: ObstacleState) -> Sphere:
        radius = 0.5 * np.max(obs.size) + self.bbox_inflation_m
        sphere = Sphere.from_center_and_radius(
            np.array([0.0, 0.0, 0.0]),
            np.array([radius]),
        )
        return sphere.transform_from_wxyz_position(
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=obs.center,
        )

    def bbox_to_box(self, obs: ObstacleState) -> Box:
        extent = obs.size + 2.0 * self.bbox_inflation_m
        return Box.from_extent(
            extent=extent,
            position=obs.center,
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )

    def obstacle_geom(self, obs: ObstacleState):
        if self.use_box_collision and self.box_supported:
            try:
                return self.bbox_to_box(obs)
            except Exception as exc:
                self.box_supported = False
                self.get_logger().warning(f"Box collision disabled, fallback to sphere: {exc}")
        return self.bbox_to_sphere(obs)

    def inactive_obstacle_geom(self):
        fake = ObstacleState(
            object_id="inactive",
            class_name="inactive",
            center=np.array([10.0, 10.0, 10.0]),
            size=np.array([0.02, 0.02, 0.02]),
            score=0.0,
            stamp_sec=0.0,
        )
        return self.obstacle_geom(fake)

    def prewarm_planner(self):
        target_pos, target_wxyz = self.current_waypoint()
        world_coll_list = [self.plane_coll]
        if self.keep_obstacle_slot:
            world_coll_list.append(self.inactive_obstacle_geom())

        t0 = time.time()
        try:
            sol_traj, _, _ = pks.solve_online_planning(
                robot=self.robot,
                robot_coll=self.robot_coll,
                world_coll=world_coll_list,
                target_link_name=self.target_link_name,
                target_position=target_pos,
                target_wxyz=target_wxyz,
                timesteps=self.traj_len,
                dt=self.dt,
                start_cfg=self.current_q,
                prev_sols=self.sol_traj,
            )
            self.sol_traj = np.array(sol_traj)
            self.get_logger().info(f"Prewarm done in {(time.time() - t0):.3f}s")
        except Exception as exc:
            self.get_logger().warning(f"Prewarm failed: {exc}")

    def advance_waypoint_if_reached(self, ee_pos: np.ndarray):
        target_pos, _ = self.current_waypoint()
        dist = np.linalg.norm(ee_pos - target_pos)

        if dist < self.waypoint_reached_pos_tol:
            self.current_wp_idx = (self.current_wp_idx + 1) % len(self.global_waypoints)
            self.prev_wp_dist = float("inf")
            self.wp_stall_count = 0
            self.get_logger().info(f"Advance to waypoint {self.current_wp_idx} (tolerance)")
            return

        if dist < self.prev_wp_dist - 1e-3:
            self.wp_stall_count = 0
        else:
            self.wp_stall_count += 1

        self.prev_wp_dist = dist

        if self.wp_stall_count >= self.stall_cycles_to_advance:
            self.current_wp_idx = (self.current_wp_idx + 1) % len(self.global_waypoints)
            self.prev_wp_dist = float("inf")
            self.wp_stall_count = 0
            self.get_logger().info(f"Advance to waypoint {self.current_wp_idx} (stall)")

    def publish_or_send_joint_command(self, q_cmd: np.ndarray):
        self.get_logger().debug(f"q_cmd={np.round(q_cmd, 4)}")

    def update_visualizer(self, q_cmd, sol_pos, sol_wxyz, obs, elapsed_ms):
        self.urdf_vis.update_cfg(q_cmd)

        wp_pos, wp_wxyz = self.current_waypoint()
        self.target_handle.position = tuple(wp_pos.tolist())
        self.target_handle.wxyz = tuple(wp_wxyz.tolist())

        if obs is not None:
            self.obstacle_handle.position = tuple(obs.center.tolist())
        else:
            self.obstacle_handle.position = (10.0, 10.0, 10.0)

        if sol_pos is not None and sol_wxyz is not None:
            pos_arr = np.array(sol_pos)
            wxyz_arr = np.array(sol_wxyz)
            if hasattr(self.target_frame_handle, "batched_positions"):
                self.target_frame_handle.batched_positions = pos_arr
                self.target_frame_handle.batched_wxyzs = wxyz_arr
            else:
                self.target_frame_handle.positions_batched = pos_arr
                self.target_frame_handle.wxyzs_batched = wxyz_arr

        self.timing_handle.value = 0.99 * self.timing_handle.value + 0.01 * elapsed_ms
        self.waypoint_handle.value = float(self.current_wp_idx)

    def control_loop(self):
        start_time = time.time()
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        with self._lock:
            obs = self._active_obstacle
            planner_frame = self._planner_frame

        target_pos, target_wxyz = self.current_waypoint()
        world_coll_list = [self.plane_coll]
        active_obs = None

        if self.keep_obstacle_slot:
            if self.obstacle_is_fresh(obs, now_sec):
                world_coll_list.append(self.obstacle_geom(obs))
                active_obs = obs
            else:
                world_coll_list.append(self.inactive_obstacle_geom())
        else:
            if self.obstacle_is_fresh(obs, now_sec):
                world_coll_list.append(self.obstacle_geom(obs))
                active_obs = obs

        try:
            sol_traj, sol_pos, sol_wxyz = pks.solve_online_planning(
                robot=self.robot,
                robot_coll=self.robot_coll,
                world_coll=world_coll_list,
                target_link_name=self.target_link_name,
                target_position=target_pos,
                target_wxyz=target_wxyz,
                timesteps=self.traj_len,
                dt=self.dt,
                start_cfg=self.current_q,
                prev_sols=self.sol_traj,
            )
            self.sol_traj = np.array(sol_traj)
        except Exception as exc:
            self.get_logger().warning(f"Planner failed: {exc}")
            return

        step_idx = 1 if len(self.sol_traj) > 1 else 0
        q_cmd = np.array(self.sol_traj[step_idx])
        self.current_q = q_cmd
        self.publish_or_send_joint_command(q_cmd)

        if sol_pos is not None and len(sol_pos) > step_idx:
            ee_pos_now = np.array(sol_pos[step_idx])
            self.advance_waypoint_if_reached(ee_pos_now)

        elapsed_ms = (time.time() - start_time) * 1000.0
        self.update_visualizer(q_cmd, sol_pos, sol_wxyz, active_obs, elapsed_ms)


def main():
    rclpy.init()
    node = PyrokiPathFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()