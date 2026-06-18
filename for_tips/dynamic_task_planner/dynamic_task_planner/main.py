import threading
import rclpy
from rclpy.node import Node

from scene_understanding_msgs.msg import SceneContext

from .config_loader import ConfigBundle
from .planner_settings import PlannerSettings
from .scene_adapter import SceneAdapter
from .scenario_manager import ScenarioManager
from .situation_evaluator import SituationEvaluator
from .pyroki_local_planner import PyrokiLocalPlanner
from .ur_rtde_bridge import URRTDEBridge, RTDESettings
from .enums import ActionMode


class DynamicTaskPlannerNode(Node):
    def __init__(self):
        super().__init__("dynamic_task_planner")

        self.declare_parameter("scenario_yaml", "")
        self.declare_parameter("planner_yaml", "")
        self.declare_parameter("extrinsics_yaml", "")
        self.declare_parameter("rtde_yaml", "")

        scenario_yaml = self.get_parameter("scenario_yaml").value
        planner_yaml = self.get_parameter("planner_yaml").value
        extrinsics_yaml = self.get_parameter("extrinsics_yaml").value
        rtde_yaml = self.get_parameter("rtde_yaml").value

        self.cfg = ConfigBundle.from_files(
            scenario_path=scenario_yaml,
            planner_path=planner_yaml,
            extrinsics_path=extrinsics_yaml,
            rtde_path=rtde_yaml,
        )

        self.planner_settings = PlannerSettings.from_yaml(self.cfg.planner)
        self.adapter = SceneAdapter.from_yaml(self.cfg.extrinsics)
        self.scenario = ScenarioManager.from_yaml(self.cfg.scenario)
        self.evaluator = SituationEvaluator.from_yaml(self.cfg.planner)
        self.rtde = URRTDEBridge.from_yaml(self.cfg.rtde)
        self.local_planner = PyrokiLocalPlanner(
            dt=self.planner_settings.nominal_plan_dt,
            traj_len=self.planner_settings.traj_len,
            bbox_inflation_m=self.planner_settings.bbox_inflation_m,
        )

        self.rtde.connect()

        self._lock = threading.Lock()
        self._scene = None

        self.subscription = self.create_subscription(
            SceneContext,
            self.planner_settings.scene_topic,
            self.scene_callback,
            10,
        )
        self.timer = self.create_timer(
            self.planner_settings.nominal_plan_dt,
            self.control_loop,
        )

    def scene_callback(self, msg: SceneContext):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        scene = self.adapter.from_msg(msg, now_sec)
        with self._lock:
            self._scene = scene

    def control_loop(self):
        with self._lock:
            scene = self._scene

        if scene is None:
            return

        step = self.scenario.current_step()
        decision = self.evaluator.evaluate(scene, step)

        if decision.action in {ActionMode.WAIT, ActionMode.DELAY_ENTRY}:
            return

        if decision.action == ActionMode.STOP:
            self.rtde.stop_motion()
            return

        nominal_target_pos, nominal_target_wxyz = self._nominal_target_from_yaml(step.step_id.value)
        obstacles = self._select_obstacles(scene)

        result = self.local_planner.plan(
            decision=decision,
            step=step,
            nominal_target_pos=nominal_target_pos,
            nominal_target_wxyz=nominal_target_wxyz,
            obstacles=obstacles,
        )

        if not result.success or result.q_cmd is None:
            if self.planner_settings.stop_on_planner_failure:
                self.rtde.stop_motion()
            return

        self.rtde.servo_joint(result.q_cmd)

        reached_goal = self._goal_reached(result, nominal_target_pos)
        self.scenario.advance_if_complete(reached_goal)

    def _nominal_target_from_yaml(self, step_id: str):
        targets = self.cfg.planner["planner"]["nominal_targets"]
        t = targets[step_id]
        import numpy as np
        return np.array(t["position_xyz"], dtype=float), np.array(t["quaternion_wxyz"], dtype=float)

    def _select_obstacles(self, scene):
        return [
            obj for obj in scene.objects.values()
            if obj.class_name in self.cfg.planner["planner"]["replanning"]["class_whitelist_for_avoidance"]
        ]

    def _goal_reached(self, result, nominal_target_pos, tol=None):
        import numpy as np
        if tol is None:
            tol = self.planner_settings.waypoint_reached_pos_tol
        if result.ee_positions is None or len(result.ee_positions) == 0:
            return False
        ee = np.array(result.ee_positions[min(1, len(result.ee_positions) - 1)])
        return np.linalg.norm(ee - nominal_target_pos) < tol

def main():
    rclpy.init()
    node = DynamicTaskPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()