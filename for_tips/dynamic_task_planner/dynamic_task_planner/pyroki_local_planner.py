from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pyroki as pk

from pyroki.collision import Box, HalfSpace, RobotCollision, Sphere
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroki_snippets as pks

from .enums import ActionMode
from .models import Decision, ObjectState, StepDefinition


@dataclass
class PlannerResult:
    success: bool
    q_cmd: Optional[np.ndarray]
    sol_traj: Optional[np.ndarray]
    ee_positions: Optional[np.ndarray]
    ee_wxyzs: Optional[np.ndarray]
    reason: str = ""


class PyrokiLocalPlanner:
    def __init__(self, dt: float = 0.1, traj_len: int = 8, bbox_inflation_m: float = 0.01):
        urdf = load_robot_description("ur5e_description")
        self.target_link_name = "tool0"
        self.robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = RobotCollision.from_urdf(urdf)
        self.plane_coll = HalfSpace.from_point_and_normal(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        q0 = self.robot.joint_var_cls.default_factory()
        self.current_q = np.array(q0)
        self.sol_traj = np.array(q0[None].repeat(traj_len, axis=0))
        self.dt = dt
        self.traj_len = traj_len
        self.bbox_inflation_m = bbox_inflation_m

    def plan(
        self,
        decision: Decision,
        step: StepDefinition,
        nominal_target_pos: np.ndarray,
        nominal_target_wxyz: np.ndarray,
        obstacles: List[ObjectState],
    ) -> PlannerResult:
        if decision.action in {ActionMode.WAIT, ActionMode.STOP, ActionMode.DELAY_ENTRY}:
            return PlannerResult(False, None, None, None, None, reason=decision.action.value)

        world_coll = [self.plane_coll]

        if decision.action == ActionMode.AVOID:
            for obs in obstacles:
                world_coll.append(self._bbox_to_box(obs))

        try:
            sol_traj, sol_pos, sol_wxyz = pks.solve_online_planning(
                robot=self.robot,
                robot_coll=self.robot_coll,
                world_coll=world_coll,
                target_link_name=self.target_link_name,
                target_position=nominal_target_pos,
                target_wxyz=nominal_target_wxyz,
                timesteps=self.traj_len,
                dt=self.dt,
                start_cfg=self.current_q,
                prev_sols=self.sol_traj,
            )
            self.sol_traj = np.array(sol_traj)
            step_idx = 1 if len(self.sol_traj) > 1 else 0
            q_cmd = np.array(self.sol_traj[step_idx])
            self.current_q = q_cmd
            return PlannerResult(True, q_cmd, np.array(sol_traj), np.array(sol_pos), np.array(sol_wxyz))
        except Exception as exc:
            return PlannerResult(False, None, None, None, None, reason=str(exc))

    def _bbox_to_box(self, obs: ObjectState) -> Box:
        extent = obs.size + 2.0 * self.bbox_inflation_m
        return Box.from_extent(
            extent=extent,
            position=obs.center,
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )