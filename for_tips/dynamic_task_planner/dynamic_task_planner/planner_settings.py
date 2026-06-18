from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class PlannerSettings:
    planner_frame: str
    nominal_plan_dt: float
    traj_len: int
    waypoint_reached_pos_tol: float
    stall_cycles_to_advance: int
    scene_topic: str
    min_object_score: float
    min_relation_score: float
    obstacle_timeout_sec: float
    scene_stale_timeout_sec: float
    required_persistence_frames: int
    use_box_collision: bool
    fallback_to_sphere: bool
    bbox_inflation_m: float
    keep_obstacle_slot: bool
    include_ground_plane: bool
    attached_object_inflation_m: float
    allow_s3_replanning_only: bool
    stop_on_planner_failure: bool
    stop_on_scene_timeout: bool
    situation_priority: List[str]

    @classmethod
    def from_yaml(cls, cfg: Dict[str, Any]) -> "PlannerSettings":
        p = cfg["planner"]
        scene = p["scene"]
        collision = p["collision"]
        behavior = p["behavior"]

        return cls(
            planner_frame=p["planner_frame"],
            nominal_plan_dt=float(p["nominal_plan_dt"]),
            traj_len=int(p["traj_len"]),
            waypoint_reached_pos_tol=float(p["waypoint_reached_pos_tol"]),
            stall_cycles_to_advance=int(p["stall_cycles_to_advance"]),
            scene_topic=scene["topic"],
            min_object_score=float(scene["min_object_score"]),
            min_relation_score=float(scene["min_relation_score"]),
            obstacle_timeout_sec=float(scene["obstacle_timeout_sec"]),
            scene_stale_timeout_sec=float(scene["scene_stale_timeout_sec"]),
            required_persistence_frames=int(scene["required_persistence_frames"]),
            use_box_collision=bool(collision["use_box_collision"]),
            fallback_to_sphere=bool(collision["fallback_to_sphere"]),
            bbox_inflation_m=float(collision["bbox_inflation_m"]),
            keep_obstacle_slot=bool(collision["keep_obstacle_slot"]),
            include_ground_plane=bool(collision["include_ground_plane"]),
            attached_object_inflation_m=float(collision["attached_object_inflation_m"]),
            allow_s3_replanning_only=bool(behavior["allow_s3_replanning_only"]),
            stop_on_planner_failure=bool(behavior["stop_on_planner_failure"]),
            stop_on_scene_timeout=bool(behavior["stop_on_scene_timeout"]),
            situation_priority=list(p["situation_priority"]),
        )