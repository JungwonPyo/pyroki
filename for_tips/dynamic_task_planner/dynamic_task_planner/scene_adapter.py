from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np

from .models import ObjectState, RelationState, SceneSnapshot


@dataclass
class ExtrinsicsSettings:
    planner_frame: str
    parent_frame: str
    child_frame: str
    translation_xyz: np.ndarray
    quaternion_xyzw: np.ndarray
    use_message_bbox_frame_id: bool
    assume_identity_if_same_frame: bool
    reject_unknown_frames: bool
    transform_bbox_corners: bool
    rebuild_axis_aligned_bbox_in_planner_frame: bool

    @classmethod
    def from_yaml(cls, cfg: Dict[str, Any]) -> "ExtrinsicsSettings":
        e = cfg["extrinsics"]
        c2b = e["camera_to_base"]
        policy = e["transform_policy"]
        return cls(
            planner_frame=e["planner_frame"],
            parent_frame=c2b["parent_frame"],
            child_frame=c2b["child_frame"],
            translation_xyz=np.array(c2b["translation_xyz"], dtype=float),
            quaternion_xyzw=np.array(c2b["quaternion_xyzw"], dtype=float),
            use_message_bbox_frame_id=bool(policy["use_message_bbox_frame_id"]),
            assume_identity_if_same_frame=bool(policy["assume_identity_if_same_frame"]),
            reject_unknown_frames=bool(policy["reject_unknown_frames"]),
            transform_bbox_corners=bool(policy["transform_bbox_corners"]),
            rebuild_axis_aligned_bbox_in_planner_frame=bool(policy["rebuild_axis_aligned_bbox_in_planner_frame"]),
        )


class SceneAdapter:
    def __init__(self, settings: ExtrinsicsSettings):
        self.settings = settings

    @classmethod
    def from_yaml(cls, cfg: Dict[str, Any]) -> "SceneAdapter":
        return cls(ExtrinsicsSettings.from_yaml(cfg))

    def from_msg(self, msg, now_sec: float) -> SceneSnapshot:
        objects = {}
        relations = []

        for obj in msg.objects:
            bbox = obj.bbox_3d
            if not bbox.valid:
                continue

            center = np.array([bbox.center.x, bbox.center.y, bbox.center.z], dtype=float)
            size = np.array([bbox.size.x, bbox.size.y, bbox.size.z], dtype=float)
            min_corner = np.array([bbox.min_corner.x, bbox.min_corner.y, bbox.min_corner.z], dtype=float)
            max_corner = np.array([bbox.max_corner.x, bbox.max_corner.y, bbox.max_corner.z], dtype=float)

            state = ObjectState(
                object_id=obj.id,
                class_name=obj.class_name,
                score=float(obj.score),
                frame_id=bbox.frame_id,
                center=center,
                size=size,
                min_corner=min_corner,
                max_corner=max_corner,
                stamp_sec=now_sec,
                valid=True,
                source_method=bbox.method,
            )
            objects[obj.id] = self._normalize_object_frame(state)

        for rel in msg.relationships:
            relations.append(
                RelationState(
                    subject_id=rel.subject_id,
                    predicate=rel.predicate,
                    object_id=rel.object_id,
                    score=float(rel.score),
                )
            )

        return SceneSnapshot(
            scene_id=msg.scene_id,
            planner_frame=msg.planner_frame or self.settings.planner_frame,
            stamp_sec=now_sec,
            situation_label=msg.situation.label,
            situation_confidence=float(msg.situation.confidence),
            objects=objects,
            relations=relations,
        )

    def _normalize_object_frame(self, obj: ObjectState) -> ObjectState:
        if obj.frame_id == self.settings.planner_frame and self.settings.assume_identity_if_same_frame:
            return obj

        if self.settings.reject_unknown_frames:
            allowed = {self.settings.child_frame, self.settings.parent_frame, self.settings.planner_frame}
            if obj.frame_id not in allowed:
                raise ValueError(f"Unknown bbox frame: {obj.frame_id}")

        if obj.frame_id == self.settings.child_frame:
            corners = self._bbox_corners(obj.min_corner, obj.max_corner)
            corners_tf = np.array([self._transform_point(p) for p in corners])
            min_corner = np.min(corners_tf, axis=0)
            max_corner = np.max(corners_tf, axis=0)
            center = 0.5 * (min_corner + max_corner)
            size = max_corner - min_corner
            obj.center = center
            obj.size = size
            obj.min_corner = min_corner
            obj.max_corner = max_corner
            obj.frame_id = self.settings.planner_frame
            return obj

        return obj

    def _bbox_corners(self, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
        return np.array([
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ], dtype=float)

    def _transform_point(self, p: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = self.settings.quaternion_xyzw
        tx, ty, tz = self.settings.translation_xyz
        R = self._quat_to_rot(qx, qy, qz, qw)
        return R @ p + np.array([tx, ty, tz], dtype=float)

    def _quat_to_rot(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ], dtype=float)