from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .enums import ActionMode, SituationCode, StepId, StepPhase


@dataclass
class ObjectState:
    object_id: str
    class_name: str
    score: float
    frame_id: str
    center: np.ndarray
    size: np.ndarray
    min_corner: np.ndarray
    max_corner: np.ndarray
    stamp_sec: float
    valid: bool = True
    source_method: str = ""


@dataclass
class RelationState:
    subject_id: str
    predicate: str
    object_id: str
    score: float


@dataclass
class SceneSnapshot:
    scene_id: str
    planner_frame: str
    stamp_sec: float
    situation_label: str
    situation_confidence: float
    objects: Dict[str, ObjectState] = field(default_factory=dict)
    relations: List[RelationState] = field(default_factory=list)


@dataclass
class RelationPattern:
    subject_class: Optional[str] = None
    subject_id: Optional[str] = None
    predicate: Optional[str] = None
    object_class: Optional[str] = None
    object_id: Optional[str] = None
    min_score: float = 0.0


@dataclass
class StepCondition:
    required_classes: List[str] = field(default_factory=list)
    forbidden_classes: List[str] = field(default_factory=list)
    expected_relations: List[RelationPattern] = field(default_factory=list)


@dataclass
class StepDefinition:
    step_id: StepId
    phase: StepPhase
    nominal_name: str
    target_class: Optional[str] = None
    support_class: Optional[str] = None
    destination_class: Optional[str] = None
    replanning_enabled: bool = False
    step_condition: StepCondition = field(default_factory=StepCondition)


@dataclass
class Decision:
    step_id: StepId
    situation: SituationCode
    action: ActionMode
    reason: str
    target_object_id: Optional[str] = None
    attached_object_id: Optional[str] = None
    blocking_object_ids: List[str] = field(default_factory=list)