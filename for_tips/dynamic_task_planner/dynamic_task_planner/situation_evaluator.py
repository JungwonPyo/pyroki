from dataclasses import dataclass
from typing import Any, Dict, List

from .enums import ActionMode, SituationCode
from .models import Decision, SceneSnapshot, StepDefinition


@dataclass
class SituationSettings:
    situation_map: Dict[str, ActionMode]
    situation_priority: List[str]

    @classmethod
    def from_yaml(cls, cfg: Dict[str, Any]) -> "SituationSettings":
        p = cfg["planner"]
        raw_map = p["situation_map"]
        mapped = {k: ActionMode(v) for k, v in raw_map.items()}
        return cls(
            situation_map=mapped,
            situation_priority=list(p["situation_priority"]),
        )


class SituationEvaluator:
    def __init__(self, settings: SituationSettings):
        self.settings = settings

    @classmethod
    def from_yaml(cls, cfg: Dict[str, Any]) -> "SituationEvaluator":
        return cls(SituationSettings.from_yaml(cfg))

    def evaluate(self, scene: SceneSnapshot, step: StepDefinition) -> Decision:
        label = (scene.situation_label or "NORMAL").upper()
        if label not in self.settings.situation_map:
            label = "NORMAL"

        action = self.settings.situation_map[label]
        return Decision(
            step_id=step.step_id,
            situation=SituationCode(label),
            action=action,
            reason=f"Situation {label} mapped to {action.value}",
        )

    def _collect_blockers(self, scene: SceneSnapshot) -> List[str]:
        blockers = []
        for obj in scene.objects.values():
            if obj.class_name in {"hand", "screwdriver", "wrench", "cable bundle", "plastic tray"}:
                blockers.append(obj.object_id)
        return blockers