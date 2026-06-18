from typing import Any, Dict, List

from .enums import StepId, StepPhase
from .models import RelationPattern, StepCondition, StepDefinition


class ScenarioManager:
    def __init__(self, steps: List[StepDefinition]):
        self.steps = steps
        self.step_index = 0

    @classmethod
    def from_yaml(cls, cfg: Dict[str, Any]) -> "ScenarioManager":
        scenario = cfg["scenario"]
        steps_cfg = scenario["steps"]
        steps = [cls._parse_step(step_cfg) for step_cfg in steps_cfg]
        return cls(steps)

    @staticmethod
    def _parse_step(step_cfg: Dict[str, Any]) -> StepDefinition:
        pre = step_cfg.get("preconditions", {})
        expected_relations = [
            RelationPattern(
                subject_class=r.get("subject_class"),
                subject_id=r.get("subject_id"),
                predicate=r.get("predicate"),
                object_class=r.get("object_class"),
                object_id=r.get("object_id"),
                min_score=float(r.get("min_score", 0.0)),
            )
            for r in pre.get("expected_relations", [])
        ]

        cond = StepCondition(
            required_classes=pre.get("required_classes", []),
            forbidden_classes=pre.get("forbidden_classes", []),
            expected_relations=expected_relations,
        )

        return StepDefinition(
            step_id=StepId(step_cfg["id"]),
            phase=StepPhase(step_cfg["phase"]),
            nominal_name=step_cfg["nominal_name"],
            target_class=step_cfg.get("target_class"),
            support_class=step_cfg.get("support_class"),
            destination_class=step_cfg.get("destination_class"),
            replanning_enabled=bool(step_cfg.get("replanning_enabled", False)),
            step_condition=cond,
        )

    def current_step(self) -> StepDefinition:
        return self.steps[self.step_index]

    def advance_if_complete(self, reached_goal: bool) -> bool:
        if not reached_goal:
            return False
        if self.step_index + 1 >= len(self.steps):
            return False
        self.step_index += 1
        return True

    def reset(self):
        self.step_index = 0

    def _build_steps(self) -> List[StepDefinition]:
        return [
            StepDefinition(
                step_id=StepId.STEP1_APPROACH_BOX_PICK,
                phase=StepPhase.MOTION,
                nominal_name="step1_approach_box_pick",
                target_class="workpiece",
                support_class="parts box",
                replanning_enabled=True,
                step_condition=StepCondition(
                    required_classes=["workpiece", "parts box"],
                    expected_relations=[
                        RelationPattern(
                            subject_class="workpiece",
                            predicate="inside",
                            object_class="parts box",
                            min_score=0.5,
                        )
                    ],
                ),
            ),
            StepDefinition(
                step_id=StepId.STEP1_GRASP,
                phase=StepPhase.GRIPPER,
                nominal_name="step1_grasp",
                target_class="workpiece",
                support_class="parts box",
                replanning_enabled=False,
            ),
            StepDefinition(
                step_id=StepId.STEP1_LIFT,
                phase=StepPhase.MOTION,
                nominal_name="step1_lift",
                target_class="workpiece",
                replanning_enabled=False,
            ),
            StepDefinition(
                step_id=StepId.STEP2_TRANSFER_HOLDER,
                phase=StepPhase.MOTION,
                nominal_name="step2_transfer_holder",
                target_class="workpiece",
                destination_class="holder",
                replanning_enabled=True,
            ),
            StepDefinition(
                step_id=StepId.STEP2_PLACE_HOLDER,
                phase=StepPhase.GRIPPER,
                nominal_name="step2_place_holder",
                target_class="workpiece",
                destination_class="holder",
                replanning_enabled=False,
            ),
            StepDefinition(
                step_id=StepId.STEP2_RETREAT,
                phase=StepPhase.MOTION,
                nominal_name="step2_retreat",
                destination_class="holder",
                replanning_enabled=False,
            ),
            StepDefinition(
                step_id=StepId.STEP3_APPROACH_HOLDER_PICK,
                phase=StepPhase.MOTION,
                nominal_name="step3_approach_holder_pick",
                target_class="workpiece",
                support_class="holder",
                replanning_enabled=True,
                step_condition=StepCondition(
                    required_classes=["workpiece", "holder"],
                    expected_relations=[
                        RelationPattern(
                            subject_class="workpiece",
                            predicate="inside",
                            object_class="holder",
                            min_score=0.5,
                        )
                    ],
                ),
            ),
            StepDefinition(
                step_id=StepId.STEP3_GRASP,
                phase=StepPhase.GRIPPER,
                nominal_name="step3_grasp",
                target_class="workpiece",
                support_class="holder",
                replanning_enabled=False,
            ),
            StepDefinition(
                step_id=StepId.STEP3_LIFT,
                phase=StepPhase.MOTION,
                nominal_name="step3_lift",
                target_class="workpiece",
                replanning_enabled=False,
            ),
            StepDefinition(
                step_id=StepId.STEP4_TRANSFER_BOX2,
                phase=StepPhase.MOTION,
                nominal_name="step4_transfer_box2",
                target_class="workpiece",
                destination_class="parts box",
                replanning_enabled=True,
            ),
            StepDefinition(
                step_id=StepId.STEP4_PLACE_BOX2,
                phase=StepPhase.GRIPPER,
                nominal_name="step4_place_box2",
                target_class="workpiece",
                destination_class="parts box",
                replanning_enabled=False,
            ),
            StepDefinition(
                step_id=StepId.STEP4_RETREAT_FINISH,
                phase=StepPhase.MOTION,
                nominal_name="step4_retreat_finish",
                replanning_enabled=False,
            ),
        ]