from typing import Dict, List, Optional

from .models import ObjectState, RelationPattern, RelationState


def find_objects_by_class(objects: Dict[str, ObjectState], class_name: str) -> List[ObjectState]:
    return [obj for obj in objects.values() if obj.class_name == class_name and obj.valid]


def relation_matches(
    rel: RelationState,
    objects: Dict[str, ObjectState],
    pattern: RelationPattern,
) -> bool:
    if rel.score < pattern.min_score:
        return False

    if pattern.predicate is not None and rel.predicate != pattern.predicate:
        return False

    subj = objects.get(rel.subject_id)
    obj = objects.get(rel.object_id)

    if pattern.subject_id is not None and rel.subject_id != pattern.subject_id:
        return False
    if pattern.object_id is not None and rel.object_id != pattern.object_id:
        return False

    if pattern.subject_class is not None:
        if subj is None or subj.class_name != pattern.subject_class:
            return False

    if pattern.object_class is not None:
        if obj is None or obj.class_name != pattern.object_class:
            return False

    return True


def has_relation(
    relations: List[RelationState],
    objects: Dict[str, ObjectState],
    pattern: RelationPattern,
) -> bool:
    return any(relation_matches(rel, objects, pattern) for rel in relations)


def list_relations(
    relations: List[RelationState],
    objects: Dict[str, ObjectState],
    pattern: RelationPattern,
) -> List[RelationState]:
    return [rel for rel in relations if relation_matches(rel, objects, pattern)]