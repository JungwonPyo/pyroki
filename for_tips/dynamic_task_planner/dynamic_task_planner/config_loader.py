from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


class ConfigBundle:
    def __init__(
        self,
        scenario_cfg: Dict[str, Any],
        planner_cfg: Dict[str, Any],
        extrinsics_cfg: Dict[str, Any],
        rtde_cfg: Dict[str, Any],
    ):
        self.scenario = scenario_cfg
        self.planner = planner_cfg
        self.extrinsics = extrinsics_cfg
        self.rtde = rtde_cfg

    @classmethod
    def from_files(
        cls,
        scenario_path: str,
        planner_path: str,
        extrinsics_path: str,
        rtde_path: str,
    ) -> "ConfigBundle":
        return cls(
            scenario_cfg=load_yaml_file(scenario_path),
            planner_cfg=load_yaml_file(planner_path),
            extrinsics_cfg=load_yaml_file(extrinsics_path),
            rtde_cfg=load_yaml_file(rtde_path),
        )