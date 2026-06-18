from launch import LaunchDescription
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description():
    pkg_share = Path("/home/dxr-labtop/pyroki/for_tips/dynamic_task_planner/config")

    return LaunchDescription([
        Node(
            package="dynamic_task_planner",
            executable="dynamic_task_planner_node",
            name="dynamic_task_planner",
            output="screen",
            parameters=[{
                "scenario_yaml": str(pkg_share / "scenario.yaml"),
                "planner_yaml": str(pkg_share / "planner.yaml"),
                "extrinsics_yaml": str(pkg_share / "extrinsics.yaml"),
                "rtde_yaml": str(pkg_share / "rtde.yaml"),
            }],
        )
    ])