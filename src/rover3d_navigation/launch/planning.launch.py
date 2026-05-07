#!/usr/bin/env python3
"""
Planning Launch：GMM 宏观规划 + 微观轨迹生成。
"""
from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _parse_robot_names(raw: str) -> List[str]:
    return [n.strip() for n in raw.split(",") if n.strip()] or ["bot1"]


def _planning_setup(context, *args, **kwargs):
    robots_raw = LaunchConfiguration("robots").perform(context)
    robot_names = _parse_robot_names(robots_raw)
    config_file = LaunchConfiguration("config_file")
    return [
        Node(
            package="rover3d_navigation",
            executable="planning_node.py",
            name="planning_node",
            output="screen",
            parameters=[config_file, {"robot_names": robot_names}],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robots",
                default_value="bot1,bot2,bot3,bot4",
                description="逗号分隔的机器人命名空间，如 bot1,bot2,bot3",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("rover3d_navigation"),
                        "config",
                        "planning.yaml",
                    ]
                ),
                description="YAML 配置文件，包含 planning_node 参数",
            ),
            OpaqueFunction(function=_planning_setup),
        ]
    )
