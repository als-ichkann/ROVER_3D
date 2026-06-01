#!/usr/bin/env python3
"""
Planning Launch：GMM 宏观规划 + 微观轨迹生成。
"""
from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _parse_robot_names(raw: str) -> List[str]:
    return [n.strip() for n in raw.split(",") if n.strip()] or ["bot1"]


def _planning_setup(context, *args, **kwargs):
    robots_raw = LaunchConfiguration("robots").perform(context)
    robot_names = _parse_robot_names(robots_raw)
    config_file = LaunchConfiguration("config_file")
    publish_goal = LaunchConfiguration("publish_goal").perform(context).lower() in ("1", "true", "yes")
    goal_config = LaunchConfiguration("goal_config_file")

    actions = [
        Node(
            package="rover3d_navigation",
            executable="planning_node.py",
            name="planning_node",
            output="screen",
            emulate_tty=True,
            additional_env={"PYTHONUNBUFFERED": "1"},
            parameters=[config_file, {"robot_names": robot_names}],
        )
    ]

    if publish_goal:
        actions.append(
            TimerAction(
                period=1.5,
                actions=[
                    ExecuteProcess(
                        cmd=[
                            "ros2",
                            "run",
                            "rover3d_navigation",
                            "publish_gmm_goal.py",
                            "--config",
                            goal_config,
                            "--sleep",
                            "0.8",
                        ],
                        output="screen",
                    )
                ],
            )
        )
    return actions


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("rover3d_navigation")
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
                        pkg_share,
                        "config",
                        "planning.yaml",
                    ]
                ),
                description="YAML 配置文件，包含 planning_node 参数",
            ),
            DeclareLaunchArgument(
                "publish_goal",
                default_value="true",
                description="启动后自动向 /goal_gmm 发布一次 GMM 目标（来自 goal_config_file）",
            ),
            DeclareLaunchArgument(
                "goal_config_file",
                default_value=PathJoinSubstitution(
                    [
                        pkg_share,
                        "config",
                        "gmm_goal_publisher.yaml",
                    ]
                ),
                description="GMM 目标 YAML（publish_gmm_goal.py 同格式）",
            ),
            OpaqueFunction(function=_planning_setup),
        ]
    )
