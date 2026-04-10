#!/usr/bin/env python3
"""
Planning APF Launch：GMM 宏观规划 + APF 轨迹生成

仅启动规划相关节点：GMM 目标发布（可选）和 planning_apf 轨迹规划。
ESDF 建图、MPC 控制等由其他 launch 负责启动。

使用方法:
  ros2 launch rover3d_navigation planning_apf.launch.py
  ros2 launch rover3d_navigation planning_apf.launch.py robots:=bot1,bot2

GMM 目标需手动发布（与 config/gmm_goal_publisher.yaml 一致）:
  ros2 run rover3d_navigation publish_gmm_goal.py

逻辑:
  1. planning_apf_node: 订阅 apf_goal、/{robot}/global_odom（Swarm-LIO2+map_fusion），订阅 /esdf/grid_full（FIESTA 发布的完整 ESDF 栅格）做 local 避障查询，
     运行 SLP 宏观优化 + APF 势场，发布 /{robot}/apf_trajectory
     （use_esdf_grid_cache:=true 时订阅 topic；false 时改调用 esdf/query 服务）
"""
from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _parse_robot_names(raw: str) -> List[str]:
    return [n.strip() for n in raw.split(",") if n.strip()] or ["bot1"]


def _planning_apf_setup(context, *args, **kwargs):
    robots_raw = LaunchConfiguration("robots").perform(context)
    robot_names = _parse_robot_names(robots_raw)
    config_file = LaunchConfiguration("config_file")
    nodes = []

    # Planning APF 节点（GMM 由 ros2 run rover3d_navigation publish_gmm_goal.py 手动发布）
    nodes.append(
        Node(
            package="rover3d_navigation",
            executable="planning_apf_node.py",
            name="planning_apf_node",
            output="screen",
            parameters=[
                config_file,
                {"robot_names": robot_names},
            ],
        )
    )

    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "robots",
            default_value="bot1,bot2,bot3,bot4",
            description="逗号分隔的机器人命名空间，如 bot1,bot2,bot3",
        ),
        DeclareLaunchArgument(
            "config_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("rover3d_navigation"),
                "config",
                "planning_apf.yaml",
            ]),
            description="YAML 配置文件，包含 planning_apf_node 参数",
        ),
        OpaqueFunction(function=_planning_apf_setup),
    ])
