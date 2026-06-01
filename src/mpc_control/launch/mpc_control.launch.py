#!/usr/bin/env python3
"""
MPC Drone Control Launch

为每台机器人启动 MPC 轨迹跟踪控制器，订阅 trajectory，发布 cmd_vel。
定位统一使用 Swarm-LIO2 + map_fusion 发布的 global_odom。

使用方法:
  ros2 launch mpc_control mpc_control.launch.py
  ros2 launch mpc_control mpc_control.launch.py robots:=bot1,bot2,bot3

前提：
  - planning_node 已运行，并发布 /{robot}/trajectory
  - map_fusion 节点已运行，发布 /{robot}/global_odom（依赖 Swarm-LIO2）
"""
from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _parse_robot_names(raw: str) -> List[str]:
    return [n.strip() for n in raw.split(",") if n.strip()] or ["bot1"]


def _mpc_control_setup(context, *args, **kwargs):
    robots_raw = LaunchConfiguration("robots").perform(context)
    vel_scale = LaunchConfiguration("velocity_scale", default="1.0").perform(context)
    min_speed = LaunchConfiguration("min_speed", default="0.0").perform(context)
    odom_suffix = LaunchConfiguration("odom_suffix", default="global_odom").perform(context)
    use_esdf = LaunchConfiguration("use_esdf", default="true").perform(context)
    simple_mode = LaunchConfiguration("simple_mode", default="false").perform(context)
    simple_kp_xy = LaunchConfiguration("simple_kp_xy", default="0.85").perform(context)
    simple_kp_z = LaunchConfiguration("simple_kp_z", default="0.75").perform(context)
    simple_max_speed_xy = LaunchConfiguration("simple_max_speed_xy", default="0.65").perform(context)
    simple_max_speed_z = LaunchConfiguration("simple_max_speed_z", default="0.25").perform(context)
    simple_max_accel = LaunchConfiguration("simple_max_accel", default="0.65").perform(context)
    simple_lookahead = LaunchConfiguration("simple_lookahead", default="3").perform(context)
    simple_goal_tolerance = LaunchConfiguration("simple_goal_tolerance", default="0.05").perform(context)
    simple_cmd_smoothing_alpha = LaunchConfiguration("simple_cmd_smoothing_alpha", default="0.55").perform(context)
    esdf_mode = LaunchConfiguration("esdf_mode", default="shm").perform(context)
    esdf_shm_name = LaunchConfiguration("esdf_shm_name", default="/fiesta_esdf").perform(context)
    esdf_grid_topic = LaunchConfiguration("esdf_grid_topic", default="/esdf/grid_full").perform(context)
    esdf_frame_id = LaunchConfiguration("esdf_frame_id", default="map_origin").perform(context)
    esdf_d_safe = LaunchConfiguration("esdf_d_safe", default="0.05").perform(context)
    robot_names = _parse_robot_names(robots_raw)

    params = {
        "velocity_scale": float(vel_scale),
        "min_speed": float(min_speed),
        "odom_suffix": odom_suffix,
        "simple_mode": simple_mode.lower() in ("true", "1", "yes"),
        "simple_kp_xy": float(simple_kp_xy),
        "simple_kp_z": float(simple_kp_z),
        "simple_max_speed_xy": float(simple_max_speed_xy),
        "simple_max_speed_z": float(simple_max_speed_z),
        "simple_max_accel": float(simple_max_accel),
        "simple_lookahead": int(simple_lookahead),
        "simple_goal_tolerance": float(simple_goal_tolerance),
        "simple_cmd_smoothing_alpha": float(simple_cmd_smoothing_alpha),
        "use_esdf": use_esdf.lower() in ("true", "1", "yes"),
        "esdf_mode": esdf_mode,
        "esdf_shm_name": esdf_shm_name,
        "esdf_grid_topic": esdf_grid_topic,
        "esdf_frame_id": esdf_frame_id,
        "esdf_d_safe": float(esdf_d_safe),
    }

    nodes = []
    for name in robot_names:
        prefix = f"/{name}"
        nodes.append(
            Node(
                package="mpc_control",
                executable="mpc_drone_control",
                name=f"mpc_drone_control_{name}",
                namespace=name,
                output="screen",
                parameters=[params],
                remappings=[
                    (odom_suffix, f"{prefix}/{odom_suffix}"),
                    ("trajectory", f"{prefix}/trajectory"),
                    ("cmd_vel", f"{prefix}/cmd_vel"),
                ],
            )
        )
        # Gazebo MulticopterVelocityControl 需收到 enable=true 才响应 cmd_vel
        from launch.actions import ExecuteProcess
        enable_cmd = ExecuteProcess(
            cmd=["ros2", "topic", "pub", "--once", f"{prefix}/enable", "std_msgs/msg/Bool", "{data: true}"],
            output="log",
        )
        nodes.append(TimerAction(period=2.0, actions=[enable_cmd]))

    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "robots",
            default_value="bot1,bot2,bot3,bot4",
            description="逗号分隔的机器人命名空间，如 bot1,bot2,bot3",
        ),
        DeclareLaunchArgument(
            "velocity_scale",
            default_value="0.8",
            description="在 MPC 线速度基础上再乘该系数（world→body 之后）",
        ),
        DeclareLaunchArgument(
            "min_speed",
            default_value="0.0",
            description="水平面最小合速度 (m/s)；垂向 vz 不参与 min_speed，避免被放大成下坠",
        ),
        DeclareLaunchArgument(
            "odom_suffix",
            default_value="global_odom",
            description="定位话题：默认 global_odom（Swarm-LIO2+map_fusion）；纯仿真可改为 gt/odom",
        ),
        DeclareLaunchArgument(
            "simple_mode",
            default_value="true",
            description="基础控制模式：P 速度跟踪 + 限速/限加速度（推荐仿真默认开启）",
        ),
        DeclareLaunchArgument(
            "simple_kp_xy",
            default_value="0.85",
            description="基础控制 XY 比例增益",
        ),
        DeclareLaunchArgument(
            "simple_kp_z",
            default_value="0.75",
            description="基础控制 Z 比例增益",
        ),
        DeclareLaunchArgument(
            "simple_max_speed_xy",
            default_value="0.65",
            description="基础控制 XY 最大速度 [m/s]",
        ),
        DeclareLaunchArgument(
            "simple_max_speed_z",
            default_value="0.25",
            description="基础控制 Z 最大速度 [m/s]",
        ),
        DeclareLaunchArgument(
            "simple_max_accel",
            default_value="0.65",
            description="基础控制线加速度限幅 [m/s^2]，调小可抑制轨迹振荡",
        ),
        DeclareLaunchArgument(
            "simple_lookahead",
            default_value="3",
            description="基础控制轨迹前视点偏移（按离最近点索引）",
        ),
        DeclareLaunchArgument(
            "simple_goal_tolerance",
            default_value="0.05",
            description="基础控制终点停止阈值 [m]",
        ),
        DeclareLaunchArgument(
            "simple_cmd_smoothing_alpha",
            default_value="0.55",
            description="速度指令一阶低通系数，越小越平滑但响应越慢",
        ),
        DeclareLaunchArgument(
            "use_esdf",
            default_value="false",
            description="是否启用 ESDF 避障不等式约束（零拷贝：shm 或 grid_cache）",
        ),
        DeclareLaunchArgument(
            "esdf_mode",
            default_value="shm",
            description="ESDF 模式：shm=共享内存零拷贝，grid_cache=订阅 /esdf/grid_full",
        ),
        DeclareLaunchArgument(
            "esdf_d_safe",
            default_value="0.3",
            description="ESDF 安全距离 [m]，机器人中心到障碍物表面的最小距离",
        ),
        OpaqueFunction(function=_mpc_control_setup),
    ])
