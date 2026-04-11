#!/usr/bin/env python3
"""
MPC Drone Control ROS2 Node

订阅 odom（默认 global_odom）和 trajectory (apf_trajectory)，发布 cmd_vel (Twist)。
默认使用“基础控制模式”（P 速度跟踪 + 限速/限加速度）以适配 Gazebo 无人机速度控制插件。
保留原 MPC 链路，可通过参数 simple_mode:=false 切换。
"""

import numpy as np
import threading
from typing import Dict
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.time import Time as RosTime
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class MPCDroneControlNode(Node):
    """单机器人 MPC 轨迹跟踪节点"""

    def __init__(self):
        super().__init__("mpc_drone_control")
        self.declare_parameter("control_dt", 0.1)
        self.declare_parameter("control_frequency", 10)
        self.declare_parameter("velocity_scale", 1.0)
        self.declare_parameter("min_speed", 0.0)
        self.declare_parameter("odom_suffix", "global_odom")
        self.declare_parameter("simple_mode", True)
        self.declare_parameter("simple_kp_xy", 1.0)
        self.declare_parameter("simple_kp_z", 1.2)
        self.declare_parameter("simple_max_speed_xy", 0.9)
        self.declare_parameter("simple_max_speed_z", 0.45)
        self.declare_parameter("simple_max_accel", 1.5)
        self.declare_parameter("simple_lookahead", 3)
        self.declare_parameter("simple_goal_tolerance", 0.25)
        self.declare_parameter("use_esdf", False)
        self.declare_parameter("esdf_shm_name", "/fiesta_esdf")
        self.declare_parameter("esdf_frame_id", "map_origin")
        self.declare_parameter("esdf_d_safe", 0.3)
        self.declare_parameter("map_origin_x", -5.0)
        self.declare_parameter("map_origin_y", -7.5)
        self.declare_parameter("map_origin_z", 0.0)
        self.declare_parameter("map_size_x", 22.0)
        self.declare_parameter("map_size_y", 17.0)
        self.declare_parameter("map_size_z", 6.0)
        self.declare_parameter("esdf_resolution", 0.15)
        dt = float(self.get_parameter("control_dt").value)
        self._dt = dt
        self._control_frequency = float(self.get_parameter("control_frequency").value)
        self._velocity_scale = float(self.get_parameter("velocity_scale").value)
        self._min_speed = float(self.get_parameter("min_speed").value)
        self._simple_mode = bool(self.get_parameter("simple_mode").value)
        self._simple_kp_xy = float(self.get_parameter("simple_kp_xy").value)
        self._simple_kp_z = float(self.get_parameter("simple_kp_z").value)
        self._simple_max_speed_xy = float(self.get_parameter("simple_max_speed_xy").value)
        self._simple_max_speed_z = float(self.get_parameter("simple_max_speed_z").value)
        self._simple_max_accel = float(self.get_parameter("simple_max_accel").value)
        self._simple_lookahead = int(self.get_parameter("simple_lookahead").value)
        self._simple_goal_tolerance = float(self.get_parameter("simple_goal_tolerance").value)

        # 共享状态
        self._lock = threading.Lock()
        self._state = np.zeros(12)  # [x,vx,ax, y,vy,ay, z,vz,az, yaw,roll,pitch]
        self._state_valid = False
        self._trajectory_points = None  # (N,3) ndarray
        self._new_trajectory_flag = False
        self._mpc = None
        self._mpc_initialized = False
        self._current_step = 0
        self._lastz = None
        self._lastslacknum = 0
        self._actual_state = np.zeros((1, 12))
        self._num_robots = 1
        self._esdf_adapter = None
        self._last_odom_pos = None
        self._last_odom_time_ns = None
        self._last_cmd_world = np.zeros(3, dtype=float)
        self._state_frame_id = None  # 由 Odometry.header.frame_id 决定
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # Humble 的 RcutilsLogger 无 info_throttle/warn_throttle，用手动节流键控
        self._warn_last_ns: Dict[str, int] = {}
        if self.get_parameter("use_esdf").value:
            try:
                from rover3d_navigation.esdf_adapter import EsdfShmAdapter
                kw = dict(
                    frame_id=str(self.get_parameter("esdf_frame_id").value),
                    map_origin_x=float(self.get_parameter("map_origin_x").value),
                    map_origin_y=float(self.get_parameter("map_origin_y").value),
                    map_origin_z=float(self.get_parameter("map_origin_z").value),
                    map_size_x=float(self.get_parameter("map_size_x").value),
                    map_size_y=float(self.get_parameter("map_size_y").value),
                    map_size_z=float(self.get_parameter("map_size_z").value),
                    resolution=float(self.get_parameter("esdf_resolution").value),
                )
                self._esdf_adapter = EsdfShmAdapter(
                    self, shm_name=str(self.get_parameter("esdf_shm_name").value), **kw
                )
                self.get_logger().info(
                    "ESDF constraint enabled (SHM + trilinear): d_safe=%.2f"
                    % self.get_parameter("esdf_d_safe").value
                )
            except Exception as e:
                self.get_logger().warn("ESDF constraint disabled: %s" % e)

        # 订阅 / 发布（定位可用 gt/odom 或 global_odom）
        self._odom_suffix = str(self.get_parameter("odom_suffix").value)
        self._odom_sub = self.create_subscription(
            Odometry, self._odom_suffix, self._odom_cb, 10
        )
        self._traj_sub = self.create_subscription(
            Path, "trajectory", self._trajectory_cb, 10
        )
        self._cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)

        # 控制定时器 (10 Hz)
        self._timer = self.create_timer(1.0 / self._control_frequency, self._control_timer_cb)

        if self._simple_mode:
            self.get_logger().info("mpc_drone_control node started in simple_mode")
        else:
            self.get_logger().info("mpc_drone_control node started in mpc_mode")

    def _throttle_allow(self, key: str, period_sec: float) -> bool:
        """Humble 无 Logger.*_throttle：用节点时钟做同类节流。返回 True 表示本次应输出日志。"""
        now_ns = self.get_clock().now().nanoseconds
        period_ns = int(max(period_sec, 0.0) * 1e9)
        prev = self._warn_last_ns.get(key)
        if prev is None or (now_ns - prev) >= period_ns:
            self._warn_last_ns[key] = now_ns
            return True
        return False

    def _quat_to_yaw(self, q):
        """从四元数 (x,y,z,w) 提取 yaw（绕 z 轴）"""
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _world_to_body_velocity(self, v_world, yaw):
        """将世界坐标系速度转为机体坐标系（MulticopterVelocityControl 期望机体系）"""
        cy = np.cos(-yaw)
        sy = np.sin(-yaw)
        v_body_x = v_world[0] * cy - v_world[1] * sy
        v_body_y = v_world[0] * sy + v_world[1] * cy
        v_body_z = v_world[2]  # Z 轴不受 yaw 影响
        return np.array([v_body_x, v_body_y, v_body_z])

    def _apply_min_speed_horizontal(self, v_world: np.ndarray) -> np.ndarray:
        """
        只对水平速度施加最小模长，避免「全矢量 min_speed」把微小的负 vz 放大成强下坠指令。
        """
        v = np.asarray(v_world, dtype=float).reshape(3).copy()
        xy = v[:2]
        n_xy = float(np.linalg.norm(xy))
        if n_xy > 1e-9 and n_xy < self._min_speed:
            v[:2] = xy / n_xy * self._min_speed
        return v

    def _odom_cb(self, msg):
        """从 Odometry 更新 state。TF 来源的 Odom 不含线速度，故只更新位置和 yaw，不读假速度。"""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = self._quat_to_yaw(q)
        pos = np.array([p.x, p.y, p.z], dtype=float)
        now_ns = self.get_clock().now().nanoseconds
        vel = None
        if self._last_odom_pos is not None and self._last_odom_time_ns is not None:
            dt = (now_ns - self._last_odom_time_ns) * 1e-9
            if dt > 1e-4:
                vel = (pos - self._last_odom_pos) / dt
        self._last_odom_pos = pos
        self._last_odom_time_ns = now_ns
        if self._state_frame_id is None and getattr(msg.header, "frame_id", ""):
            self._state_frame_id = msg.header.frame_id
        with self._lock:
            self._state[0] = p.x
            self._state[3] = p.y
            self._state[6] = p.z
            if vel is not None and np.isfinite(vel).all():
                alpha = 0.4
                self._state[1] = (1.0 - alpha) * self._state[1] + alpha * float(vel[0])
                self._state[4] = (1.0 - alpha) * self._state[4] + alpha * float(vel[1])
                self._state[7] = (1.0 - alpha) * self._state[7] + alpha * float(vel[2])
            self._state[9] = yaw
            self._state_valid = True

    def _simple_control_step(self, state: np.ndarray, trajectory_points: np.ndarray) -> Twist:
        """基础轨迹跟踪：P 速度控制 + 速度/加速度限幅，适配 Gazebo MulticopterVelocityControl。"""
        cur = np.array([state[0], state[3], state[6]], dtype=float)
        pts = np.asarray(trajectory_points, dtype=float).reshape(-1, 3)
        if pts.shape[0] == 0:
            self._last_cmd_world[:] = 0.0
            return Twist()

        dist_all = np.linalg.norm(pts - cur.reshape(1, 3), axis=1)
        nearest = int(np.argmin(dist_all))
        target_idx = min(nearest + max(self._simple_lookahead, 0), pts.shape[0] - 1)
        target = pts[target_idx]
        goal = pts[-1]
        err = target - cur
        goal_dist = float(np.linalg.norm(goal - cur))

        if goal_dist < self._simple_goal_tolerance:
            self._last_cmd_world[:] = 0.0
            return Twist()

        v_world = np.array(
            [
                self._simple_kp_xy * err[0],
                self._simple_kp_xy * err[1],
                self._simple_kp_z * err[2],
            ],
            dtype=float,
        )

        v_xy = v_world[:2]
        n_xy = float(np.linalg.norm(v_xy))
        if n_xy > self._simple_max_speed_xy > 1e-6:
            v_world[:2] = v_xy / n_xy * self._simple_max_speed_xy
        v_world[2] = float(np.clip(v_world[2], -self._simple_max_speed_z, self._simple_max_speed_z))
        if self._min_speed > 1e-6 and goal_dist > self._simple_goal_tolerance * 2.0:
            v_world = self._apply_min_speed_horizontal(v_world)

        dv = v_world - self._last_cmd_world
        max_dv = max(self._simple_max_accel, 1e-6) * self._dt
        dv_norm = float(np.linalg.norm(dv))
        if dv_norm > max_dv:
            dv = dv / dv_norm * max_dv
            v_world = self._last_cmd_world + dv
        self._last_cmd_world = v_world

        v_body = self._world_to_body_velocity(v_world, state[9])
        v_body[0] *= self._velocity_scale
        v_body[1] *= self._velocity_scale
        twist = Twist()
        twist.linear.x = float(v_body[0])
        twist.linear.y = float(v_body[1])
        twist.linear.z = float(v_body[2])
        return twist

    def _trajectory_cb(self, msg):
        """收到新 Path 时更新参考轨迹，并标记可中断当前执行。"""
        n = len(msg.poses)
        if n == 0:
            # 规划侧可能发空 Path（优化间隙/停机）；不刷屏
            return

        # 轨迹帧对齐：把 Path 里的点从各自 frame_id 变换到 MPC 当前使用的 state 坐标系。
        # 若尚未收到 Odometry.header.frame_id，则退回直接使用 msg 里的数值。
        target_frame = self._state_frame_id
        if not target_frame:
            pts = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses], dtype=float)
        else:
            # tf2 对于 stamp=0 通常可解释为“取最新”，失败则直接使用原点（并报 warning）。
            stamp = getattr(msg.header, "stamp", None)
            tf_time = RosTime()  # 默认 latest
            try:
                if stamp is not None:
                    tf_time = RosTime.from_msg(stamp)
            except Exception:
                tf_time = RosTime()

            pts_list = []
            for ps in msg.poses:
                src_frame = getattr(ps.header, "frame_id", "")
                if not src_frame or src_frame == target_frame:
                    pts_list.append([ps.pose.position.x, ps.pose.position.y, ps.pose.position.z])
                    continue
                try:
                    transform = self._tf_buffer.lookup_transform(
                        target_frame, src_frame, tf_time, timeout_sec=0.1
                    )
                    ps_t = do_transform_pose(ps, transform)
                    pts_list.append([ps_t.pose.position.x, ps_t.pose.position.y, ps_t.pose.position.z])
                except Exception:
                    # 变换失败：直接用原值，同时避免频繁刷屏
                    pts_list.append([ps.pose.position.x, ps.pose.position.y, ps.pose.position.z])
                    self.get_logger().warn(
                        f"TF transform Path point failed: {src_frame} -> {target_frame}. Using raw points.",
                    )
            pts = np.asarray(pts_list, dtype=float)

        # 规划在「单步 APF / 单点保持」时可能只发 1 个 pose；get_3d_trajectories + MPC 至少需要 2 点
        if pts.shape[0] < 2:
            pts = np.vstack([pts, pts[-1:]])
            if self._throttle_allow("traj_single_pose_dup", 10.0):
                self.get_logger().info(
                    "Reference Path had only one pose; duplicated end point for MPC."
                )

        with self._lock:
            self._trajectory_points = pts
            self._new_trajectory_flag = True

    def _build_mpc_from_trajectory(self, current_pos, trajectory_raw):
        """根据当前位置和原始轨迹，构建/更新 MPC
        """
        from . import robot_3D

        clipped = robot_3D.get_3d_trajectories(current_pos, [trajectory_raw])
        if len(clipped[0]) == 0:
            self.get_logger().warn("Trajectory clip result empty")
            return False

        traj_clipped = clipped[0]
        pos = np.array(current_pos).reshape(-1, 3)

        rp = np.vstack((pos, traj_clipped))
        if len(rp) < 2:
            rp = np.vstack((rp, rp[-1:]))

        start_point = [list(current_pos)]
        trajectories = [rp.tolist()]

        if self._mpc is None:
            dummy = [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]]]
            esdf_d_safe = float(self.get_parameter("esdf_d_safe").value)
            from .esdf_constraint import make_mpc_esdf_adapter
            mpc_adapter = make_mpc_esdf_adapter(self._esdf_adapter)
            self._mpc = robot_3D.MPC_3D(
                num_robots=1, N=6, dt=self._dt, discrete_points=dummy,
                esdf_adapter=mpc_adapter,
                esdf_d_safe=esdf_d_safe,
            )

        self._mpc.control(start_point, trajectories)
        self._current_step = 0
        self._actual_state[0] = np.copy(self._state)
        self._mpc.actualState = np.zeros((1, 12))
        self._mpc.actualState[0] = np.copy(self._state)
        # 同步 agent 初始状态
        agent = self._mpc.agents[0]
        agent.position = np.array(current_pos).reshape(-1, 3)
        agent.velocity = [self._state[1], self._state[4], self._state[7]]
        agent.acceleration = [self._state[2], self._state[5], self._state[8]]
        agent.theta = [
            self._state[9], self._state[10], self._state[11]
        ]
        self._mpc_initialized = True
        return True

    def _control_timer_cb(self):
        """控制定时器：执行单步 MPC，发布 cmd_vel"""
        with self._lock:
            if not self._state_valid:
                return
            state = np.copy(self._state)
            trajectory_points = self._trajectory_points
            new_flag = self._new_trajectory_flag
            self._new_trajectory_flag = False

        if trajectory_points is None or len(trajectory_points) < 2:
            # 无轨迹：发布零速
            self._last_cmd_world[:] = 0.0
            twist = Twist()
            self._cmd_pub.publish(twist)
            return

        current_pos = [state[0], state[3], state[6]]
        if self._simple_mode:
            twist = self._simple_control_step(state, trajectory_points)
            self._cmd_pub.publish(twist)
            return

        with self._lock:
            if new_flag or not self._mpc_initialized:
                ok = self._build_mpc_from_trajectory(current_pos, trajectory_points)
                if not ok:
                    twist = Twist()
                    self._cmd_pub.publish(twist)
                    return
                # 新轨迹已应用，直接执行第一步
            mpc = self._mpc
            if mpc is None or not self._mpc_initialized:
                twist = Twist()
                self._cmd_pub.publish(twist)
                return

        agent = mpc.agents[0]
        NT = mpc.NT

        if self._current_step >= NT:
            # 轨迹已完成（MPC 输出已是物理速度，直接信任）
            v = np.array(getattr(agent, "velocity", [0, 0, 0]), dtype=float)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            v = self._apply_min_speed_horizontal(v)
            ov = np.nan_to_num(np.array(getattr(agent, "angular_velocity", [0, 0, 0])), nan=0.0, posinf=0.0, neginf=0.0)
            # 世界系 -> 机体系（MulticopterVelocityControl 期望机体系 cmd_vel）
            v_body = self._world_to_body_velocity(v, state[9])
            v_body[0] *= self._velocity_scale
            v_body[1] *= self._velocity_scale
            # 垂向不乘 velocity_scale，减轻「水平加速 + 垂向误差」被同步放大导致的下坠
            twist = Twist()
            twist.linear.x = float(v_body[0])
            twist.linear.y = float(v_body[1])
            twist.linear.z = float(v_body[2])
            twist.angular.x = float(ov[0])
            twist.angular.y = float(ov[1])
            twist.angular.z = float(ov[2])
            self._cmd_pub.publish(twist)
            return

        # 更新其他 agent 位置（单机时仅自身）
        state_list = {"state_0": state}
        for i in range(mpc.num_robots):
            mpc.agents[i].position = np.array([state_list[f"state_{i}"][0], state_list[f"state_{i}"][3], state_list[f"state_{i}"][6]]).reshape(-1, 3)

        actual_state = np.zeros((mpc.num_robots, 12))
        actual_state[0] = state
        k = self._current_step

        try:
            from . import robot_3D

            _, _, _, agent, self._lastz, self._lastslacknum = robot_3D.agent_thread_3D(
                agent, mpc, 0, actual_state, k, self._lastz, self._lastslacknum
            )
        except Exception as e:
            self.get_logger().error(f"agent_thread_3D failed: {e}")
            twist = Twist()
            self._cmd_pub.publish(twist)
            return

        # 更新本地状态：actual_state[0] 已含 MPC 预测的速度（controller 写回），不再被 Odom 假速度覆盖
        with self._lock:
            self._state = np.copy(actual_state[0])
            self._current_step += 1

        # 发布 Twist：MPC 受 v_max 限制；min_speed 仅水平；velocity_scale 仅 x/y 机体系分量
        v = np.array(agent.velocity, dtype=float)
        if not np.isfinite(v).all():
            if self._throttle_allow("mpc_vel_nan", 1.0):
                self.get_logger().warn("MPC velocity contains NaN/inf, clamping to 0")
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = self._apply_min_speed_horizontal(v)
        ov = np.array(getattr(agent, "angular_velocity", [0.0, 0.0, 0.0]), dtype=float)
        if not np.isfinite(ov).all():
            if self._throttle_allow("mpc_ang_nan", 1.0):
                self.get_logger().warn("MPC angular_velocity contains NaN/inf, clamping to 0")
        ov = np.nan_to_num(ov, nan=0.0, posinf=0.0, neginf=0.0)
        yaw = state[9]
        v_body = self._world_to_body_velocity(v, yaw)
        v_body[0] *= self._velocity_scale
        v_body[1] *= self._velocity_scale
        twist = Twist()
        twist.linear.x = float(v_body[0])
        twist.linear.y = float(v_body[1])
        twist.linear.z = float(v_body[2])
        twist.angular.x = float(ov[0])
        twist.angular.y = float(ov[1])
        twist.angular.z = float(ov[2])
        self._cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = MPCDroneControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"mpc_drone_control exited: {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
