#!/usr/bin/env python3
"""
规划节点：采用 PlanningProcess 逻辑，执行宏观规划并调用当前微观轨迹生成器发布轨迹。
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from navigation_msgs.msg import GMM
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rover3d_navigation.ROVER_3D import PlanningProcess
from rover3d_navigation.esdf_adapter import EsdfShmAdapter


class PlanningNode(Node):
    """规划节点：订阅 GMM 目标与多机 odom，发布每机器人轨迹 Path。"""

    def __init__(self) -> None:
        super().__init__("planning_node")

        self.declare_parameter("robot_names", ["bot1", "bot2"])
        self.declare_parameter("odom_suffix", "global_odom")
        self.declare_parameter("goal_topic", "goal_gmm")
        self.declare_parameter("trajectory_suffix", "trajectory")
        self.declare_parameter("control_rate", 5.0)
        self.declare_parameter("max_micro_try", 10)
        self.declare_parameter("gmm_interp_steps", 5)
        self.declare_parameter("micro_goal_lock_radius", 0.25)
        self.declare_parameter("slp_epsilon", 0.2)
        self.declare_parameter("use_gmm_trajectory_slp", True)
        self.declare_parameter("micro_controller", "apf")
        self.declare_parameter("esdf_mode", "shm")
        self.declare_parameter("esdf_shm_name", "/fiesta_esdf")
        self.declare_parameter("signed_sdf_enable", True)
        self.declare_parameter("signed_sdf_occupied_eps", 1e-6)
        self.declare_parameter("signed_sdf_inside_offset_vox", 0.5)
        self.declare_parameter("esdf_frame_id", "map_origin")
        self.declare_parameter("map_origin_x", -5.0)
        self.declare_parameter("map_origin_y", -7.5)
        self.declare_parameter("map_origin_z", 0.0)
        self.declare_parameter("map_size_x", 22.0)
        self.declare_parameter("map_size_y", 17.0)
        self.declare_parameter("map_size_z", 6.0)
        self.declare_parameter("esdf_resolution", 0.15)
        self.declare_parameter("config_dir", "")

        # 兼容旧参数命名
        self.declare_parameter("max_apf_try", 10)
        self.declare_parameter("apf_goal_lock_radius", 0.25)

        robot_names = self.get_parameter("robot_names").value
        if isinstance(robot_names, str):
            robot_names = [n.strip() for n in robot_names.split(",") if n.strip()]
        self._robot_names = list(robot_names)
        self._odom_suffix = str(self.get_parameter("odom_suffix").value)
        self._goal_topic = str(self.get_parameter("goal_topic").value)
        self._traj_suffix = str(self.get_parameter("trajectory_suffix").value)
        self._control_rate = float(self.get_parameter("control_rate").value)

        max_micro_try = int(self.get_parameter("max_micro_try").value)
        max_apf_try = int(self.get_parameter("max_apf_try").value)
        self._max_micro_try = max_micro_try if max_micro_try > 0 else max_apf_try
        self._gmm_interp_steps = int(self.get_parameter("gmm_interp_steps").value)

        micro_goal_lock_radius = float(self.get_parameter("micro_goal_lock_radius").value)
        apf_goal_lock_radius = float(self.get_parameter("apf_goal_lock_radius").value)
        self._micro_goal_lock_radius = (
            micro_goal_lock_radius if micro_goal_lock_radius > 0 else apf_goal_lock_radius
        )
        self._slp_epsilon = float(self.get_parameter("slp_epsilon").value)
        self._use_gmm_trajectory_slp = bool(self.get_parameter("use_gmm_trajectory_slp").value)
        self._micro_controller = str(self.get_parameter("micro_controller").value).strip().lower()
        if self._micro_controller not in ("apf", "density"):
            self.get_logger().warn(
                f"Invalid micro_controller={self._micro_controller}, fallback to 'apf'"
            )
            self._micro_controller = "apf"
        config_dir_param = str(self.get_parameter("config_dir").value)
        if config_dir_param:
            self._config_dir = config_dir_param
        else:
            try:
                from ament_index_python.packages import get_package_share_directory

                pkg_share = get_package_share_directory("rover3d_navigation")
                self._config_dir = os.path.join(pkg_share, "config")
            except Exception:
                self._config_dir = ""

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_gmm_goal = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self._odom_cache: Dict[str, Optional[Odometry]] = {name: None for name in self._robot_names}
        self._gmm_goal: Optional[tuple] = None
        self._planning_process: Optional[PlanningProcess] = None
        self._log_no_odom_done = False
        self._log_no_traj_done = False
        self._log_no_config_dir_done = False
        self._last_traj_log_time = 0.0
        self._first_publish_done = False

        for name in self._robot_names:
            self.create_subscription(
                Odometry,
                f"{name}/{self._odom_suffix}",
                lambda msg, n=name: self._cb_odom(n, msg),
                qos,
            )
        self.create_subscription(GMM, self._goal_topic, self._cb_gmm, qos_gmm_goal)

        self._path_pubs: Dict[str, object] = {}
        for name in self._robot_names:
            self._path_pubs[name] = self.create_publisher(Path, f"{name}/{self._traj_suffix}", 10)

        esdf_mode = str(self.get_parameter("esdf_mode").value).lower()
        if esdf_mode != "shm":
            self.get_logger().warn(f"esdf_mode={esdf_mode} 已弃用，当前仅支持 shm，自动回退到 shm")
        self._esdf = EsdfShmAdapter(
            self,
            shm_name=str(self.get_parameter("esdf_shm_name").value),
            frame_id=self.get_parameter("esdf_frame_id").value,
            map_origin_x=float(self.get_parameter("map_origin_x").value),
            map_origin_y=float(self.get_parameter("map_origin_y").value),
            map_origin_z=float(self.get_parameter("map_origin_z").value),
            map_size_x=float(self.get_parameter("map_size_x").value),
            map_size_y=float(self.get_parameter("map_size_y").value),
            map_size_z=float(self.get_parameter("map_size_z").value),
            resolution=float(self.get_parameter("esdf_resolution").value),
            signed_sdf_enable=bool(self.get_parameter("signed_sdf_enable").value),
            signed_sdf_occupied_eps=float(self.get_parameter("signed_sdf_occupied_eps").value),
            signed_sdf_inside_offset_vox=float(self.get_parameter("signed_sdf_inside_offset_vox").value),
        )

        self.create_timer(1.0 / self._control_rate, self._control_loop)
        self.get_logger().info(
            f"Planning node: robots={self._robot_names}, goal={self._goal_topic}, "
            f"gmm_interp_steps={self._gmm_interp_steps}"
        )

    def _cb_odom(self, name: str, msg: Odometry) -> None:
        self._odom_cache[name] = msg

    def _cb_gmm(self, msg: GMM) -> None:
        n = len(msg.means)
        if n == 0:
            self._gmm_goal = None
            self._planning_process = None
            return
        if len(msg.covariances) < n * 9 or len(msg.weights) < n:
            self.get_logger().warn("GMM msg invalid: covariances or weights length mismatch")
            return
        means = [(m.x, m.y, m.z) for m in msg.means]
        covs = []
        for i in range(n):
            start = i * 9
            end = start + 9
            c = np.array(msg.covariances[start:end]).reshape(3, 3)
            covs.append(c)
        weights = list(msg.weights[:n])
        self._gmm_goal = (means, covs, weights)
        self._planning_process = None
        self.get_logger().info(f"New GMM goal: {n} components")
        self.get_logger().info(f"Raw goal means={means}, weights={weights}")

    def _get_robots_positions(self) -> Optional[np.ndarray]:
        positions = []
        missing = []
        for name in self._robot_names:
            odom = self._odom_cache.get(name)
            if odom is None:
                missing.append(name)
            else:
                p = odom.pose.pose.position
                positions.append([p.x, p.y, p.z])
        if missing:
            if not self._log_no_odom_done:
                self.get_logger().warn(
                    f"Waiting for odom: missing {missing}. "
                    f"Run sim and ensure /<robot>/{self._odom_suffix} is published for each robot."
                )
                self._log_no_odom_done = True
            return None
        self._log_no_odom_done = False
        return np.array(positions, dtype=float)

    def _control_loop(self) -> None:
        if self._gmm_goal is None:
            self._publish_empty_paths()
            return
        if hasattr(self._esdf, "refresh"):
            self._esdf.refresh()
        if hasattr(self._esdf, "is_ready") and not self._esdf.is_ready:
            return
        robots_positions = self._get_robots_positions()
        if robots_positions is None or len(robots_positions) == 0:
            return
        means, covs, weights = self._gmm_goal
        if len(means) == 0:
            self._publish_empty_paths()
            return

        if self._planning_process is None:
            if not self._config_dir:
                if not self._log_no_config_dir_done:
                    self.get_logger().error(
                        "无法创建 PlanningProcess：缺少 config_dir（且无法解析包内 config）。"
                        "请设置参数 config_dir 为含 GC_means_3D.json 等预计算文件的目录，"
                        "或运行 scripts/precompute_config_prior.py 生成后指向该目录。"
                    )
                    self._log_no_config_dir_done = True
                return
            self.get_logger().info("Creating PlanningProcess")
            xa = float(self.get_parameter("map_origin_x").value)
            ya = float(self.get_parameter("map_origin_y").value)
            za = float(self.get_parameter("map_origin_z").value)
            xb = xa + float(self.get_parameter("map_size_x").value)
            yb = ya + float(self.get_parameter("map_size_y").value)
            zb = za + float(self.get_parameter("map_size_z").value)
            self._planning_process = PlanningProcess(
                num_robots=len(self._robot_names),
                esdf_map=self._esdf,
                xa=xa,
                xb=xb,
                ya=ya,
                yb=yb,
                za=za,
                zb=zb,
                goal_means=means,
                goal_covs=covs,
                goal_weights=weights,
                gmm_interp_steps=self._gmm_interp_steps,
                max_micro_try=self._max_micro_try,
                micro_goal_lock_radius=self._micro_goal_lock_radius,
                slp_epsilon=self._slp_epsilon,
                use_gmm_trajectory_slp=self._use_gmm_trajectory_slp,
                micro_controller=self._micro_controller,
                config_dir=self._config_dir,
            )

        try:
            result = self._planning_process.run_one_cycle(robots_positions)
        except Exception as e:
            import traceback

            self.get_logger().error(f"PlanningProcess.run_one_cycle failed: {e}")
            self.get_logger().error(traceback.format_exc())
            return
        if result is None:
            if not self._log_no_traj_done:
                self.get_logger().info(
                    "run_one_cycle returned None (GMM optimizing or done). "
                    "Trajectories will publish when ready."
                )
                self._log_no_traj_done = True
            if self._planning_process.stop_flag:
                self._publish_empty_paths()
            return
        self._log_no_traj_done = False
        trajectories, _ = result
        path_frame = "map_origin"
        stamp = self.get_clock().now().to_msg()
        for i, name in enumerate(self._robot_names):
            path = Path()
            path.header.stamp = stamp
            path.header.frame_id = path_frame
            if i < len(trajectories):
                for pt in trajectories[i]:
                    ps = PoseStamped()
                    ps.header = path.header
                    ps.pose.position.x = float(pt[0])
                    ps.pose.position.y = float(pt[1])
                    ps.pose.position.z = float(pt[2])
                    ps.pose.orientation.w = 1.0
                    path.poses.append(ps)
            if len(path.poses) == 0:
                pos = robots_positions[i]
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = float(pos[0])
                ps.pose.position.y = float(pos[1])
                ps.pose.position.z = float(pos[2])
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self._path_pubs[name].publish(path)
        num_pts = sum(len(trajectories[i]) for i in range(min(len(trajectories), len(self._robot_names))))
        if not self._first_publish_done:
            self.get_logger().info(
                f"Published first trajectories: {num_pts} pose(s), frame={path_frame}, "
                f"topics={[f'{r}/{self._traj_suffix}' for r in self._robot_names]}"
            )
            self._first_publish_done = True
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_traj_log_time > 2.0:
            self.get_logger().info(
                f"Published trajectories: {num_pts} pose(s) for {len(self._robot_names)} robots"
            )
            self._last_traj_log_time = now

    def _publish_empty_paths(self) -> None:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map_origin"
        for pub in self._path_pubs.values():
            pub.publish(path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
