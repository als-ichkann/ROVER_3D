#!/usr/bin/env python3
"""
从 YAML（与 config/gmm_goal_publisher.yaml 同结构）读取 GMM，向 apf_goal 发布一条消息后退出。

用法:
  ros2 run rover3d_navigation publish_gmm_goal.py
  ros2 run rover3d_navigation publish_gmm_goal.py --config /path/to/gmm_goal_publisher.yaml

需已 source 工作空间；planning_apf_node 等订阅方应先运行或随后运行（TRANSIENT_LOCAL 可缓存最后一条）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, List

import rclpy
import yaml
from geometry_msgs.msg import Point
from navigation_msgs.msg import GMM
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

_GMM_GOAL_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


def _load_gmm_params(yaml_path: Path) -> dict[str, Any]:
    with yaml_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 根节点须为 mapping: {yaml_path}")

    if "gmm_goal_publisher" in data:
        inner = data["gmm_goal_publisher"]
        if isinstance(inner, dict) and "ros__parameters" in inner:
            return dict(inner["ros__parameters"])

    for v in data.values():
        if isinstance(v, dict) and "ros__parameters" in v:
            p = v["ros__parameters"]
            if isinstance(p, dict) and "means" in p:
                return dict(p)

    raise ValueError(
        f"未找到 GMM 参数（期望 gmm_goal_publisher.ros__parameters 或含 means 的 ros__parameters）: {yaml_path}"
    )


def _parse_means(raw: Any) -> List[List[float]]:
    means: List[List[float]] = []
    if not raw:
        return [[0.0, 0.0, 0.0]]
    if isinstance(raw[0], (int, float)):
        raw = [raw]
    for m in raw:
        if isinstance(m, (list, tuple)) and len(m) >= 3:
            means.append([float(m[0]), float(m[1]), float(m[2])])
        elif isinstance(m, dict) and "x" in m and "y" in m and "z" in m:
            means.append([float(m["x"]), float(m["y"]), float(m["z"])])
    return means if means else [[0.0, 0.0, 0.0]]


def _parse_covs_weights(
    covs_raw: list, weights_raw: list, n: int, cov_scale: float
) -> tuple[list, list]:
    weights: List[float] = []
    for w in weights_raw[:n]:
        weights.append(float(w))
    while len(weights) < n:
        weights.append(1.0)
    weights = weights[:n]
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]

    covs: List[list] = []
    if len(covs_raw) >= n * 9:
        for i in range(n):
            start = i * 9
            c = list(covs_raw[start : start + 9])
            if len(c) == 9:
                covs.append(c)
        if len(covs) == n:
            return covs, weights

    for _ in range(n):
        covs.append(
            [
                float(cov_scale),
                0.0,
                0.0,
                0.0,
                float(cov_scale),
                0.0,
                0.0,
                0.0,
                float(cov_scale),
            ]
        )
    return covs, weights


def _build_gmm_msg(stamp_fn, means: list, covs: list, weights: list) -> GMM:
    msg = GMM()
    msg.header.stamp = stamp_fn()
    msg.header.frame_id = "map_origin"
    for m in means:
        p = Point()
        p.x, p.y, p.z = float(m[0]), float(m[1]), float(m[2])
        msg.means.append(p)
    msg.covariances = [float(v) for c in covs for v in c]
    msg.weights = [float(w) for w in weights]
    return msg


def main() -> int:
    parser = argparse.ArgumentParser(description="从 YAML 发布一条 GMM 到 apf_goal")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="GMM YAML（默认: rover3d_navigation/config/gmm_goal_publisher.yaml）",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.4,
        help="发布后等待秒数，便于 DDS 发出（默认 0.4）",
    )
    args = parser.parse_args()

    cfg = args.config
    if cfg is None:
        try:
            from ament_index_python.packages import get_package_share_directory

            share = Path(get_package_share_directory("rover3d_navigation"))
            cfg = share / "config" / "gmm_goal_publisher.yaml"
        except Exception as e:
            print("请用 --config 指定 YAML，或先 colcon 安装 rover3d_navigation。", file=sys.stderr)
            print(e, file=sys.stderr)
            return 1

    if not cfg.is_file():
        print(f"配置文件不存在: {cfg}", file=sys.stderr)
        return 1

    try:
        params = _load_gmm_params(cfg)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    goal_topic = str(params.get("goal_topic", "apf_goal"))
    means_raw = params.get("means", [0.0, 0.0, 0.0])
    covs_raw = list(params.get("covariances", []) or [])
    weights_raw = list(params.get("weights", [1.0]) or [1.0])
    cov_scale = float(params.get("default_covariance_scale", 0.5))

    means = _parse_means(means_raw)
    covs, weights = _parse_covs_weights(covs_raw, weights_raw, len(means), cov_scale)

    rclpy.init()
    node = rclpy.create_node("publish_gmm_goal_cli")
    pub = node.create_publisher(GMM, goal_topic, _GMM_GOAL_QOS)
    msg = _build_gmm_msg(lambda: node.get_clock().now().to_msg(), means, covs, weights)

    # 等待匹配订阅（可选）
    deadline = time.monotonic() + 5.0
    while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    pub.publish(msg)
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.sleep:
        rclpy.spin_once(node, timeout_sec=0.05)

    n_sub = pub.get_subscription_count()
    node.get_logger().info(
        f"Published GMM ({len(means)} component(s)) to '{goal_topic}' "
        f"(matched subscribers: {n_sub})"
    )

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
