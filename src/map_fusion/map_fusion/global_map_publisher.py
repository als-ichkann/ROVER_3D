#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.exceptions import ParameterAlreadyDeclaredException

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2

from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from map_fusion.utils import BotFrameResolver, quat_to_rot_matrix, pcd_to_xyz_fast, pcd_to_xyzi_fast, voxelize_numpy


@dataclass
class BotSub:
    bot_id: int
    cloud_topic: str
    base_frame: str
    sub: any


class MapFusionNode(Node):
    def __init__(self) -> None:
        super().__init__('global_map_node')

        
        try:
            self.declare_parameter('use_sim_time', True)
        except ParameterAlreadyDeclaredException:
            pass
        self.declare_parameter('bot_prefix', 'bot')
        self.declare_parameter('bot_cloud_topic', '/cloud_registered')
        self.declare_parameter('bot_cloud_frame', 'world')
        self.declare_parameter('publish_rate_hz', 0.5)
        self.declare_parameter('combined_map_topic', '/global_downsampled_map')
        self.declare_parameter('origin_frame', 'map_origin')
        self.declare_parameter('voxel_leaf_size', 0.1)
        self.declare_parameter('filter_high_intensity', True)
        self.declare_parameter('intensity_max_threshold', 1000.0)
        self.declare_parameter('debug_profile', True)
        self.declare_parameter('debug_profile_every', 10)

        self.bot_prefix: str = str(self.get_parameter('bot_prefix').value)
        self.bot_cloud_topic: str = str(self.get_parameter('bot_cloud_topic').value).lstrip('/')
        self.bot_cloud_frame: str = str(self.get_parameter('bot_cloud_frame').value).lstrip('/')
        self.origin_frame_id: str = str(self.get_parameter('origin_frame').value).lstrip('/')
        self.combined_map_topic: str = str(self.get_parameter('combined_map_topic').value)
        rate_hz: float = float(self.get_parameter('publish_rate_hz').value)
        self.voxel_size: float = float(self.get_parameter('voxel_leaf_size').value)
        self.filter_high_intensity: bool = bool(self.get_parameter('filter_high_intensity').value)
        self.intensity_max_threshold: float = float(self.get_parameter('intensity_max_threshold').value)
        self.debug_profile: bool = bool(self.get_parameter('debug_profile').value)
        self.debug_profile_every: int = int(self.get_parameter('debug_profile_every').value)
        self._profile_count: int = 0

        self.bot_resolver = BotFrameResolver(self.bot_prefix)
        self.bots_dict: Dict[int, BotSub] = {}

        qos_map = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        qos_default = QoSProfile(depth=10)
        qos_tf = QoSProfile(
            depth=100,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_tf_static = QoSProfile(
            depth=100,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.local_maps: Dict[int, np.ndarray] = {}
        self._map_qos = qos_map

        # TF buffer/listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(TFMessage, '/tf', self._on_tf_msg, qos_tf)
        self.create_subscription(TFMessage, '/tf_static', self._on_tf_msg, qos_tf_static)

        self.global_pub = self.create_publisher(PointCloud2, self.combined_map_topic, qos_default)
        self.create_timer(1.0 / max(rate_hz, 0.1), self._publish_global_map)
        if self.debug_profile:
            self.create_timer(2.0, self._heartbeat)

        self._pc_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.get_logger().info(
            f"global_map_node TF-fusion started. global_frame={self.origin_frame_id}, "
            f"combined_map_topic={self.combined_map_topic}, tf children=/{self.bot_prefix}<id>/{self.bot_cloud_frame}, "
            f"cloud_topic=/{self.bot_prefix}<id>/{self.bot_cloud_topic}, voxel_size={self.voxel_size}, "
            f"filter_high_intensity={self.filter_high_intensity}, intensity_max_threshold={self.intensity_max_threshold}"
        )

    def _register_bot(self, bot_id: int | None) -> None:
        if bot_id is None or bot_id in self.bots_dict:
            return

        topic = f"/{self.bot_prefix}{bot_id}/{self.bot_cloud_topic}"
        base_frame = f"{self.bot_prefix}{bot_id}/{self.bot_cloud_frame}".lstrip('/')
        callback = lambda msg, uid=bot_id: self._map_callback(uid, msg)
        sub = self.create_subscription(PointCloud2, topic, callback, self._map_qos)
        self.bots_dict[bot_id] = BotSub(bot_id=bot_id, cloud_topic=topic, base_frame=base_frame, sub=sub)
        self.get_logger().info(f"Discovered {self.bot_prefix}{bot_id}: subscribing to '{topic}' (frame '{base_frame}')")

    def _on_tf_msg(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            self._register_bot(self.bot_resolver.bot_id_from_frame(t.header.frame_id))
            self._register_bot(self.bot_resolver.bot_id_from_frame(t.child_frame_id))

    def _map_callback(self, uid: int, msg: PointCloud2) -> None:
        t0 = time.perf_counter()
        xyz, intensity = pcd_to_xyzi_fast(msg)
        if xyz.size == 0:
            return
        if self.filter_high_intensity and intensity is not None:
            mask = intensity.ravel() <= self.intensity_max_threshold
            xyz = xyz[mask]
        if xyz.size == 0:
            return
        pts = voxelize_numpy(xyz, self.voxel_size)
        if pts.size == 0:
            return

        existing = self.local_maps.get(uid)
        if existing is None:
            existing = np.empty((0, pts.shape[1]), dtype=pts.dtype)

        combined = pts if existing.size == 0 else np.concatenate((existing, pts), axis=0)
        self.local_maps[uid] = voxelize_numpy(combined, self.voxel_size)

        if self.debug_profile:
            self._profile_count += 1
            if self.debug_profile_every > 0 and (self._profile_count % self.debug_profile_every == 0):
                dt_ms = (time.perf_counter() - t0) * 1000.0
                self.get_logger().info(
                    f"[profile] map_cb bot={uid} pts_in={pts.shape[0]} stored={self.local_maps[uid].shape[0]} dt={dt_ms:.2f}ms"
                )

    def _lookup_extrinsic(self, uid: int) -> Tuple[np.ndarray, np.ndarray] | None:
        if uid not in self.bots_dict:
            return None
        child = self.bots_dict.get(uid).base_frame
        try:
            # latest available transform
            tf = self.tf_buffer.lookup_transform(
                self.origin_frame_id,  # target (parent)
                child,                 # source (child)
                rclpy.time.Time()      # latest
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        R = quat_to_rot_matrix(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z], dtype=float)
        return R, trans

    def build_global_map(self) -> np.ndarray | None:
        if not self.bots_dict:
            return None

        merged_pts: np.ndarray | None = None

        for bot in self.bots_dict.values():
            uid = bot.bot_id
            pts = self.local_maps.get(uid)
            if pts is None or pts.size == 0:
                continue

            extr = self._lookup_extrinsic(uid)
            if extr is None:
                continue
            R, t = extr

            base_int = float(100 * (uid - 1))
            transformed = pts @ R.T + t
            transformed = transformed.astype(np.float32, copy=False)
            intensities = np.full((transformed.shape[0], 1), base_int, dtype=np.float32)
            combined = np.hstack((transformed, intensities))
            if merged_pts is None:
                merged_pts = combined
            else:
                merged_pts = np.concatenate((merged_pts, combined), axis=0)

            # Collapse accumulated data back into single ndarray to bound growth
            self.local_maps[uid] = pts

        return merged_pts

    def _publish_global_map(self) -> None:
        t0 = time.perf_counter()
        merged_pts = self.build_global_map()
        if merged_pts is None or merged_pts.size == 0:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.origin_frame_id
        cloud = pc2.create_cloud(header, self._pc_fields, merged_pts.tolist())
        self.global_pub.publish(cloud)

        if self.debug_profile:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.get_logger().info(
                f"[profile] publish pts={merged_pts.shape[0]} dt={dt_ms:.2f}ms bots={len(self.bots_dict)}"
            )

    def _heartbeat(self) -> None:
        self.get_logger().info(
            f"[profile] bots={len(self.bots_dict)} caches={[f'{k}:{v.shape[0]}' for k,v in self.local_maps.items()]}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
