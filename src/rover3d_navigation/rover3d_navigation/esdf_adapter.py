"""
ESDF Map Adapters for APF obstacle avoidance.
- EsdfMapAdapter: Uses esdf/query ROS2 service (higher latency).
- EsdfGridCache: Subscribes to /esdf/grid_full, local in-memory query (low latency).
- EsdfShmAdapter: 共享内存零拷贝，mmap 直接读 FIESTA 导出的 ESDF，无 ROS 序列化。
  需 FIESTA 以 use_esdf_shm:=true 启动；比 topic 快数十倍。
All expose get_esdf, compute_gradient, is_collision_line_segment interface.
"""

from __future__ import annotations

import os
import struct
import sys
from typing import Optional, Union

import numpy as np
import rclpy
from rclpy.node import Node

try:
    from esdf_map.srv import QueryEsdf
except ImportError:
    QueryEsdf = None

try:
    from std_msgs.msg import Float32MultiArray
except ImportError:
    Float32MultiArray = None

# 共享内存布局（与 FIESTA esdf_shm.hpp 一致）
_SHM_MAGIC = b"FIESESDF"
_SHM_HEADER_SIZE = 56


class EsdfMapAdapter:
    """
    Adapter for esdf/query service. Exposes origin, dims, resolution and
    get_esdf/compute_gradient/is_collision_line_segment for APF planners.
    """

    def __init__(
        self,
        node: Node,
        service_name: str = "esdf/query",
        frame_id: str = "map_origin",
        map_origin_x: float = -5.0,
        map_origin_y: float = -7.5,
        map_origin_z: float = 0.0,
        map_size_x: float = 22.0,
        map_size_y: float = 17.0,
        map_size_z: float = 6.0,
        resolution: float = 0.15,
    ) -> None:
        self._node = node
        self._service_name = service_name
        self._frame_id = frame_id
        self.origin = (map_origin_x, map_origin_y, map_origin_z)
        nx = max(1, int(round(map_size_x / resolution)))
        ny = max(1, int(round(map_size_y / resolution)))
        nz = max(1, int(round(map_size_z / resolution)))
        self.dims = np.array([nx, ny, nz], dtype=int)
        self.resolution = resolution

        if QueryEsdf is None:
            node.get_logger().warn(
                "esdf_map.srv.QueryEsdf not found; ESDF queries will return defaults"
            )
            self._client = None
            return

        self._client = node.create_client(QueryEsdf, service_name)
        if not self._client.wait_for_service(timeout_sec=5.0):
            node.get_logger().warn(
                f"esdf/query service '{service_name}' not available; "
                "obstacle avoidance may be disabled"
            )

    def _query(self, x: float, y: float, z: float) -> Optional[tuple[float, np.ndarray]]:
        """Query esdf/query service. Returns (distance, gradient) or None."""
        if self._client is None or QueryEsdf is None:
            return (5.0, np.zeros(3))  # safe default

        if not self._client.service_is_ready():
            return (5.0, np.zeros(3))

        req = QueryEsdf.Request()
        req.position.x = float(x)
        req.position.y = float(y)
        req.position.z = float(z)
        req.frame_id = self._frame_id

        try:
            future = self._client.call_async(req)
            rclpy.spin_until_future_complete(
                self._node, future, timeout_sec=1.0
            )
            if not future.done():
                return None
            resp = future.result()
            if resp is None or not resp.success:
                return None
            grad = np.array([resp.gradient.x, resp.gradient.y, resp.gradient.z])
            return (float(resp.distance), grad)
        except Exception:
            return None

    def get_esdf(self, pos: Union[np.ndarray, list, tuple]) -> Union[float, np.ndarray]:
        """
        Get ESDF distance at position(s).
        pos: (3,) or (N, 3) array. Returns float or (N,) array.
        """
        pos = np.asarray(pos, dtype=float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        n = pos.shape[0]
        dists = np.full(n, 5.0)  # default safe distance
        for i in range(n):
            r = self._query(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
            if r is not None:
                dists[i] = r[0]
        return float(dists[0]) if n == 1 else dists

    def compute_gradient(self, pos: Union[np.ndarray, list, tuple]) -> Optional[np.ndarray]:
        """Get ESDF gradient at position. Returns (3,) array or None."""
        pos = np.asarray(pos, dtype=float).flatten()
        if len(pos) < 3:
            return None
        r = self._query(float(pos[0]), float(pos[1]), float(pos[2]))
        if r is None:
            return None
        return r[1]

    def is_collision_line_segment(
        self,
        point1: Union[np.ndarray, list, tuple],
        point2: Union[np.ndarray, list, tuple],
        safe_margin: float = 0.2,
        num_samples: int = 10,
    ) -> bool:
        """
        Check if line segment from point1 to point2 collides with obstacles.
        Samples along the segment and returns True if any sample has distance < safe_margin.
        """
        p1 = np.asarray(point1, dtype=float).flatten()[:3]
        p2 = np.asarray(point2, dtype=float).flatten()[:3]
        for t in np.linspace(0, 1, num_samples):
            pt = p1 + t * (p2 - p1)
            r = self._query(float(pt[0]), float(pt[1]), float(pt[2]))
            if r is None:
                continue
            if r[0] < safe_margin:
                return True
        return False


class EsdfGridCache:
    """
    ESDF adapter backed by local grid cache. Subscribes to /esdf/grid_full,
    queries in-memory with trilinear interpolation. Zero service round-trip.
    Use when FIESTA publishes grid_full (publish_grid_full:=true).
    """

    def __init__(
        self,
        node: Node,
        grid_topic: str = "/esdf/grid_full",
        frame_id: str = "map_origin",
        map_origin_x: float = -5.0,
        map_origin_y: float = -7.5,
        map_origin_z: float = 0.0,
        map_size_x: float = 22.0,
        map_size_y: float = 17.0,
        map_size_z: float = 6.0,
        resolution: float = 0.15,
    ) -> None:
        self._node = node
        self._frame_id = frame_id
        self.origin = (map_origin_x, map_origin_y, map_origin_z)
        self.resolution = float(resolution)
        ox, oy, oz = self.origin
        # Expected dims; updated when grid arrives
        nx = max(1, int(round(map_size_x / resolution)))
        ny = max(1, int(round(map_size_y / resolution)))
        nz = max(1, int(round(map_size_z / resolution)))
        self.dims = np.array([nx, ny, nz], dtype=int)
        self._grid: Optional[np.ndarray] = None  # shape (nx, ny, nz) or (nx, ny, nz, 4)
        self._has_gradient = False
        self._ready = False

        if Float32MultiArray is None:
            node.get_logger().warn("std_msgs.Float32MultiArray not found")
            return

        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._sub = node.create_subscription(
            Float32MultiArray,
            grid_topic,
            self._cb_grid,
            qos,
        )
        node.get_logger().info(
            f"EsdfGridCache: subscribing to {grid_topic} (in-memory, zero service latency)"
        )

    def _cb_grid(self, msg: "Float32MultiArray") -> None:
        if len(msg.layout.dim) < 3:
            return
        nx = msg.layout.dim[0].size
        ny = msg.layout.dim[1].size
        nz = msg.layout.dim[2].size
        nch = msg.layout.dim[3].size if len(msg.layout.dim) > 3 else 1
        total = nx * ny * nz * max(nch, 1)
        if len(msg.data) < total:
            return
        data = np.array(msg.data, dtype=np.float32)
        if nch >= 4:
            self._grid = data.reshape(nx, ny, nz, 4)
            self._has_gradient = True
        else:
            self._grid = data.reshape(nx, ny, nz)
            self._has_gradient = False
        self.dims = np.array([nx, ny, nz], dtype=int)
        self._ready = True

    def _trilinear_interp(
        self, g: np.ndarray, ix0: int, iy0: int, iz0: int, dx: float, dy: float, dz: float
    ) -> np.ndarray:
        """Trilinear interpolation. g[i,j,k] or g[i,j,k,:], returns scalar or (4,) array."""
        if g.ndim == 4:
            v000 = g[ix0, iy0, iz0, :]
            v001 = g[ix0, iy0, iz0 + 1, :]
            v010 = g[ix0, iy0 + 1, iz0, :]
            v011 = g[ix0, iy0 + 1, iz0 + 1, :]
            v100 = g[ix0 + 1, iy0, iz0, :]
            v101 = g[ix0 + 1, iy0, iz0 + 1, :]
            v110 = g[ix0 + 1, iy0 + 1, iz0, :]
            v111 = g[ix0 + 1, iy0 + 1, iz0 + 1, :]
        else:
            v000 = g[ix0, iy0, iz0]
            v001 = g[ix0, iy0, iz0 + 1]
            v010 = g[ix0, iy0 + 1, iz0]
            v011 = g[ix0, iy0 + 1, iz0 + 1]
            v100 = g[ix0 + 1, iy0, iz0]
            v101 = g[ix0 + 1, iy0, iz0 + 1]
            v110 = g[ix0 + 1, iy0 + 1, iz0]
            v111 = g[ix0 + 1, iy0 + 1, iz0 + 1]
        c00 = v000 * (1 - dz) + v001 * dz
        c01 = v010 * (1 - dz) + v011 * dz
        c10 = v100 * (1 - dz) + v101 * dz
        c11 = v110 * (1 - dz) + v111 * dz
        c0 = c00 * (1 - dy) + c01 * dy
        c1 = c10 * (1 - dy) + c11 * dy
        return c0 * (1 - dx) + c1 * dx

    def _trilinear(self, x: float, y: float, z: float) -> tuple[float, np.ndarray]:
        """Trilinear interpolation. Returns (dist, grad) from FIESTA grid or finite-diff fallback."""
        if self._grid is None or not self._ready:
            return (5.0, np.zeros(3))
        ox, oy, oz = self.origin
        res = self.resolution
        inv_res = 1.0 / res
        ix_f = (x - ox) / res
        iy_f = (y - oy) / res
        iz_f = (z - oz) / res
        nx, ny, nz = self.dims
        ix0 = int(np.clip(np.floor(ix_f), 0, nx - 2))
        iy0 = int(np.clip(np.floor(iy_f), 0, ny - 2))
        iz0 = int(np.clip(np.floor(iz_f), 0, nz - 2))
        dx = np.clip(ix_f - ix0, 0.0, 1.0)
        dy = np.clip(iy_f - iy0, 0.0, 1.0)
        dz = np.clip(iz_f - iz0, 0.0, 1.0)

        if self._has_gradient:
            interp = self._trilinear_interp(self._grid, ix0, iy0, iz0, dx, dy, dz)
            dist = float(interp[0])
            grad = np.array([interp[1], interp[2], interp[3]], dtype=float)
            return (dist, grad)
        else:
            def _val(ix_f: float, iy_f: float, iz_f: float) -> float:
                i0 = int(np.clip(np.floor(ix_f), 0, nx - 2))
                j0 = int(np.clip(np.floor(iy_f), 0, ny - 2))
                k0 = int(np.clip(np.floor(iz_f), 0, nz - 2))
                dx0 = np.clip(ix_f - i0, 0.0, 1.0)
                dy0 = np.clip(iy_f - j0, 0.0, 1.0)
                dz0 = np.clip(iz_f - k0, 0.0, 1.0)
                return float(self._trilinear_interp(self._grid, i0, j0, k0, dx0, dy0, dz0))

            dist = _val(ix_f, iy_f, iz_f)
            grad_x = (_val(ix_f + 1, iy_f, iz_f) - _val(ix_f - 1, iy_f, iz_f)) * inv_res * 0.5
            grad_y = (_val(ix_f, iy_f + 1, iz_f) - _val(ix_f, iy_f - 1, iz_f)) * inv_res * 0.5
            grad_z = (_val(ix_f, iy_f, iz_f + 1) - _val(ix_f, iy_f, iz_f - 1)) * inv_res * 0.5
            return (dist, np.array([grad_x, grad_y, grad_z], dtype=float))

    def get_esdf(self, pos: Union[np.ndarray, list, tuple]) -> Union[float, np.ndarray]:
        pos = np.asarray(pos, dtype=float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        n = pos.shape[0]
        dists = np.full(n, 5.0)
        for i in range(n):
            d, _ = self._trilinear(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
            dists[i] = d
        return float(dists[0]) if n == 1 else dists

    def compute_gradient(self, pos: Union[np.ndarray, list, tuple]) -> Optional[np.ndarray]:
        pos = np.asarray(pos, dtype=float).flatten()
        if len(pos) < 3:
            return None
        _, grad = self._trilinear(float(pos[0]), float(pos[1]), float(pos[2]))
        return grad

    def is_collision_line_segment(
        self,
        point1: Union[np.ndarray, list, tuple],
        point2: Union[np.ndarray, list, tuple],
        safe_margin: float = 0.2,
        num_samples: int = 10,
    ) -> bool:
        p1 = np.asarray(point1, dtype=float).flatten()[:3]
        p2 = np.asarray(point2, dtype=float).flatten()[:3]
        for t in np.linspace(0, 1, num_samples):
            pt = p1 + t * (p2 - p1)
            d, _ = self._trilinear(float(pt[0]), float(pt[1]), float(pt[2]))
            if d < safe_margin:
                return True
        return False

    @property
    def is_ready(self) -> bool:
        """True if grid has been received and cache is valid."""
        return self._ready


class EsdfShmAdapter:
    """
    共享内存 ESDF 适配器。mmap 直接读取 FIESTA 导出的 /dev/shm/fiesta_esdf，
    无 topic 序列化、无网络拷贝，查询延迟在微秒级。

    要求: FIESTA 以 use_esdf_shm:=true 启动，且先于 Planner 运行一帧以完成首次导出。
    """

    def __init__(
        self,
        node: Node,
        shm_name: str = "/fiesta_esdf",
        frame_id: str = "map_origin",
        map_origin_x: float = -5.0,
        map_origin_y: float = -7.5,
        map_origin_z: float = 0.0,
        map_size_x: float = 22.0,
        map_size_y: float = 17.0,
        map_size_z: float = 6.0,
        resolution: float = 0.15,
    ) -> None:
        self._node = node
        self._shm_name = shm_name
        self._frame_id = frame_id
        self.origin = (map_origin_x, map_origin_y, map_origin_z)
        self.resolution = float(resolution)
        nx = max(1, int(round(map_size_x / resolution)))
        ny = max(1, int(round(map_size_y / resolution)))
        nz = max(1, int(round(map_size_z / resolution)))
        self.dims = np.array([nx, ny, nz], dtype=int)
        self._grid: Optional[np.ndarray] = None
        self._mmap_handle = None
        self._mmap_data = None
        self._ready = False
        self._refresh()

        node.get_logger().info(
            f"EsdfShmAdapter: use shared memory {shm_name} (zero-copy, no ROS topic)"
        )

    def _shm_path(self) -> str:
        """Linux: /fiesta_esdf -> /dev/shm/fiesta_esdf"""
        name = self._shm_name.lstrip("/") or "fiesta_esdf"
        if sys.platform == "linux":
            return f"/dev/shm/{name}"
        return self._shm_name

    def _refresh(self) -> bool:
        """从共享内存刷新栅格。返回 True 表示成功。"""
        path = self._shm_path()
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                header = f.read(_SHM_HEADER_SIZE)
            if len(header) < _SHM_HEADER_SIZE or header[:8] != _SHM_MAGIC:
                return False
            nx, ny, nz = struct.unpack("<III", header[8:20])
            res, ox, oy, oz = struct.unpack("<dddd", header[20:52])
            data_size = nx * ny * nz * 4 * 4  # 4 floats per voxel
            with open(path, "rb") as f:
                f.seek(_SHM_HEADER_SIZE)
                data = np.frombuffer(f.read(data_size), dtype=np.float32)
            if data.size != nx * ny * nz * 4:
                return False
            self._grid = data.reshape(nx, ny, nz, 4)
            self.dims = np.array([nx, ny, nz], dtype=int)
            self.origin = (ox, oy, oz)
            self.resolution = float(res)
            self._ready = True
            return True
        except (OSError, struct.error, ValueError):
            return False

    def _ensure_ready(self) -> bool:
        """确保栅格有效，启动时可能需等待 FIESTA 首次导出。"""
        if self._ready and self._grid is not None:
            return True
        return self._refresh()

    def _trilinear(
        self, x: float, y: float, z: float
    ) -> tuple[float, np.ndarray]:
        """三线性插值，返回 (dist, grad)。"""
        if not self._ensure_ready() or self._grid is None:
            return (5.0, np.zeros(3))
        ox, oy, oz = self.origin
        res = self.resolution
        nx, ny, nz = self.dims
        ix_f = (x - ox) / res
        iy_f = (y - oy) / res
        iz_f = (z - oz) / res
        ix0 = int(np.clip(np.floor(ix_f), 0, nx - 2))
        iy0 = int(np.clip(np.floor(iy_f), 0, ny - 2))
        iz0 = int(np.clip(np.floor(iz_f), 0, nz - 2))
        dx = np.clip(ix_f - ix0, 0.0, 1.0)
        dy = np.clip(iy_f - iy0, 0.0, 1.0)
        dz = np.clip(iz_f - iz0, 0.0, 1.0)
        g = self._grid
        v000 = g[ix0, iy0, iz0, :]
        v001 = g[ix0, iy0, iz0 + 1, :]
        v010 = g[ix0, iy0 + 1, iz0, :]
        v011 = g[ix0, iy0 + 1, iz0 + 1, :]
        v100 = g[ix0 + 1, iy0, iz0, :]
        v101 = g[ix0 + 1, iy0, iz0 + 1, :]
        v110 = g[ix0 + 1, iy0 + 1, iz0, :]
        v111 = g[ix0 + 1, iy0 + 1, iz0 + 1, :]
        c00 = v000 * (1 - dz) + v001 * dz
        c01 = v010 * (1 - dz) + v011 * dz
        c10 = v100 * (1 - dz) + v101 * dz
        c11 = v110 * (1 - dz) + v111 * dz
        c0 = c00 * (1 - dy) + c01 * dy
        c1 = c10 * (1 - dy) + c11 * dy
        interp = c0 * (1 - dx) + c1 * dx
        return (float(interp[0]), np.array(interp[1:4], dtype=float))

    def refresh(self) -> bool:
        """主动刷新共享内存内容（规划前调用以获取最新 ESDF）。"""
        return self._refresh()

    def query_trilinear_mpc(self, x: float, y: float, z: float) -> tuple[float, np.ndarray]:
        """
        供 MPC 控制专用的三线性插值，获取无人机位置的精确 ESDF 值与梯度。
        采用显式权重形式。
        """
        if not self._ensure_ready() or self._grid is None:
            return (5.0, np.zeros(3))
        from .trilinear_esdf import query_esdf_trilinear
        return query_esdf_trilinear(
            self._grid,
            tuple(self.origin),
            float(self.resolution),
            tuple(self.dims),
            float(x), float(y), float(z),
            has_gradient=True,
        )

    def get_esdf(self, pos: Union[np.ndarray, list, tuple]) -> Union[float, np.ndarray]:
        pos = np.asarray(pos, dtype=float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        n = pos.shape[0]
        dists = np.full(n, 5.0)
        for i in range(n):
            d, _ = self._trilinear(
                float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])
            )
            dists[i] = d
        return float(dists[0]) if n == 1 else dists

    def compute_gradient(
        self, pos: Union[np.ndarray, list, tuple]
    ) -> Optional[np.ndarray]:
        pos = np.asarray(pos, dtype=float).flatten()
        if len(pos) < 3:
            return None
        _, grad = self._trilinear(
            float(pos[0]), float(pos[1]), float(pos[2])
        )
        return grad

    def is_collision_line_segment(
        self,
        point1: Union[np.ndarray, list, tuple],
        point2: Union[np.ndarray, list, tuple],
        safe_margin: float = 0.2,
        num_samples: int = 10,
    ) -> bool:
        p1 = np.asarray(point1, dtype=float).flatten()[:3]
        p2 = np.asarray(point2, dtype=float).flatten()[:3]
        for t in np.linspace(0, 1, num_samples):
            pt = p1 + t * (p2 - p1)
            d, _ = self._trilinear(
                float(pt[0]), float(pt[1]), float(pt[2])
            )
            if d < safe_margin:
                return True
        return False

    @property
    def is_ready(self) -> bool:
        return self._ready
