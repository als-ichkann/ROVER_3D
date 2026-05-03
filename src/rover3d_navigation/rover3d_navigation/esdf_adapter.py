"""
ESDF adapter for planner: shared-memory zero-copy only.

说明：
- 仅保留 EsdfShmAdapter，不再支持 service/topic 查询。
- 在 SHM 读取的非负 ESDF 上构建 signed SDF（障碍内部为负值），
  以支持 CVaR 长尾风险计算。
"""

from __future__ import annotations

import os
import struct
import sys
from typing import Optional, Union

import numpy as np
from rclpy.node import Node
from scipy.ndimage import distance_transform_edt

from .trilinear_esdf import query_esdf_trilinear

_SHM_MAGIC = b"FIESESDF"
_SHM_HEADER_SIZE = 56


def _trilinear_scalar(
    grid: np.ndarray, ix0: int, iy0: int, iz0: int, dx: float, dy: float, dz: float
) -> float:
    v000 = float(grid[ix0, iy0, iz0])
    v001 = float(grid[ix0, iy0, iz0 + 1])
    v010 = float(grid[ix0, iy0 + 1, iz0])
    v011 = float(grid[ix0, iy0 + 1, iz0 + 1])
    v100 = float(grid[ix0 + 1, iy0, iz0])
    v101 = float(grid[ix0 + 1, iy0, iz0 + 1])
    v110 = float(grid[ix0 + 1, iy0 + 1, iz0])
    v111 = float(grid[ix0 + 1, iy0 + 1, iz0 + 1])
    c00 = v000 * (1.0 - dz) + v001 * dz
    c01 = v010 * (1.0 - dz) + v011 * dz
    c10 = v100 * (1.0 - dz) + v101 * dz
    c11 = v110 * (1.0 - dz) + v111 * dz
    c0 = c00 * (1.0 - dy) + c01 * dy
    c1 = c10 * (1.0 - dy) + c11 * dy
    return float(c0 * (1.0 - dx) + c1 * dx)


def _build_signed_sdf(
    dist_grid: np.ndarray,
    resolution: float,
    occupied_eps: float,
    inside_offset_vox: float,
) -> np.ndarray:
    """
    将非负 ESDF 转换为 signed SDF:
    - 外部保持正值
    - 障碍内部设为负值，幅值由“到自由空间边界的体素距离”给出
    """
    signed = np.asarray(dist_grid, dtype=np.float32).copy()
    occ = signed <= float(occupied_eps)
    if not np.any(occ):
        return signed

    inside_vox = distance_transform_edt(occ)
    inside_m = np.maximum(inside_vox - float(inside_offset_vox), 0.0) * float(resolution)
    signed[occ] = -inside_m[occ].astype(np.float32)
    return signed


class EsdfShmAdapter:
    """
    共享内存 ESDF 适配器（唯一实现）。
    读取 FIESTA 导出的 SHM 栅格并构建 signed SDF。
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
        signed_sdf_enable: bool = True,
        signed_sdf_occupied_eps: float = 1e-6,
        signed_sdf_inside_offset_vox: float = 0.5,
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
        self._grid_raw: Optional[np.ndarray] = None
        self._dist_grid: Optional[np.ndarray] = None
        self._ready = False

        self.signed_sdf_enable = bool(signed_sdf_enable)
        self.signed_sdf_occupied_eps = float(signed_sdf_occupied_eps)
        self.signed_sdf_inside_offset_vox = float(signed_sdf_inside_offset_vox)
        self._signed_stats_logged = False

        self._refresh()
        self._node.get_logger().info(
            f"EsdfShmAdapter: use shared memory {shm_name}; "
            f"signed_sdf={'on' if self.signed_sdf_enable else 'off'}"
        )

    def _shm_path(self) -> str:
        name = self._shm_name.lstrip("/") or "fiesta_esdf"
        if sys.platform == "linux":
            return f"/dev/shm/{name}"
        return self._shm_name

    def _refresh(self) -> bool:
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
            data_size = nx * ny * nz * 4 * 4

            with open(path, "rb") as f:
                f.seek(_SHM_HEADER_SIZE)
                data = np.frombuffer(f.read(data_size), dtype=np.float32)
            if data.size != nx * ny * nz * 4:
                return False

            self._grid_raw = data.reshape(nx, ny, nz, 4)
            self.dims = np.array([nx, ny, nz], dtype=int)
            self.origin = (ox, oy, oz)
            self.resolution = float(res)

            base_dist = self._grid_raw[..., 0]
            if self.signed_sdf_enable:
                self._dist_grid = _build_signed_sdf(
                    base_dist,
                    self.resolution,
                    self.signed_sdf_occupied_eps,
                    self.signed_sdf_inside_offset_vox,
                )
            else:
                self._dist_grid = base_dist.copy()

            self._ready = True
            return True
        except (OSError, struct.error, ValueError):
            return False

    def _ensure_ready(self) -> bool:
        if self._ready and self._dist_grid is not None:
            return True
        return self._refresh()

    def _sample_signed(self, gx: float, gy: float, gz: float) -> float:
        if self._dist_grid is None:
            return 5.0
        nx, ny, nz = self.dims
        ix0 = int(np.clip(np.floor(gx), 0, nx - 2))
        iy0 = int(np.clip(np.floor(gy), 0, ny - 2))
        iz0 = int(np.clip(np.floor(gz), 0, nz - 2))
        dx = float(np.clip(gx - ix0, 0.0, 1.0))
        dy = float(np.clip(gy - iy0, 0.0, 1.0))
        dz = float(np.clip(gz - iz0, 0.0, 1.0))
        return _trilinear_scalar(self._dist_grid, ix0, iy0, iz0, dx, dy, dz)

    def _trilinear(self, x: float, y: float, z: float) -> tuple[float, np.ndarray]:
        if not self._ensure_ready() or self._dist_grid is None:
            return (5.0, np.zeros(3))
        ox, oy, oz = self.origin
        res = self.resolution
        inv_res = 1.0 / res
        gx = (x - ox) / res
        gy = (y - oy) / res
        gz = (z - oz) / res

        dist = self._sample_signed(gx, gy, gz)
        grad_x = (self._sample_signed(gx + 1.0, gy, gz) - self._sample_signed(gx - 1.0, gy, gz)) * inv_res * 0.5
        grad_y = (self._sample_signed(gx, gy + 1.0, gz) - self._sample_signed(gx, gy - 1.0, gz)) * inv_res * 0.5
        grad_z = (self._sample_signed(gx, gy, gz + 1.0) - self._sample_signed(gx, gy, gz - 1.0)) * inv_res * 0.5
        return (float(dist), np.array([grad_x, grad_y, grad_z], dtype=float))

    def refresh(self) -> bool:
        ok = self._refresh()
        if ok and self.signed_sdf_enable and not self._signed_stats_logged and self._dist_grid is not None:
            g = self._dist_grid
            self._node.get_logger().info(
                "Signed SDF ready: min=%.4f max=%.4f neg_voxels=%d"
                % (float(np.min(g)), float(np.max(g)), int(np.sum(g < 0.0)))
            )
            self._signed_stats_logged = True
        return ok

    def query_trilinear_mpc(self, x: float, y: float, z: float) -> tuple[float, np.ndarray]:
        if not self._ensure_ready() or self._dist_grid is None:
            return (5.0, np.zeros(3))
        return query_esdf_trilinear(
            self._dist_grid,
            tuple(self.origin),
            float(self.resolution),
            tuple(self.dims),
            float(x),
            float(y),
            float(z),
            has_gradient=False,
        )

    def get_esdf(self, pos: Union[np.ndarray, list, tuple]) -> Union[float, np.ndarray]:
        pos = np.asarray(pos, dtype=float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        n = pos.shape[0]
        dists = np.full(n, 5.0, dtype=float)
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
        for t in np.linspace(0.0, 1.0, num_samples):
            pt = p1 + t * (p2 - p1)
            d, _ = self._trilinear(float(pt[0]), float(pt[1]), float(pt[2]))
            if d < safe_margin:
                return True
        return False

    @property
    def is_ready(self) -> bool:
        return self._ready
