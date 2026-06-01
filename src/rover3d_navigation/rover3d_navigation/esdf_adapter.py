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
from typing import Optional, Tuple, Union

import numpy as np
from rclpy.node import Node
from scipy.ndimage import distance_transform_edt

_SHM_MAGIC = b"FIESESDF"
_SHM_HEADER_SIZE = 56
_SHM_LAYOUT_VERSION_SIGNED = 2


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


def _analytic_gradient(
    fx: float,
    fy: float,
    fz: float,
    v000: float,
    v100: float,
    v010: float,
    v110: float,
    v001: float,
    v101: float,
    v011: float,
    v111: float,
) -> np.ndarray:
    """三线性多项式解析梯度（栅格坐标系）。"""
    fy0, fz0 = 1.0 - fy, 1.0 - fz
    fx0 = 1.0 - fx
    gx = fy0 * fz0 * (v100 - v000) + fy * fz0 * (v110 - v010) + fy0 * fz * (v101 - v001) + fy * fz * (v111 - v011)
    gy = fx0 * fz0 * (v010 - v000) + fx * fz0 * (v110 - v100) + fx0 * fz * (v011 - v001) + fx * fz * (v111 - v101)
    gz = fx0 * fy0 * (v001 - v000) + fx * fy0 * (v101 - v100) + fx0 * fy * (v011 - v010) + fx * fy * (v111 - v110)
    return np.array([gx, gy, gz], dtype=float)


def trilinear_interp(
    grid: np.ndarray,
    ix0: int,
    iy0: int,
    iz0: int,
    fx: float,
    fy: float,
    fz: float,
    has_gradient: bool = True,
) -> Tuple[float, np.ndarray]:
    """三线性插值，支持 (nx,ny,nz) 与 (nx,ny,nz,4) 栅格。"""
    fx0, fy0, fz0 = 1.0 - fx, 1.0 - fy, 1.0 - fz
    w000 = fx0 * fy0 * fz0
    w100 = fx * fy0 * fz0
    w010 = fx0 * fy * fz0
    w110 = fx * fy * fz0
    w001 = fx0 * fy0 * fz
    w101 = fx * fy0 * fz
    w011 = fx0 * fy * fz
    w111 = fx * fy * fz

    if has_gradient and grid.ndim == 4:
        v000 = grid[ix0, iy0, iz0, :]
        v100 = grid[ix0 + 1, iy0, iz0, :]
        v010 = grid[ix0, iy0 + 1, iz0, :]
        v110 = grid[ix0 + 1, iy0 + 1, iz0, :]
        v001 = grid[ix0, iy0, iz0 + 1, :]
        v101 = grid[ix0 + 1, iy0, iz0 + 1, :]
        v011 = grid[ix0, iy0 + 1, iz0 + 1, :]
        v111 = grid[ix0 + 1, iy0 + 1, iz0 + 1, :]
        dist = (
            w000 * v000[0] + w100 * v100[0] + w010 * v010[0] + w110 * v110[0]
            + w001 * v001[0] + w101 * v101[0] + w011 * v011[0] + w111 * v111[0]
        )
        grad = np.array(
            [
                w000 * v000[1] + w100 * v100[1] + w010 * v010[1] + w110 * v110[1]
                + w001 * v001[1] + w101 * v101[1] + w011 * v011[1] + w111 * v111[1],
                w000 * v000[2] + w100 * v100[2] + w010 * v010[2] + w110 * v110[2]
                + w001 * v001[2] + w101 * v101[2] + w011 * v011[2] + w111 * v111[2],
                w000 * v000[3] + w100 * v100[3] + w010 * v010[3] + w110 * v110[3]
                + w001 * v001[3] + w101 * v101[3] + w011 * v011[3] + w111 * v111[3],
            ],
            dtype=float,
        )
        return (float(dist), grad)

    v000 = float(grid[ix0, iy0, iz0])
    v100 = float(grid[ix0 + 1, iy0, iz0])
    v010 = float(grid[ix0, iy0 + 1, iz0])
    v110 = float(grid[ix0 + 1, iy0 + 1, iz0])
    v001 = float(grid[ix0, iy0, iz0 + 1])
    v101 = float(grid[ix0 + 1, iy0, iz0 + 1])
    v011 = float(grid[ix0, iy0 + 1, iz0 + 1])
    v111 = float(grid[ix0 + 1, iy0 + 1, iz0 + 1])
    dist = (
        w000 * v000 + w100 * v100 + w010 * v010 + w110 * v110
        + w001 * v001 + w101 * v101 + w011 * v011 + w111 * v111
    )
    grad = _analytic_gradient(
        fx,
        fy,
        fz,
        v000,
        v100,
        v010,
        v110,
        v001,
        v101,
        v011,
        v111,
    )
    return (float(dist), grad)


def query_esdf_trilinear(
    grid: np.ndarray,
    origin: Tuple[float, float, float],
    resolution: float,
    dims: Tuple[int, int, int],
    x: float,
    y: float,
    z: float,
    has_gradient: bool = True,
    default_dist: float = 5.0,
) -> Tuple[float, np.ndarray]:
    """在 ESDF 栅格中三线性插值查询某点精确值。"""
    ox, oy, oz = origin
    nx, ny, nz = dims
    if nx < 2 or ny < 2 or nz < 2:
        return (default_dist, np.zeros(3))

    gx = (x - ox) / resolution
    gy = (y - oy) / resolution
    gz = (z - oz) / resolution

    ix = int(np.clip(np.floor(gx), 0, nx - 2))
    iy = int(np.clip(np.floor(gy), 0, ny - 2))
    iz = int(np.clip(np.floor(gz), 0, nz - 2))

    fx = float(np.clip(gx - ix, 0.0, 1.0))
    fy = float(np.clip(gy - iy, 0.0, 1.0))
    fz = float(np.clip(gz - iz, 0.0, 1.0))

    dist, grad_grid = trilinear_interp(grid, ix, iy, iz, fx, fy, fz, has_gradient)
    grad_world = grad_grid * (1.0 / resolution)
    return (float(dist), grad_world)


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
        source_is_signed: bool = True,
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
        self._map_populated = False
        self._layout_version = 0
        self._source_is_signed_runtime = bool(source_is_signed)

        self.signed_sdf_enable = bool(signed_sdf_enable)
        self.source_is_signed = bool(source_is_signed)
        self.signed_sdf_occupied_eps = float(signed_sdf_occupied_eps)
        self.signed_sdf_inside_offset_vox = float(signed_sdf_inside_offset_vox)
        self._signed_stats_logged = False

        self._refresh()
        self._node.get_logger().info(
            f"EsdfShmAdapter: use shared memory {shm_name}; "
            f"signed_sdf={'on' if self.signed_sdf_enable else 'off'}; "
            f"source_is_signed={'yes' if self.source_is_signed else 'no'}"
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
            (layout_version,) = struct.unpack("<I", header[52:56])
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
            self._layout_version = int(layout_version)
            self._source_is_signed_runtime = bool(
                self.source_is_signed or (self._layout_version >= _SHM_LAYOUT_VERSION_SIGNED)
            )

            base_dist = self._grid_raw[..., 0]
            if self.signed_sdf_enable:
                if self._source_is_signed_runtime:
                    self._dist_grid = base_dist.copy()
                else:
                    self._dist_grid = _build_signed_sdf(
                        base_dist,
                        self.resolution,
                        self.signed_sdf_occupied_eps,
                        self.signed_sdf_inside_offset_vox,
                    )
            else:
                self._dist_grid = base_dist.copy()

            # 仅当 ESDF 真正含有自由空间（存在正距离体素）时才视为就绪。
            # FIESTA 启动早期写入的是全 0 空图，此时不应让规划/控制抢跑。
            self._map_populated = bool(np.any(self._dist_grid > 0.0))
            self._ready = self._map_populated
            return self._ready
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
                "Signed SDF ready: min=%.4f max=%.4f neg_voxels=%d layout_ver=%d source_is_signed=%s"
                % (
                    float(np.min(g)),
                    float(np.max(g)),
                    int(np.sum(g < 0.0)),
                    int(self._layout_version),
                    "yes" if self._source_is_signed_runtime else "no",
                )
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
