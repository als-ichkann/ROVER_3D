"""
ESDF 三线性插值模块

在单位立方体 [0,1]³ 内，对八个顶点采用显式权重形式插值，获取栅格中任意点的
精确 ESDF 距离与梯度，用于 MPC 控制中无人机位置的约束计算。

显式权重：w_ijk = (1-i+2i·fx)(1-j+2j·fy)(1-k+2k·fz)，其中 i,j,k∈{0,1}
插值值：d = Σ w_ijk · d_ijk
梯度：对多项式求偏导得解析梯度（世界坐标需乘 inv_resolution）
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


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
    """
    三线性插值（显式权重形式）。

    八个顶点 (i,j,k)，权重 w_ijk = (1-fx)^(1-i)*fx^i * (1-fy)^(1-j)*fy^j * (1-fz)^(1-k)*fz^k
    插值值 d = Σ w_ijk * v_ijk

    参数
    -----
    grid : (nx,ny,nz) 或 (nx,ny,nz,4)，4通道为 [dist,gx,gy,gz]
    ix0,iy0,iz0 : 基 voxel 索引
    fx,fy,fz : [0,1] 相对坐标
    has_gradient : 是否含梯度通道
    """
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
        grad = np.array([
            w000 * v000[1] + w100 * v100[1] + w010 * v010[1] + w110 * v110[1]
            + w001 * v001[1] + w101 * v101[1] + w011 * v011[1] + w111 * v111[1],
            w000 * v000[2] + w100 * v100[2] + w010 * v010[2] + w110 * v110[2]
            + w001 * v001[2] + w101 * v101[2] + w011 * v011[2] + w111 * v111[2],
            w000 * v000[3] + w100 * v100[3] + w010 * v010[3] + w110 * v110[3]
            + w001 * v001[3] + w101 * v101[3] + w011 * v011[3] + w111 * v111[3],
        ], dtype=float)
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
        fx, fy, fz, v000, v100, v010, v110, v001, v101, v011, v111
    )
    return (float(dist), grad)


def _analytic_gradient(
    fx: float, fy: float, fz: float,
    v000: float, v100: float, v010: float, v110: float,
    v001: float, v101: float, v011: float, v111: float,
) -> np.ndarray:
    """三线性多项式解析梯度（栅格坐标 ∂/∂fx 等），世界坐标梯度 = grad * inv_res"""
    fy0, fz0 = 1.0 - fy, 1.0 - fz
    fx0 = 1.0 - fx
    gx = fy0 * fz0 * (v100 - v000) + fy * fz0 * (v110 - v010) + fy0 * fz * (v101 - v001) + fy * fz * (v111 - v011)
    gy = fx0 * fz0 * (v010 - v000) + fx * fz0 * (v110 - v100) + fx0 * fz * (v011 - v001) + fx * fz * (v111 - v101)
    gz = fx0 * fy0 * (v001 - v000) + fx * fy0 * (v101 - v100) + fx0 * fy * (v011 - v010) + fx * fy * (v111 - v110)
    return np.array([gx, gy, gz], dtype=float)


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
    """
    在 ESDF 栅格中三线性插值查询某点精确值（用于 MPC 无人机位置约束）。

    参数
    -----
    grid : (nx,ny,nz) 或 (nx,ny,nz,4)，SHM 格式为 4 通道
    origin, resolution, dims : 栅格参数
    x,y,z : 世界坐标
    """
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
    inv_res = 1.0 / resolution
    grad_world = grad_grid * inv_res
    return (float(dist), grad_world)
