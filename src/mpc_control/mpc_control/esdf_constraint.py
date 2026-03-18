"""
ESDF 不等式约束接口，用于 MPC 避障。

从 SHM 零拷贝读取 ESDF，采用三线性插值（参考 zhuanlan.zhihu.com/p/671385710）获取
无人机位置的精确距离 d(p_ref) 与梯度 ∇d，基于一阶泰勒展开：

    d(p) ≈ d(p_ref) + ∇d^T (p - p_ref)

避障约束 d(p) >= d_safe 化为：

    ∇d^T p >= ∇d^T p_ref + d_safe - d(p_ref)
"""

from typing import Optional, Tuple, Union

import numpy as np


def make_mpc_esdf_adapter(raw_adapter):
    """将 EsdfShmAdapter 包装为 MPC 专用 adapter，使用三线性插值获取精确 ESDF。"""
    if raw_adapter is None:
        return None
    if hasattr(raw_adapter, "query_trilinear_mpc"):
        return MpcEsdfTrilinearAdapter(raw_adapter)
    return raw_adapter


class MpcEsdfTrilinearAdapter:
    """MPC 专用 ESDF 适配器，通过三线性插值获取无人机位置的精确距离与梯度。"""

    def __init__(self, raw_adapter):
        self._adapter = raw_adapter

    def get_esdf(self, pos):
        pos = np.asarray(pos, dtype=float).flatten()[:3]
        if len(pos) < 3:
            return 5.0
        d, _ = self._adapter.query_trilinear_mpc(float(pos[0]), float(pos[1]), float(pos[2]))
        return float(d)

    def compute_gradient(self, pos):
        pos = np.asarray(pos, dtype=float).flatten()[:3]
        if len(pos) < 3:
            return np.zeros(3)
        _, g = self._adapter.query_trilinear_mpc(float(pos[0]), float(pos[1]), float(pos[2]))
        return g

    @property
    def origin(self):
        return self._adapter.origin

    def refresh(self):
        if hasattr(self._adapter, "refresh"):
            return self._adapter.refresh()
        return False


def compute_esdf_inequality_constraints(
    positions: np.ndarray,
    d_safe: float,
    adapter,
    n_state: int = 9,
    n_control: int = 3,
    n_horizon: int = 6,
    min_gradient_norm: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 ESDF 一阶泰勒不等式约束 F_esdf @ z <= g_esdf。

    约束（用户形式）：∇d^T p >= ∇d^T p_ref + d_safe - d(p_ref)
    QP 形式：-∇d^T p <= -(∇d^T p_ref + d_safe - d(p_ref))
    即 F_row = -∇d，g_val = d(p_ref) - d_safe - ∇d^T p_ref

    adapter 从 SHM 等零拷贝源读取 get_esdf(pos)、compute_gradient(pos)。

    参数
    -----
    positions : (N+1, 3) ndarray
        各预测步的参考点位置 p_ref，用于泰勒展开线性化。
    d_safe : float
        安全余量 [m]。
    adapter : get_esdf(pos)、compute_gradient(pos)
    """
    n_steps = positions.shape[0]
    nz = (n_horizon + 1) * n_state + n_horizon * n_control
    F_rows = []
    g_vals = []

    for t in range(min(n_steps, n_horizon + 1)):
        p = np.asarray(positions[t], dtype=float).flatten()[:3]
        if len(p) < 3:
            continue

        try:
            d = adapter.get_esdf(p)
            g = adapter.compute_gradient(p)
        except Exception:
            continue

        dist = float(d) if not hasattr(d, "__len__") else float(d[0]) if len(d) else 5.0
        grad = np.asarray(g, dtype=float).flatten()[:3] if g is not None else np.zeros(3)

        if np.linalg.norm(grad) < min_gradient_norm:
            continue

        # 约束：∇d^T p >= ∇d^T p_ref + d_safe - d(p_ref)
        # QP：-∇d^T p <= -(∇d^T p_ref + d_safe - d(p_ref))
        # g_val = d(p_ref) - d_safe - ∇d^T p_ref
        gx, gy, gz = -grad[0], -grad[1], -grad[2]
        row = np.zeros(nz)
        base = t * n_state
        row[base + 0] = gx
        row[base + 3] = gy
        row[base + 6] = gz

        g_val = float(dist) - d_safe - np.dot(grad, p)

        if g_val > -1e6:
            F_rows.append(row)
            g_vals.append(g_val)

    if len(F_rows) == 0:
        return np.zeros((0, nz)), np.zeros(0)

    F_esdf = np.vstack(F_rows)
    g_esdf = np.array(g_vals, dtype=float)
    return F_esdf, g_esdf
