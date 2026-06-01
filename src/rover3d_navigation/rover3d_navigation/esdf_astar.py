"""3D A* path planning on FIESTA ESDF (EsdfShmAdapter)."""

from __future__ import annotations

import heapq
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# 26-connected voxel neighbors (dx, dy, dz, step_cost multiplier)
_NEIGHBORS_26: List[Tuple[int, int, int, float]] = []
for _dx in (-1, 0, 1):
    for _dy in (-1, 0, 1):
        for _dz in (-1, 0, 1):
            if _dx == 0 and _dy == 0 and _dz == 0:
                continue
            _NEIGHBORS_26.append((_dx, _dy, _dz, float(np.linalg.norm((_dx, _dy, _dz)))))


def _as_vec3(pos: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    return np.asarray(pos, dtype=float).reshape(3)


def _world_to_voxel(
    pos: Union[np.ndarray, Sequence[float]],
    origin: Tuple[float, float, float],
    resolution: float,
) -> Tuple[int, int, int]:
    ox, oy, oz = origin
    p = _as_vec3(pos)
    return (
        int(np.floor((p[0] - ox) / resolution)),
        int(np.floor((p[1] - oy) / resolution)),
        int(np.floor((p[2] - oz) / resolution)),
    )


def _voxel_to_world(
    ix: int,
    iy: int,
    iz: int,
    origin: Tuple[float, float, float],
    resolution: float,
) -> np.ndarray:
    ox, oy, oz = origin
    return np.array(
        [ox + (ix + 0.5) * resolution, oy + (iy + 0.5) * resolution, oz + (iz + 0.5) * resolution],
        dtype=float,
    )


def _in_bounds(ix: int, iy: int, iz: int, dims: np.ndarray) -> bool:
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    return 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz


def _esdf_distance(esdf_map, world_pos: np.ndarray) -> float:
    d = esdf_map.get_esdf(world_pos)
    return float(np.asarray(d, dtype=float).reshape(-1)[0])


def is_world_point_free(esdf_map, world_pos, safe_margin: float) -> bool:
    if esdf_map is None:
        return True
    if hasattr(esdf_map, "_ensure_ready") and not esdf_map._ensure_ready():
        return True
    return _esdf_distance(esdf_map, _as_vec3(world_pos)) >= float(safe_margin)


def _find_nearest_free_voxel(
    esdf_map,
    seed_vox: Tuple[int, int, int],
    origin: Tuple[float, float, float],
    resolution: float,
    dims: np.ndarray,
    safe_margin: float,
    max_radius: int = 12,
) -> Optional[Tuple[int, int, int]]:
    sx, sy, sz = seed_vox
    if _in_bounds(sx, sy, sz, dims):
        w = _voxel_to_world(sx, sy, sz, origin, resolution)
        if is_world_point_free(esdf_map, w, safe_margin):
            return (sx, sy, sz)
    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if max(abs(dx), abs(dy), abs(dz)) != r:
                        continue
                    ix, iy, iz = sx + dx, sy + dy, sz + dz
                    if not _in_bounds(ix, iy, iz, dims):
                        continue
                    w = _voxel_to_world(ix, iy, iz, origin, resolution)
                    if is_world_point_free(esdf_map, w, safe_margin):
                        return (ix, iy, iz)
    return None


def project_to_free(
    esdf_map,
    pos: Union[np.ndarray, Sequence[float]],
    safe_margin: float,
    max_iters: int = 24,
    step_scale: float = 0.6,
) -> np.ndarray:
    """沿 ESDF 梯度将点投影到满足 safe_margin 的自由空间。"""
    p = _as_vec3(pos).copy()
    margin = float(safe_margin)
    for _ in range(max_iters):
        d = _esdf_distance(esdf_map, p)
        if d >= margin:
            return p
        grad = None
        if hasattr(esdf_map, "compute_gradient"):
            grad = esdf_map.compute_gradient(p)
        if grad is None:
            break
        g = np.asarray(grad, dtype=float).reshape(3)
        gn = float(np.linalg.norm(g))
        if gn < 1e-9:
            break
        push = max(margin - d + 0.02, 0.02) * step_scale
        p = p + (g / gn) * push
    return p


def path_has_segment_collision(
    esdf_map,
    path: np.ndarray,
    safe_margin: float,
    num_samples: int = 12,
) -> bool:
    pts = np.asarray(path, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        return not is_world_point_free(esdf_map, pts[0], safe_margin)
    if hasattr(esdf_map, "is_collision_line_segment"):
        for i in range(pts.shape[0] - 1):
            if esdf_map.is_collision_line_segment(
                pts[i], pts[i + 1], safe_margin=safe_margin, num_samples=num_samples
            ):
                return True
        return False
    for p in pts:
        if not is_world_point_free(esdf_map, p, safe_margin):
            return True
    return False


def sanitize_path(
    esdf_map,
    path: np.ndarray,
    safe_margin: float,
) -> np.ndarray:
    """确保路径点与段均满足安全裕度；必要时截断到最后一个有效点。"""
    pts = np.asarray(path, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    clean = [project_to_free(esdf_map, pts[0], safe_margin)]
    for i in range(1, pts.shape[0]):
        nxt = project_to_free(esdf_map, pts[i], safe_margin)
        if hasattr(esdf_map, "is_collision_line_segment") and esdf_map.is_collision_line_segment(
            clean[-1], nxt, safe_margin=safe_margin, num_samples=12
        ):
            break
        clean.append(nxt)
    return np.asarray(clean, dtype=float)


def astar_3d(
    esdf_map,
    start: Union[np.ndarray, Sequence[float]],
    goal: Union[np.ndarray, Sequence[float]],
    safe_margin: float = 0.3,
    max_iterations: int = 80000,
) -> Optional[np.ndarray]:
    """
    在 ESDF 栅格上运行 3D A*，返回世界坐标路径 (N, 3)，含起点与终点。
    若 ESDF 未就绪或搜索失败则返回 None。
    """
    if esdf_map is None:
        return None
    if hasattr(esdf_map, "refresh"):
        esdf_map.refresh()
    if hasattr(esdf_map, "_ensure_ready") and not esdf_map._ensure_ready():
        return None

    origin = tuple(float(x) for x in np.asarray(esdf_map.origin, dtype=float).reshape(3))
    resolution = float(esdf_map.resolution)
    dims = np.asarray(esdf_map.dims, dtype=int).reshape(3)

    start_w = _as_vec3(start)
    goal_w = _as_vec3(goal)
    start_vox = _world_to_voxel(start_w, origin, resolution)
    goal_vox = _world_to_voxel(goal_w, origin, resolution)

    start_free = _find_nearest_free_voxel(esdf_map, start_vox, origin, resolution, dims, safe_margin)
    goal_free = _find_nearest_free_voxel(esdf_map, goal_vox, origin, resolution, dims, safe_margin)
    if start_free is None or goal_free is None:
        return None

    if start_free == goal_free:
        seg = np.vstack([start_w, goal_w])
        return sanitize_path(esdf_map, seg, safe_margin)

    goal_end = goal_w if is_world_point_free(esdf_map, goal_w, safe_margin) else _voxel_to_world(
        goal_free[0], goal_free[1], goal_free[2], origin, resolution
    )
    goal_end = project_to_free(esdf_map, goal_end, safe_margin)

    # 直线可达则直接返回（采样更密，终点用安全投影）
    if hasattr(esdf_map, "is_collision_line_segment"):
        if not esdf_map.is_collision_line_segment(
            start_w, goal_end, safe_margin=safe_margin, num_samples=24
        ):
            return sanitize_path(esdf_map, np.vstack([start_w, goal_end]), safe_margin)

    open_heap: List[Tuple[float, float, Tuple[int, int, int]]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start_free))
    came_from: dict = {}
    g_score = {start_free: 0.0}
    closed: set = set()
    goal_reached = False

    gx, gy, gz = goal_free
    for _ in range(int(max_iterations)):
        if not open_heap:
            break
        _, g_curr, curr = heapq.heappop(open_heap)
        if curr in closed:
            continue
        closed.add(curr)
        if curr == goal_free:
            goal_reached = True
            break

        cx, cy, cz = curr
        w_curr = _voxel_to_world(cx, cy, cz, origin, resolution)
        for dx, dy, dz, step_mul in _NEIGHBORS_26:
            nb = (cx + dx, cy + dy, cz + dz)
            if nb in closed or not _in_bounds(nb[0], nb[1], nb[2], dims):
                continue
            w_nb = _voxel_to_world(nb[0], nb[1], nb[2], origin, resolution)
            if not is_world_point_free(esdf_map, w_nb, safe_margin):
                continue
            if hasattr(esdf_map, "is_collision_line_segment") and esdf_map.is_collision_line_segment(
                w_curr, w_nb, safe_margin=safe_margin, num_samples=8
            ):
                continue
            tentative = g_curr + step_mul * resolution
            if tentative >= g_score.get(nb, float("inf")):
                continue
            came_from[nb] = curr
            g_score[nb] = tentative
            h = float(np.linalg.norm(np.array(nb) - np.array(goal_free))) * resolution
            heapq.heappush(open_heap, (tentative + h, tentative, nb))

    if not goal_reached:
        return None

    path_vox = [goal_free]
    cur = goal_free
    while cur in came_from:
        cur = came_from[cur]
        path_vox.append(cur)
    path_vox.reverse()

    path_world = np.array(
        [_voxel_to_world(v[0], v[1], v[2], origin, resolution) for v in path_vox],
        dtype=float,
    )
    path_world[0] = start_w
    path_world[-1] = goal_end
    return sanitize_path(esdf_map, path_world, safe_margin)


def resample_path(
    path: np.ndarray,
    step_size: float,
    max_points: int,
) -> np.ndarray:
    """按弧长重采样路径，至少返回终点。"""
    pts = np.asarray(path, dtype=float).reshape(-1, 3)
    if pts.shape[0] <= 1:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(seg))
    if total <= 1e-9:
        return pts[-1:].copy()
    step = max(float(step_size), 1e-6)
    n = min(max(1, int(np.ceil(total / step))), max(1, int(max_points)))
    targets = np.linspace(step, total, n)
    out = []
    acc = 0.0
    j = 0
    for t in targets:
        while j < len(seg) and acc + seg[j] < t - 1e-9:
            acc += seg[j]
            j += 1
        if j >= len(seg):
            out.append(pts[-1])
            continue
        alpha = (t - acc) / max(seg[j], 1e-9)
        out.append((1.0 - alpha) * pts[j] + alpha * pts[j + 1])
    resampled = np.asarray(out, dtype=float)
    return resampled if resampled.shape[0] > 0 else pts[-1:].copy()


def paths_to_agent_snapshots(
    agent_paths: List[np.ndarray],
) -> List[np.ndarray]:
    """将每架无人机的路径转为同步快照 list[(n_agents,3), ...]。"""
    if not agent_paths:
        return []
    max_len = max(p.shape[0] for p in agent_paths)
    snapshots = []
    for t in range(max_len):
        snap = np.zeros((len(agent_paths), 3), dtype=float)
        for i, path in enumerate(agent_paths):
            idx = min(t, path.shape[0] - 1)
            snap[i] = path[idx]
        snapshots.append(snap)
    return snapshots
