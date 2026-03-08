from __future__ import annotations

import re
from typing import Optional
import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

class BotFrameResolver:
    """Parse bot ids from TF frame names like '<prefix><id>/...'. """

    def __init__(self, bot_prefix: str) -> None:
        self._pattern = re.compile(rf"(?:^|/){re.escape(bot_prefix)}(\d+)(?:/|$)")

    def bot_id_from_frame(self, frame_id: str) -> Optional[int]:
        """Extract numeric bot id from a frame path segment, if present."""
        normalized = frame_id.lstrip('/')
        match = self._pattern.search(normalized)
        return int(match.group(1)) if match else None


def quat_to_rot_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x*x + y*y + z*z + w*w
    if n < 1e-12: return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([
        [1.0 - (yy + zz),       xy - wz,       xz + wy],
        [      xy + wz, 1.0 - (xx + zz),       yz - wx],
        [      xz - wy,       yz + wx, 1.0 - (xx + yy)],
    ], dtype=float)

#  -------------------

def pcd_to_xyz_safe(msg: PointCloud2, skip_rate:int=1) -> np.ndarray:
    """Return Nx3 float32 array of xyz points; be robust to malformed input."""
    raw = np.fromiter(
        pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True),
        dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)],
    )

    if raw.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    pts = np.stack((raw['x'], raw['y'], raw['z']), axis=1).astype(np.float32, copy=False)
    return pts[::skip_rate] if skip_rate > 1 else pts

# ------------------Fast Zero-copy XYZ view over PointCloud2----------------------
def pcd_to_xyz_fast(msg: PointCloud2, skip_rate:int=1) -> np.ndarray:
    fields = [f.name for f in msg.fields]
    if not all(k in fields for k in ('x','y','z')):
        return np.zeros((0, 3), dtype=np.float32)
    dtype = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('_', f'V{msg.point_step - 12}')
    ])
    raw = np.frombuffer(msg.data, dtype=dtype)
    pts = np.vstack((raw['x'], raw['y'], raw['z'])).T
    return pts[::skip_rate] if skip_rate > 1 else pts


def pcd_to_xyzi_fast(msg: PointCloud2, skip_rate: int = 1) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return (xyz Nx3, intensity Nx1 or None if no intensity field).
    Used for filtering high-reflectivity points (teammates) before ESDF.
    """
    field_names = [f.name for f in msg.fields]
    if not all(k in field_names for k in ('x', 'y', 'z')):
        return np.zeros((0, 3), dtype=np.float32), None
    if 'intensity' not in field_names:
        return pcd_to_xyz_fast(msg, skip_rate), None

    def _get_offset(name: str) -> int:
        for f in msg.fields:
            if f.name == name:
                return f.offset
        return -1

    oi = _get_offset('intensity')
    if oi < 0:
        return pcd_to_xyz_fast(msg, skip_rate), None

    dt = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('_pad', f'V{max(0, oi - 12)}'),
        ('intensity', np.float32),
        ('_rest', f'V{max(0, msg.point_step - oi - 4)}'),
    ])
    raw = np.frombuffer(msg.data, dtype=dt)
    pts = np.vstack((raw['x'], raw['y'], raw['z'])).T.astype(np.float32)
    intensity = raw['intensity'].reshape(-1, 1).astype(np.float32)
    if skip_rate > 1:
        pts = pts[::skip_rate]
        intensity = intensity[::skip_rate]
    return pts, intensity


# ------------------Fast Voxelization----------------------
def voxelize_numpy(pts: np.ndarray, voxel_size: float=0.1) -> np.ndarray:
    """Fastest voxel filter"""
    return voxel_centroids_ravel_bincount(pts, voxel_size)

def voxel_centroids_ravel_bincount(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0: return points
    vs = float(voxel_size)
    grid = np.floor(points / vs).astype(np.int64)
    gmin = grid.min(axis=0)
    grid -= gmin
    span = grid.max(axis=0) + 1
    cap = np.iinfo(np.int64).max
    if int(span[0]) * int(span[1]) * int(span[2]) >= cap:
        return voxel_centroids_sort_reduce(points, voxel_size)
    lin = grid[:, 0] + span[0] * (grid[:, 1] + span[1] * grid[:, 2])
    lin = lin.astype(np.int64)
    uniq, inv, counts = np.unique(lin, return_inverse=True, return_counts=True)
    x = np.bincount(inv, weights=points[:, 0].astype(np.float64), minlength=uniq.shape[0])
    y = np.bincount(inv, weights=points[:, 1].astype(np.float64), minlength=uniq.shape[0])
    z = np.bincount(inv, weights=points[:, 2].astype(np.float64), minlength=uniq.shape[0])
    centers = np.stack((x, y, z), axis=1) / counts[:, None]
    return centers.astype(np.float32)

def voxel_centroids_sort_reduce(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0: return points
    vs = float(voxel_size)
    grid = np.floor(points / vs).astype(np.int64)
    order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0]))
    g = grid[order]
    p = points[order].astype(np.float64)
    change = np.any(np.diff(g, axis=0) != 0, axis=1)
    idx = np.concatenate(([True], change)).nonzero()[0]
    counts = np.diff(np.append(idx, g.shape[0]))
    sums = np.add.reduceat(p, idx, axis=0)
    return (sums / counts[:, None]).astype(np.float32)