import numpy as np
import heapq
from scipy.linalg import sqrtm
from scipy.optimize import linprog
from scipy.stats import multivariate_normal


def _to_real_matrix(mat: np.ndarray) -> np.ndarray:
    return np.asarray(np.real_if_close(mat), dtype=float)


def _symmetrize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def _normalize_weights(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w[w < 0] = 0.0
    s = float(np.sum(w))
    if s <= eps:
        return np.ones_like(w) / max(len(w), 1)
    return w / s


def _solve_ot_plan(src_weights: np.ndarray, dst_weights: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    src_weights = _normalize_weights(src_weights)
    dst_weights = _normalize_weights(dst_weights)
    na, nb = cost_matrix.shape

    c = np.asarray(cost_matrix, dtype=float).reshape(-1)
    A_eq = []
    b_eq = []
    for i in range(na):
        row = np.zeros(na * nb, dtype=float)
        row[i * nb:(i + 1) * nb] = 1.0
        A_eq.append(row)
        b_eq.append(src_weights[i])
    for j in range(nb):
        col = np.zeros(na * nb, dtype=float)
        col[j::nb] = 1.0
        A_eq.append(col)
        b_eq.append(dst_weights[j])

    res = linprog(
        c=c,
        A_eq=np.asarray(A_eq, dtype=float),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=(0, None),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Density OT failed: {res.message}")

    pi = res.x.reshape(na, nb)
    pi[np.abs(pi) < 1e-12] = 0.0
    pi = np.maximum(pi, 0.0)
    s = float(np.sum(pi))
    if s > 1e-12:
        pi /= s
    return pi


def _w2_cost_matrix(means_a, covs_a, means_b, covs_b) -> np.ndarray:
    na, nb = len(means_a), len(means_b)
    C = np.zeros((na, nb), dtype=float)
    for i in range(na):
        mu0 = np.asarray(means_a[i], dtype=float)
        sigma0 = _symmetrize(np.asarray(covs_a[i], dtype=float))
        s0 = _to_real_matrix(sqrtm(sigma0))
        for j in range(nb):
            mu1 = np.asarray(means_b[j], dtype=float)
            sigma1 = _symmetrize(np.asarray(covs_b[j], dtype=float))
            mean_term = float(np.sum((mu0 - mu1) ** 2))
            middle = _symmetrize(s0 @ sigma1 @ s0)
            middle_sqrt = _to_real_matrix(sqrtm(middle))
            cov_term = float(np.trace(sigma0 + sigma1 - 2.0 * middle_sqrt))
            C[i, j] = max(0.0, mean_term + cov_term)
    return C


class DensityController3D:
    """Wasserstein density-control based micro trajectory stepper."""

    def __init__(
        self,
        seed: int = 0,
        traj_steps: int = 8,
        esdf_map=None,
        safe_margin: float = 0.1,
        use_astar_path: bool = True,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._component_idx = None
        self._traj_steps = max(1, int(traj_steps))
        self._esdf = esdf_map
        self._safe_margin = max(0.0, float(safe_margin))
        self._use_astar_path = bool(use_astar_path and (esdf_map is not None))
        self._bounds_min = None
        self._bounds_max = None
        if esdf_map is not None:
            try:
                origin = np.asarray(esdf_map.origin, dtype=float).reshape(3)
                dims = np.asarray(esdf_map.dims, dtype=float).reshape(3)
                res = float(esdf_map.resolution)
                self._bounds_min = origin
                self._bounds_max = origin + dims * res
            except Exception:
                self._bounds_min = None
                self._bounds_max = None

    def reset(self) -> None:
        self._component_idx = None

    def _map_resolution(self) -> float:
        if self._esdf is None:
            return 0.1
        try:
            return max(float(getattr(self._esdf, "resolution", 0.1)), 1e-3)
        except Exception:
            return 0.1

    def _clip_bounds(self, pos: np.ndarray) -> np.ndarray:
        p = np.asarray(pos, dtype=float).reshape(-1, 3)
        if self._bounds_min is None or self._bounds_max is None:
            return p
        return np.minimum(np.maximum(p, self._bounds_min), self._bounds_max)

    def _pos_to_index(self, pos: np.ndarray):
        if self._esdf is None:
            return None
        try:
            if hasattr(self._esdf, "pos_to_index"):
                return tuple(self._esdf.pos_to_index(pos).astype(int).tolist())
            origin = np.asarray(getattr(self._esdf, "origin"), dtype=float).reshape(3)
            dims = np.asarray(getattr(self._esdf, "dims"), dtype=int).reshape(3)
            res = self._map_resolution()
            idx = np.floor((np.asarray(pos, dtype=float).reshape(3) - origin) / res).astype(int)
            idx = np.minimum(np.maximum(idx, 0), dims - 1)
            return tuple(idx.tolist())
        except Exception:
            return None

    def _index_to_pos(self, idx):
        if self._esdf is None:
            return None
        try:
            if hasattr(self._esdf, "index_to_pos"):
                return np.asarray(self._esdf.index_to_pos(np.asarray(idx, dtype=int)), dtype=float).reshape(3)
            origin = np.asarray(getattr(self._esdf, "origin"), dtype=float).reshape(3)
            res = self._map_resolution()
            idx_arr = np.asarray(idx, dtype=float).reshape(3)
            return origin + (idx_arr + 0.5) * res
        except Exception:
            return None

    def _is_index_free(self, idx) -> bool:
        if self._esdf is None:
            return True
        p = self._index_to_pos(idx)
        if p is None:
            return False
        try:
            return float(self._esdf.get_esdf(p)) >= self._safe_margin
        except Exception:
            return False

    def _nearest_free_index(self, idx, max_radius: int = 4):
        if self._esdf is None:
            return idx
        try:
            dims = np.asarray(self._esdf.dims, dtype=int)
            idx = tuple(np.asarray(idx, dtype=int).tolist())
            if self._is_index_free(idx):
                return idx
            for r in range(1, int(max_radius) + 1):
                best = None
                best_score = np.inf
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        for dz in range(-r, r + 1):
                            if max(abs(dx), abs(dy), abs(dz)) != r:
                                continue
                            cand = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                            if not (
                                0 <= cand[0] < dims[0]
                                and 0 <= cand[1] < dims[1]
                                and 0 <= cand[2] < dims[2]
                            ):
                                continue
                            if not self._is_index_free(cand):
                                continue
                            score = float(dx * dx + dy * dy + dz * dz)
                            if score < best_score:
                                best_score = score
                                best = cand
                if best is not None:
                    return best
        except Exception:
            return idx
        return idx

    def _fallback_path(self, start: np.ndarray, goal: np.ndarray):
        if self._esdf is None:
            return self._resample_polyline([start, goal], self._traj_steps)
        try:
            if (self._esdf.get_esdf(goal) >= self._safe_margin) and (
                not self._esdf.is_collision_line_segment(start, goal, safe_margin=self._safe_margin)
            ):
                return self._resample_polyline([start, goal], self._traj_steps)
        except Exception:
            pass
        return self._resample_polyline([start, start], self._traj_steps)

    def _resample_polyline(self, points, min_samples: int):
        pts = [np.asarray(p, dtype=float).reshape(3) for p in points]
        if len(pts) == 0:
            return []
        if len(pts) == 1:
            return [pts[0].copy() for _ in range(max(1, int(min_samples)))]

        seg_lens = []
        for i in range(1, len(pts)):
            seg_lens.append(float(np.linalg.norm(pts[i] - pts[i - 1])))
        total_len = float(np.sum(seg_lens))
        if total_len <= 1e-9:
            return [pts[-1].copy() for _ in range(max(1, int(min_samples)))]

        res = float(getattr(self._esdf, "resolution", 0.1)) if self._esdf is not None else 0.1
        spatial_samples = int(np.ceil(total_len / max(0.5 * res, 1e-3)))
        n_out = max(int(min_samples), spatial_samples, 2)
        targets = np.linspace(0.0, total_len, n_out)[1:]

        out = []
        cursor = 0
        seg_start = pts[0]
        seg_end = pts[1]
        seg_acc = 0.0
        cur_seg_len = seg_lens[0]
        for d in targets:
            while (cursor < len(seg_lens) - 1) and (d > seg_acc + cur_seg_len):
                seg_acc += cur_seg_len
                cursor += 1
                seg_start = pts[cursor]
                seg_end = pts[cursor + 1]
                cur_seg_len = seg_lens[cursor]
            alpha = (d - seg_acc) / max(cur_seg_len, 1e-9)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            out.append(seg_start + alpha * (seg_end - seg_start))
        return out

    def _astar_path_one(self, start_pos: np.ndarray, goal_pos: np.ndarray):
        if self._esdf is None:
            return [np.asarray(goal_pos, dtype=float)]
        if not self._use_astar_path:
            return [np.asarray(goal_pos, dtype=float)]

        start = np.asarray(start_pos, dtype=float).reshape(3)
        goal = np.asarray(goal_pos, dtype=float).reshape(3)
        if self._bounds_min is not None and self._bounds_max is not None:
            start = np.minimum(np.maximum(start, self._bounds_min), self._bounds_max)
            goal = np.minimum(np.maximum(goal, self._bounds_min), self._bounds_max)

        try:
            if (self._esdf.get_esdf(goal) >= self._safe_margin) and (
                not self._esdf.is_collision_line_segment(start, goal, safe_margin=self._safe_margin)
            ):
                return self._resample_polyline([start, goal], self._traj_steps)
        except Exception:
            return self._fallback_path(start, goal)

        try:
            start_idx = self._pos_to_index(start)
            goal_idx = self._pos_to_index(goal)
            if start_idx is None or goal_idx is None:
                return self._fallback_path(start, goal)

            start_idx = self._nearest_free_index(start_idx, max_radius=2)
            goal_idx = self._nearest_free_index(goal_idx, max_radius=6)
            if start_idx == goal_idx:
                end_pos = self._index_to_pos(goal_idx)
                if end_pos is None:
                    end_pos = goal
                return self._resample_polyline([start, end_pos], self._traj_steps)

            dims = np.asarray(self._esdf.dims, dtype=int)

            def in_bounds(idx):
                return (0 <= idx[0] < dims[0]) and (0 <= idx[1] < dims[1]) and (0 <= idx[2] < dims[2])

            def traversable(idx):
                return self._is_index_free(idx)

            def heuristic(a, b):
                d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
                return float(np.linalg.norm(d))

            nbr_offsets = [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                if not (dx == 0 and dy == 0 and dz == 0)
            ]
            open_heap = []
            g_score = {start_idx: 0.0}
            came_from = {}
            heapq.heappush(open_heap, (heuristic(start_idx, goal_idx), 0.0, start_idx))
            closed = set()
            max_expand = 200000
            expanded = 0
            found = False
            while open_heap and expanded < max_expand:
                _, g_cur, cur = heapq.heappop(open_heap)
                if cur in closed:
                    continue
                closed.add(cur)
                expanded += 1
                if cur == goal_idx:
                    found = True
                    break

                for off in nbr_offsets:
                    nxt = (cur[0] + off[0], cur[1] + off[1], cur[2] + off[2])
                    if (not in_bounds(nxt)) or (nxt in closed) or (not traversable(nxt)):
                        continue
                    step_cost = float(np.linalg.norm(np.asarray(off, dtype=float)))
                    tentative_g = g_cur + step_cost
                    if tentative_g < g_score.get(nxt, np.inf):
                        g_score[nxt] = tentative_g
                        came_from[nxt] = cur
                        f = tentative_g + heuristic(nxt, goal_idx)
                        heapq.heappush(open_heap, (f, tentative_g, nxt))

            if not found:
                return self._fallback_path(start, goal)

            path_idx = [goal_idx]
            cur = goal_idx
            while cur in came_from:
                cur = came_from[cur]
                path_idx.append(cur)
            path_idx.reverse()

            path = [self._index_to_pos(idx) for idx in path_idx[1:]]
            path = [p for p in path if p is not None]
            if len(path) == 0:
                goal_proj = self._index_to_pos(goal_idx)
                path = [goal_proj if goal_proj is not None else goal]
            else:
                last = np.asarray(path[-1], dtype=float)
                res = self._map_resolution()
                if np.linalg.norm(last - goal) > (0.5 * res):
                    try:
                        if (self._esdf.get_esdf(goal) >= self._safe_margin) and (
                            not self._esdf.is_collision_line_segment(last, goal, safe_margin=self._safe_margin)
                        ):
                            path.append(goal)
                    except Exception:
                        pass
                else:
                    path[-1] = goal
            full_path = [start] + [np.asarray(p, dtype=float) for p in path]
            return self._resample_polyline(full_path, self._traj_steps)
        except Exception:
            return self._fallback_path(start, goal)

    def _estimate_component_responsibilities(self, positions, means, covs, weights):
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        n_agents = positions.shape[0]
        n_comp = len(weights)
        if n_comp <= 0:
            return np.zeros((n_agents, 0), dtype=float)

        probs = np.zeros((n_agents, n_comp), dtype=float)
        ws = _normalize_weights(np.asarray(weights, dtype=float))
        means_arr = np.asarray(means, dtype=float)
        for k in range(n_comp):
            mu_k = np.asarray(means_arr[k], dtype=float)
            sigma_k = np.asarray(covs[k], dtype=float)
            try:
                pdf_vals = multivariate_normal(mean=mu_k, cov=sigma_k, allow_singular=True).pdf(positions)
            except Exception:
                pdf_vals = np.zeros(n_agents, dtype=float)
            probs[:, k] = ws[k] * np.asarray(pdf_vals, dtype=float)

        for i in range(n_agents):
            p = np.nan_to_num(probs[i], nan=0.0, posinf=0.0, neginf=0.0)
            s = float(np.sum(p))
            if s <= 1e-12:
                nearest = int(np.argmin(np.linalg.norm(means_arr - positions[i], axis=1)))
                p = np.zeros(n_comp, dtype=float)
                p[nearest] = 1.0
            else:
                p = p / s
            probs[i] = p
        return probs

    def _target_agent_counts(self, weights: np.ndarray, n_agents: int) -> np.ndarray:
        w = _normalize_weights(weights)
        raw = w * float(n_agents)
        base = np.floor(raw).astype(int)
        remainder = int(n_agents - int(np.sum(base)))
        if remainder > 0:
            frac = raw - base
            order = np.argsort(-frac)
            for i in order[:remainder]:
                base[i] += 1
        return base

    def _assign_agents_to_targets(self, target_score, target_counts, target_means, positions):
        n_agents, nb = target_score.shape
        next_idx = np.full(n_agents, -1, dtype=int)
        if nb == 0:
            return next_idx

        score = np.nan_to_num(target_score, nan=0.0, posinf=0.0, neginf=0.0)
        score[score < 0.0] = 0.0
        pos = np.asarray(positions, dtype=float).reshape(n_agents, 3)
        means = np.asarray(target_means, dtype=float).reshape(nb, 3)
        dist = np.linalg.norm(pos[:, None, :] - means[None, :, :], axis=2)
        dist_scale = np.maximum(np.median(dist), 1e-6)
        utility = np.log(score + 1e-12) - 0.10 * (dist / dist_scale)

        unassigned = set(range(n_agents))
        for l in range(nb):
            c = int(max(0, target_counts[l]))
            if c <= 0:
                continue
            order = np.argsort(-utility[:, l])
            taken = 0
            for i in order:
                if i in unassigned:
                    next_idx[i] = l
                    unassigned.remove(i)
                    taken += 1
                    if taken >= c:
                        break

        if len(unassigned) > 0:
            remaining = np.array(sorted(list(unassigned)), dtype=int)
            fallback = np.argmax(utility[remaining], axis=1)
            next_idx[remaining] = fallback

        return next_idx

    def _pack_full_swarm_trajectory(self, start: np.ndarray, end: np.ndarray):
        n_agents = start.shape[0]
        robot_paths = []
        max_len = 1
        for i in range(n_agents):
            if self._use_astar_path:
                p = self._astar_path_one(start[i], end[i])
            else:
                p = [end[i]]
            if len(p) == 0:
                p = [end[i]]
            path = [np.asarray(q, dtype=float).reshape(3) for q in p]
            robot_paths.append(path)
            max_len = max(max_len, len(path))

        swarm_traj = []
        for t in range(max_len):
            frame = np.zeros((n_agents, 3), dtype=float)
            for i in range(n_agents):
                idx = min(t, len(robot_paths[i]) - 1)
                frame[i] = robot_paths[i][idx]
            swarm_traj.append(self._clip_bounds(frame))
        return swarm_traj

    def make_trajectory(self, start_pos, end_pos):
        start = np.asarray(start_pos, dtype=float).reshape(-1, 3)
        end = np.asarray(end_pos, dtype=float).reshape(-1, 3)
        if self._use_astar_path and self._esdf is not None:
            return self._pack_full_swarm_trajectory(start, end)
        steps = max(1, int(self._traj_steps))
        alphas = np.linspace(0.0, 1.0, steps + 1)[1:]
        return [self._clip_bounds(start + a * (end - start)) for a in alphas]

    def _assign_initial_components(self, positions, means, covs, weights):
        resp = self._estimate_component_responsibilities(positions, means, covs, weights)
        if resp.shape[1] == 0:
            self._component_idx = np.zeros(positions.shape[0], dtype=int)
            return
        self._component_idx = np.argmax(resp, axis=1).astype(int)

    def step(
        self,
        current_means,
        current_covs,
        current_weights,
        target_means,
        target_covs,
        target_weights,
        robots_positions,
    ):
        x = np.asarray(robots_positions, dtype=float).reshape(-1, 3)
        n_agents = x.shape[0]
        if n_agents == 0:
            return x.copy(), []
        if len(target_means) == 0:
            return x.copy(), [x.copy()]
        if len(current_means) == 0:
            current_means = target_means
            current_covs = target_covs
            current_weights = target_weights

        responsibilities = self._estimate_component_responsibilities(
            x, current_means, current_covs, current_weights
        )
        if responsibilities.shape[1] == 0:
            return x.copy(), [x.copy()]
        self._component_idx = np.argmax(responsibilities, axis=1).astype(int)

        src_w = _normalize_weights(np.asarray(current_weights, dtype=float))
        dst_w = _normalize_weights(np.asarray(target_weights, dtype=float))
        cost = _w2_cost_matrix(current_means, current_covs, target_means, target_covs)
        pi = _solve_ot_plan(src_w, dst_w, cost)

        na, nb = pi.shape
        cond = np.zeros_like(pi)
        for k in range(na):
            if src_w[k] > 1e-12:
                cond[k, :] = pi[k, :] / src_w[k]
            else:
                cond[k, :] = np.ones(nb, dtype=float) / max(nb, 1)
        target_score = responsibilities @ cond
        target_counts = self._target_agent_counts(dst_w, n_agents)
        next_idx = self._assign_agents_to_targets(target_score, target_counts, target_means, x)

        next_pos = np.zeros_like(x)
        coeff_A = {}
        for k in range(na):
            sigma_a = _to_real_matrix(_symmetrize(np.asarray(current_covs[k], dtype=float)))
            sigma_a += 1e-6 * np.eye(3, dtype=float)
            S0 = _to_real_matrix(sqrtm(sigma_a))
            S0_inv = np.linalg.pinv(S0)
            for l in range(nb):
                sigma_b = _to_real_matrix(_symmetrize(np.asarray(target_covs[l], dtype=float)))
                sigma_b += 1e-6 * np.eye(3, dtype=float)
                try:
                    sigma_trans = _to_real_matrix(sqrtm(_symmetrize(S0 @ sigma_b @ S0)))
                    coeff_A[(k, l)] = _to_real_matrix(S0_inv @ sigma_trans @ S0_inv)
                except Exception:
                    coeff_A[(k, l)] = np.eye(3, dtype=float)

        for i in range(n_agents):
            k = int(np.clip(self._component_idx[i], 0, na - 1))
            l = int(np.clip(next_idx[i], 0, nb - 1))
            mu_a = np.asarray(current_means[k], dtype=float)
            mu_b = np.asarray(target_means[l], dtype=float)
            A_kl = coeff_A[(k, l)]
            next_pos[i] = mu_b + A_kl.dot(x[i] - mu_a)

        next_pos = self._clip_bounds(next_pos)
        self._component_idx = next_idx
        return next_pos, self.make_trajectory(x, next_pos)
