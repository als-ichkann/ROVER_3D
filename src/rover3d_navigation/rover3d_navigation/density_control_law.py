import numpy as np
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
    """Wasserstein density-control based micro trajectory stepper.

    每一步用最优传输的高斯仿射映射，为每架无人机算出其在目标 GMM 分布中的
    对应位置 ``next_pos``，再对 ``current -> next_pos`` 做直线插值生成微观轨迹。
    全局绕障由上层宏观规划（GMM + SLP 转移路径）负责，本类不做 A*。
    """

    def __init__(
        self,
        seed: int = 0,
        traj_steps: int = 8,
        esdf_map=None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._component_idx = None
        self._traj_steps = max(1, int(traj_steps))
        self._esdf = esdf_map
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

    def _clip_bounds(self, pos: np.ndarray) -> np.ndarray:
        p = np.asarray(pos, dtype=float).reshape(-1, 3)
        if self._bounds_min is None or self._bounds_max is None:
            return p
        return np.minimum(np.maximum(p, self._bounds_min), self._bounds_max)

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

    def make_trajectory(self, start_pos, end_pos):
        """对 current -> next_pos 做直线插值，生成 traj_steps 个微观轨迹点。"""
        start = np.asarray(start_pos, dtype=float).reshape(-1, 3)
        end = np.asarray(end_pos, dtype=float).reshape(-1, 3)
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
