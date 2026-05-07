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
    """Wasserstein density-control based micro trajectory stepper."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._component_idx = None

    def reset(self) -> None:
        self._component_idx = None

    def _assign_initial_components(self, positions, means, covs, weights):
        n_agents = positions.shape[0]
        n_comp = len(weights)
        if n_comp == 0:
            self._component_idx = np.zeros(n_agents, dtype=int)
            return

        probs = np.zeros((n_agents, n_comp), dtype=float)
        ws = _normalize_weights(np.asarray(weights, dtype=float))
        for k in range(n_comp):
            mu_k = np.asarray(means[k], dtype=float)
            sigma_k = np.asarray(covs[k], dtype=float)
            try:
                pdf_vals = multivariate_normal(mean=mu_k, cov=sigma_k, allow_singular=True).pdf(positions)
            except Exception:
                pdf_vals = np.zeros(n_agents, dtype=float)
            probs[:, k] = ws[k] * np.asarray(pdf_vals, dtype=float)

        self._component_idx = np.zeros(n_agents, dtype=int)
        for i in range(n_agents):
            p = np.nan_to_num(probs[i], nan=0.0, posinf=0.0, neginf=0.0)
            s = float(np.sum(p))
            if s <= 1e-12:
                self._component_idx[i] = int(np.argmin(np.linalg.norm(np.asarray(means) - positions[i], axis=1)))
            else:
                p = p / s
                self._component_idx[i] = int(self._rng.choice(np.arange(n_comp), p=p))

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

        if self._component_idx is None or len(self._component_idx) != n_agents:
            self._assign_initial_components(x, current_means, current_covs, current_weights)

        src_w = _normalize_weights(np.asarray(current_weights, dtype=float))
        dst_w = _normalize_weights(np.asarray(target_weights, dtype=float))
        cost = _w2_cost_matrix(current_means, current_covs, target_means, target_covs)
        pi = _solve_ot_plan(src_w, dst_w, cost)

        na, nb = pi.shape
        next_idx = np.full(n_agents, -1, dtype=int)
        for k in range(na):
            ids = np.where(self._component_idx == k)[0]
            if len(ids) == 0:
                continue
            if src_w[k] > 1e-12:
                probs = pi[k, :] / src_w[k]
            else:
                probs = np.ones(nb, dtype=float) / nb
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs[probs < 0] = 0.0
            s = float(np.sum(probs))
            probs = probs / s if s > 1e-12 else (np.ones(nb, dtype=float) / nb)
            counts = self._rng.multinomial(len(ids), probs)
            ids_perm = self._rng.permutation(ids)
            cursor = 0
            for l in range(nb):
                c = int(counts[l])
                if c > 0:
                    next_idx[ids_perm[cursor:cursor + c]] = l
                    cursor += c

        unassigned = np.where(next_idx < 0)[0]
        if len(unassigned) > 0:
            next_idx[unassigned] = self._rng.choice(np.arange(nb), size=len(unassigned), p=dst_w)

        next_pos = np.zeros_like(x)
        coeff_A = {}
        for k in range(na):
            sigma_a = _to_real_matrix(_symmetrize(np.asarray(current_covs[k], dtype=float)))
            S0 = _to_real_matrix(sqrtm(sigma_a))
            S0_inv = np.linalg.pinv(S0)
            for l in range(nb):
                sigma_b = _to_real_matrix(_symmetrize(np.asarray(target_covs[l], dtype=float)))
                sigma_trans = _to_real_matrix(sqrtm(_symmetrize(S0 @ sigma_b @ S0)))
                coeff_A[(k, l)] = _to_real_matrix(S0_inv @ sigma_trans @ S0_inv)

        for i in range(n_agents):
            k = int(np.clip(self._component_idx[i], 0, na - 1))
            l = int(np.clip(next_idx[i], 0, nb - 1))
            mu_a = np.asarray(current_means[k], dtype=float)
            mu_b = np.asarray(target_means[l], dtype=float)
            A_kl = coeff_A[(k, l)]
            next_pos[i] = mu_b + A_kl.dot(x[i] - mu_a)

        self._component_idx = next_idx
        return next_pos, [next_pos.copy()]
