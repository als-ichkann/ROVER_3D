"""
Microscopic Density Controller (C-ROVER style).

Pipeline
--------
1. Soft responsibilities + 2-Wasserstein optimal transport (``linprog``/HiGHS).
2. Quota-constrained greedy task assignment to target GMM components.
3. McCann affine pushforward per agent ``(k -> l)``.
4. ESDF-aware 3D A* micro trajectories with graceful fallbacks.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import LinAlgError, sqrtm
from scipy.optimize import linprog
from scipy.stats import multivariate_normal

try:
    from . import esdf_astar
except ImportError:
    import esdf_astar  # type: ignore

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
GMMMeans = Union[np.ndarray, Sequence[Sequence[float]]]
GMMCovs = Union[np.ndarray, Sequence[np.ndarray]]
GMMWeights = Union[np.ndarray, Sequence[float]]
EsdfMap = Any
PathSnapshots = List[np.ndarray]

_REG_COV = 1e-6
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------
def _as_float_array(x: Union[np.ndarray, Sequence[float]], shape: Tuple[int, ...]) -> np.ndarray:
    """Coerce input to ``float64`` array with expected shape."""
    arr = np.asarray(x, dtype=float)
    return arr.reshape(shape)


def _to_real_matrix(mat: np.ndarray) -> np.ndarray:
    """Drop negligible imaginary parts from ``sqrtm`` results."""
    return np.asarray(np.real_if_close(mat), dtype=float)


def _symmetrize(mat: np.ndarray) -> np.ndarray:
    """Symmetrize a square matrix."""
    return 0.5 * (mat + mat.T)


def _normalize_weights(weights: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Normalize weights to sum to one; uniform fallback if degenerate."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w[w < 0.0] = 0.0
    s = float(np.sum(w))
    if s <= eps:
        return np.ones_like(w) / max(len(w), 1)
    return w / s


def _regularize_cov(cov: np.ndarray, reg: float = _REG_COV) -> np.ndarray:
    """Symmetrize covariance and add jitter for positive-definiteness."""
    c = _symmetrize(_as_float_array(cov, (3, 3)))
    return c + float(reg) * np.eye(3, dtype=float)


def _stable_matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    """Matrix square root with identity fallback on failure."""
    try:
        return _to_real_matrix(sqrtm(_symmetrize(mat)))
    except (LinAlgError, ValueError):
        return np.eye(mat.shape[0], dtype=float)


def _stable_matrix_sqrt_inv(mat: np.ndarray) -> np.ndarray:
    """Stable ``sqrtm`` followed by pseudoinverse."""
    s = _stable_matrix_sqrt(mat)
    return np.linalg.pinv(s)


def _stack_gmm(
    means: GMMMeans,
    covs: GMMCovs,
    weights: GMMWeights,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack GMM parameters to ``(K,3)``, ``(K,3,3)``, ``(K,)`` arrays."""
    mu = _as_float_array(means, (-1, 3))
    w = _normalize_weights(np.asarray(weights, dtype=float))
    k = mu.shape[0]
    if k == 0:
        return mu, np.zeros((0, 3, 3), dtype=float), w
    sigma = np.zeros((k, 3, 3), dtype=float)
    cov_list = list(covs)
    for i in range(k):
        sigma[i] = _regularize_cov(cov_list[i] if i < len(cov_list) else np.eye(3))
    if w.shape[0] != k:
        w = _normalize_weights(np.ones(k, dtype=float))
    return mu, sigma, w


# ---------------------------------------------------------------------------
# Step 1 — Responsibilities & Optimal Transport
# ---------------------------------------------------------------------------
def compute_responsibilities(
    positions: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute soft component responsibilities ``resp[i,k] ∝ w_k N(x_i|μ_k,Σ_k)``.

    Args:
        positions: Agent positions, shape ``(N, 3)``.
        means: Component means, shape ``(K, 3)``.
        covs: Component covariances, shape ``(K, 3, 3)``.
        weights: Mixture weights, shape ``(K,)``.

    Returns:
        Row-stochastic responsibility matrix, shape ``(N, K)``.
    """
    pos = _as_float_array(positions, (-1, 3))
    mu = _as_float_array(means, (-1, 3))
    sigma = np.asarray(covs, dtype=float)
    w = _normalize_weights(weights)
    n_agents, n_comp = pos.shape[0], mu.shape[0]
    if n_comp == 0:
        return np.zeros((n_agents, 0), dtype=float)

    unnorm = np.zeros((n_agents, n_comp), dtype=float)
    log_w = np.log(np.maximum(w, _EPS))
    for k in range(n_comp):
        try:
            rv = multivariate_normal(mean=mu[k], cov=sigma[k], allow_singular=True)
            unnorm[:, k] = np.exp(log_w[k]) * rv.pdf(pos)
        except (LinAlgError, ValueError):
            unnorm[:, k] = 0.0

    row_sum = unnorm.sum(axis=1, keepdims=True)
    valid = row_sum.ravel() > _EPS
    resp = np.zeros_like(unnorm)
    resp[valid] = unnorm[valid] / row_sum[valid]

    if np.any(~valid):
        dists = np.linalg.norm(pos[~valid, None, :] - mu[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        resp[~valid, nearest] = 1.0
    return resp


def wasserstein2_cost_matrix(
    means_a: np.ndarray,
    covs_a: np.ndarray,
    means_b: np.ndarray,
    covs_b: np.ndarray,
) -> np.ndarray:
    """Build pairwise squared 2-Wasserstein costs between Gaussian components.

    Returns:
        Cost matrix ``C`` with shape ``(K_a, K_b)``.
    """
    na, nb = means_a.shape[0], means_b.shape[0]
    cost = np.zeros((na, nb), dtype=float)
    for i in range(na):
        s0 = _stable_matrix_sqrt(_regularize_cov(covs_a[i]))
        for j in range(nb):
            mu_term = float(np.sum((means_a[i] - means_b[j]) ** 2))
            middle = _symmetrize(s0 @ _regularize_cov(covs_b[j]) @ s0)
            cov_term = float(
                np.trace(
                    _regularize_cov(covs_a[i])
                    + _regularize_cov(covs_b[j])
                    - 2.0 * _stable_matrix_sqrt(middle)
                )
            )
            cost[i, j] = max(0.0, mu_term + cov_term)
    return cost


def solve_ot_plan(
    src_weights: np.ndarray,
    dst_weights: np.ndarray,
    cost_matrix: np.ndarray,
) -> np.ndarray:
    """Solve discrete OT with HiGHS; uniform coupling fallback if infeasible.

    Args:
        src_weights: Source marginal, shape ``(K_a,)``.
        dst_weights: Target marginal, shape ``(K_b,)``.
        cost_matrix: Transport costs, shape ``(K_a, K_b)``.

    Returns:
        Transport plan ``π`` with shape ``(K_a, K_b)``, rows/cols sum to marginals.
    """
    src_w = _normalize_weights(src_weights)
    dst_w = _normalize_weights(dst_weights)
    na, nb = cost_matrix.shape
    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=float)

    c = np.asarray(cost_matrix, dtype=float).reshape(-1)
    n_var = na * nb
    a_eq = np.zeros((na + nb, n_var), dtype=float)
    b_eq = np.zeros(na + nb, dtype=float)

    for i in range(na):
        a_eq[i, i * nb : (i + 1) * nb] = 1.0
        b_eq[i] = src_w[i]
    for j in range(nb):
        a_eq[na + j, j::nb] = 1.0
        b_eq[na + j] = dst_w[j]

    res = linprog(
        c=c,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if res.success and res.x is not None:
        pi = np.maximum(res.x.reshape(na, nb), 0.0)
        pi[np.abs(pi) < 1e-12] = 0.0
        s = float(np.sum(pi))
        if s > _EPS:
            return pi / s

    # Fallback: independent coupling proportional to outer product of marginals.
    pi_fb = np.outer(src_w, dst_w)
    return _normalize_weights(pi_fb.ravel()).reshape(na, nb)


# ---------------------------------------------------------------------------
# Step 2 — Quota-constrained greedy assignment
# ---------------------------------------------------------------------------
def exact_agent_counts(weights: np.ndarray, n_agents: int) -> np.ndarray:
    """Largest-remainder apportionment of ``n_agents`` to components."""
    w = _normalize_weights(weights)
    raw = w * float(n_agents)
    base = np.floor(raw).astype(int)
    remainder = int(n_agents - int(np.sum(base)))
    if remainder > 0:
        frac = raw - base
        for idx in np.argsort(-frac)[:remainder]:
            base[idx] += 1
    return base


def greedy_task_assignment(
    target_score: np.ndarray,
    target_counts: np.ndarray,
    target_means: np.ndarray,
    positions: np.ndarray,
    dist_penalty: float = 0.10,
) -> np.ndarray:
    """Assign each agent to exactly one target component with quota limits.

    Utility per agent ``i`` and component ``l``:
        ``log(score[i,l] + ε) - dist_penalty * dist(i,l) / scale``.

    Args:
        target_score: OT-conditional scores, shape ``(N, L)``.
        target_counts: Integer quotas per target component, shape ``(L,)``.
        target_means: Target component means, shape ``(L, 3)``.
        positions: Agent positions, shape ``(N, 3)``.
        dist_penalty: Distance regularizer weight.

    Returns:
        Assignment indices ``assign[i] = l``, shape ``(N,)``.
    """
    score = np.nan_to_num(target_score, nan=0.0, posinf=0.0, neginf=0.0)
    score = np.maximum(score, 0.0)
    pos = _as_float_array(positions, (-1, 3))
    means = _as_float_array(target_means, (-1, 3))
    n_agents, n_tgt = score.shape
    assign = np.full(n_agents, -1, dtype=int)
    if n_tgt == 0:
        return assign

    dist = np.linalg.norm(pos[:, None, :] - means[None, :, :], axis=2)
    scale = max(float(np.median(dist)), 1e-6)
    utility = np.log(score + _EPS) - float(dist_penalty) * (dist / scale)

    unassigned = set(range(n_agents))
    for l in range(n_tgt):
        quota = int(max(0, target_counts[l]))
        if quota <= 0:
            continue
        order = np.argsort(-utility[:, l])
        taken = 0
        for agent_idx in order:
            if agent_idx not in unassigned:
                continue
            assign[agent_idx] = l
            unassigned.remove(int(agent_idx))
            taken += 1
            if taken >= quota:
                break

    if unassigned:
        rem = np.fromiter(sorted(unassigned), dtype=int)
        assign[rem] = np.argmax(utility[rem], axis=1)
    return assign


# ---------------------------------------------------------------------------
# Step 3 — McCann affine maps
# ---------------------------------------------------------------------------
def compute_mccann_affine_maps(
    covs_a: np.ndarray,
    covs_b: np.ndarray,
) -> np.ndarray:
    """Precompute McCann matrices ``A[k,l]`` for all component pairs.

    Args:
        covs_a: Source covariances, shape ``(K, 3, 3)``.
        covs_b: Target covariances, shape ``(L, 3, 3)``.

    Returns:
        Affine coefficients with shape ``(K, L, 3, 3)``.
    """
    na, nb = covs_a.shape[0], covs_b.shape[0]
    coeff = np.zeros((na, nb, 3, 3), dtype=float)
    for k in range(na):
        sigma_a = _regularize_cov(covs_a[k])
        s0_inv = _stable_matrix_sqrt_inv(sigma_a)
        s0 = _stable_matrix_sqrt(sigma_a)
        for l in range(nb):
            sigma_b = _regularize_cov(covs_b[l])
            try:
                middle = _symmetrize(s0 @ sigma_b @ s0)
                sigma_trans = _stable_matrix_sqrt(middle)
                coeff[k, l] = _to_real_matrix(s0_inv @ sigma_trans @ s0_inv)
            except (LinAlgError, ValueError):
                coeff[k, l] = np.eye(3, dtype=float)
    return coeff


def mccann_pushforward(
    positions: np.ndarray,
    src_component: np.ndarray,
    dst_component: np.ndarray,
    means_a: np.ndarray,
    means_b: np.ndarray,
    affine_maps: np.ndarray,
) -> np.ndarray:
    """Apply per-agent McCann map ``x' = μ_l + A_kl (x - μ_k)`` (vectorized).

    Args:
        positions: Agent positions, shape ``(N, 3)``.
        src_component: Source component index per agent, shape ``(N,)``.
        dst_component: Target component index per agent, shape ``(N,)``.
        means_a: Source means, shape ``(K, 3)``.
        means_b: Target means, shape ``(L, 3)``.
        affine_maps: McCann matrices, shape ``(K, L, 3, 3)``.

    Returns:
        Mapped positions, shape ``(N, 3)``.
    """
    x = _as_float_array(positions, (-1, 3))
    k_idx = np.clip(src_component.astype(int), 0, means_a.shape[0] - 1)
    l_idx = np.clip(dst_component.astype(int), 0, means_b.shape[0] - 1)
    delta = x - means_a[k_idx]
    a_sel = affine_maps[k_idx, l_idx]  # (N, 3, 3)
    mapped = np.einsum("nij,nj->ni", a_sel, delta)
    return means_b[l_idx] + mapped


# ---------------------------------------------------------------------------
# Step 4 — ESDF A* navigator
# ---------------------------------------------------------------------------
class AStarNavigator3D:
    """ESDF-backed 3D A* wrapper with arc-length resampling for MPC."""

    def __init__(
        self,
        esdf_map: Optional[EsdfMap],
        safe_margin: float,
        robot_radius: float,
        max_iterations: int = 80_000,
        max_step_size: float = 0.08,
        traj_steps: int = 12,
        use_astar: bool = True,
    ) -> None:
        self._esdf = esdf_map
        self._extra_margin = max(0.0, float(safe_margin))
        self._robot_radius = max(0.05, float(robot_radius))
        self._max_iterations = max(1000, int(max_iterations))
        self._max_step_size = max(1e-3, float(max_step_size))
        self._traj_steps = max(1, int(traj_steps))
        self._use_astar = bool(use_astar)
        self._bounds_min: Optional[np.ndarray] = None
        self._bounds_max: Optional[np.ndarray] = None
        self._init_bounds()

    def _init_bounds(self) -> None:
        if self._esdf is None:
            return
        try:
            origin = np.asarray(self._esdf.origin, dtype=float).reshape(3)
            dims = np.asarray(self._esdf.dims, dtype=float).reshape(3)
            res = float(self._esdf.resolution)
            self._bounds_min = origin
            self._bounds_max = origin + dims * res
        except Exception:
            self._bounds_min = None
            self._bounds_max = None

    @property
    def effective_margin(self) -> float:
        """Clearance threshold = configured margin + robot radius."""
        return self._extra_margin + self._robot_radius

    def refresh_esdf(self) -> bool:
        """Refresh ESDF grid if supported."""
        if self._esdf is None:
            return False
        if hasattr(self._esdf, "refresh"):
            return bool(self._esdf.refresh())
        if hasattr(self._esdf, "_ensure_ready"):
            return bool(self._esdf._ensure_ready())
        return True

    @property
    def esdf_ready(self) -> bool:
        if self._esdf is None:
            return False
        if hasattr(self._esdf, "is_ready"):
            return bool(self._esdf.is_ready)
        if hasattr(self._esdf, "_ensure_ready"):
            return bool(self._esdf._ensure_ready())
        return True

    def clip_bounds(self, pos: np.ndarray) -> np.ndarray:
        """Clip positions to ESDF map bounding box."""
        p = _as_float_array(pos, (-1, 3))
        if self._bounds_min is None or self._bounds_max is None:
            return p
        return np.minimum(np.maximum(p, self._bounds_min), self._bounds_max)

    def project_to_free(self, points: np.ndarray) -> np.ndarray:
        """Project waypoints into free space using ESDF gradient ascent."""
        if self._esdf is None:
            return _as_float_array(points, (-1, 3))
        out = _as_float_array(points, (-1, 3)).copy()
        margin = self.effective_margin
        for i in range(out.shape[0]):
            out[i] = esdf_astar.project_to_free(self._esdf, out[i], margin)
        return out

    def plan_single(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Plan one agent path; never raises.

        Priority: A* → straight-line resample → hover at start.
        """
        s = _as_float_array(start, (3,))
        g = self.project_to_free(_as_float_array(goal, (1, 3)))[0]
        margin = self.effective_margin

        if self._use_astar and self._esdf is not None and self.esdf_ready:
            self.refresh_esdf()
            path = esdf_astar.astar_3d(
                self._esdf,
                s,
                g,
                safe_margin=margin,
                max_iterations=self._max_iterations,
            )
            if path is not None and path.shape[0] >= 1:
                resampled = esdf_astar.resample_path(
                    path,
                    step_size=self._max_step_size,
                    max_points=self._traj_steps,
                )
                safe = esdf_astar.sanitize_path(self._esdf, resampled, margin)
                if safe.shape[0] >= 1:
                    return self.clip_bounds(safe)

        # Fallback: direct arc-length resample (may clip through obstacles; stable).
        linear = self._linear_resample(s[None, :], g[None, :])[0]
        return self.clip_bounds(linear)

    def plan_swarm(
        self,
        starts: np.ndarray,
        goals: np.ndarray,
    ) -> Tuple[np.ndarray, PathSnapshots]:
        """Plan per-agent paths and align to MPC snapshot format.

        Args:
            starts: Shape ``(N, 3)``.
            goals: Shape ``(N, 3)``.

        Returns:
            Tuple of terminal positions ``(N, 3)`` and synchronized snapshots.
        """
        starts = _as_float_array(starts, (-1, 3))
        goals = _as_float_array(goals, (-1, 3))
        n = starts.shape[0]
        if n == 0:
            return starts.copy(), []

        agent_paths: List[np.ndarray] = []
        for i in range(n):
            agent_paths.append(self.plan_single(starts[i], goals[i]))

        snapshots = esdf_astar.paths_to_agent_snapshots(agent_paths)
        snapshots = [self.clip_bounds(s) for s in snapshots]
        terminals = agent_paths[-1] if agent_paths else goals
        if agent_paths:
            terminals = np.stack([p[-1] for p in agent_paths], axis=0)
        return terminals, snapshots

    def _linear_resample(self, start: np.ndarray, end: np.ndarray) -> List[np.ndarray]:
        """Uniform straight-line resampling with exactly ``traj_steps`` segments."""
        s = _as_float_array(start, (-1, 3))
        e = _as_float_array(end, (-1, 3))
        disp = np.linalg.norm(e - s, axis=1)
        max_disp = float(np.max(disp)) if disp.size else 0.0
        if max_disp <= 1e-9:
            return [e.copy()]
        steps = min(
            self._traj_steps,
            max(1, int(np.ceil(max_disp / self._max_step_size))),
        )
        alphas = np.linspace(0.0, 1.0, steps + 1)[1:]
        return [self.clip_bounds(s + a * (e - s)) for a in alphas]


# ---------------------------------------------------------------------------
# Microscopic Density Controller
# ---------------------------------------------------------------------------
class DensityController3D:
    """C-ROVER microscopic density controller for one GMM transition ``A → B``."""

    def __init__(
        self,
        seed: int = 0,
        traj_steps: int = 12,
        max_step_size: float = 0.08,
        use_astar: bool = True,
        astar_safe_margin: float = 0.4,
        astar_robot_radius: float = 0.18,
        astar_max_iterations: int = 80_000,
        esdf_map: Optional[EsdfMap] = None,
        dist_penalty: float = 0.10,
    ) -> None:
        """Initialize controller.

        Args:
            seed: RNG seed (reserved for stochastic extensions).
            traj_steps: Maximum micro waypoints per planning cycle.
            max_step_size: Arc-length spacing for path resampling (m).
            use_astar: Enable ESDF A*; otherwise linear resampling only.
            astar_safe_margin: Extra ESDF clearance beyond robot radius (m).
            astar_robot_radius: Equivalent robot radius (m).
            astar_max_iterations: A* expansion budget.
            esdf_map: FIESTA ESDF adapter exposing ``get_esdf`` / optional index API.
            dist_penalty: Greedy assignment distance regularizer.
        """
        self._rng = np.random.default_rng(seed)
        self._component_idx: Optional[np.ndarray] = None
        self._traj_steps = max(1, int(traj_steps))
        self._max_step_size = max(1e-3, float(max_step_size))
        self._dist_penalty = float(dist_penalty)
        self._navigator = AStarNavigator3D(
            esdf_map=esdf_map,
            safe_margin=astar_safe_margin,
            robot_radius=astar_robot_radius,
            max_iterations=astar_max_iterations,
            max_step_size=self._max_step_size,
            traj_steps=self._traj_steps,
            use_astar=use_astar,
        )

    def reset(self) -> None:
        """Reset internal assignment state."""
        self._component_idx = None

    def step(
        self,
        current_means: GMMMeans,
        current_covs: GMMCovs,
        current_weights: GMMWeights,
        target_means: GMMMeans,
        target_covs: GMMCovs,
        target_weights: GMMWeights,
        robots_positions: np.ndarray,
    ) -> Tuple[np.ndarray, PathSnapshots]:
        """Execute one microscopic control cycle.

        Args:
            current_means: Source GMM means (state A).
            current_covs: Source GMM covariances.
            current_weights: Source mixture weights.
            target_means: Target GMM means (state B).
            target_covs: Target GMM covariances.
            target_weights: Target mixture weights.
            robots_positions: Current agent positions, shape ``(N, 3)``.

        Returns:
            ``(terminal_positions, trajectory_snapshots)`` where snapshots is a
            list of ``(N, 3)`` arrays for MPC consumption.
        """
        x = _as_float_array(robots_positions, (-1, 3))
        n_agents = x.shape[0]
        if n_agents == 0:
            return x.copy(), []
        if len(np.atleast_1d(target_means)) == 0:
            return x.copy(), [x.copy()]

        mu_a, sigma_a, w_a = _stack_gmm(current_means, current_covs, current_weights)
        mu_b, sigma_b, w_b = _stack_gmm(target_means, target_covs, target_weights)
        if mu_a.shape[0] == 0:
            mu_a, sigma_a, w_a = mu_b, sigma_b, w_b

        # --- Step 1: responsibilities + OT ---
        resp = compute_responsibilities(x, mu_a, sigma_a, w_a)
        if resp.shape[1] == 0:
            return x.copy(), [x.copy()]

        cost = wasserstein2_cost_matrix(mu_a, sigma_a, mu_b, sigma_b)
        pi = solve_ot_plan(w_a, w_b, cost)

        # --- Step 2: greedy assignment ---
        src_w = _normalize_weights(w_a)
        cond = np.divide(
            pi,
            src_w[:, None],
            out=np.zeros_like(pi),
            where=src_w[:, None] > _EPS,
        )
        row_bad = np.all(src_w <= _EPS)
        if np.any(row_bad):
            cond[row_bad, :] = 1.0 / max(pi.shape[1], 1)
        target_score = resp @ cond
        target_counts = exact_agent_counts(w_b, n_agents)
        src_idx = np.argmax(resp, axis=1).astype(int)
        dst_idx = greedy_task_assignment(
            target_score,
            target_counts,
            mu_b,
            x,
            dist_penalty=self._dist_penalty,
        )

        # --- Step 3: McCann pushforward ---
        affine = compute_mccann_affine_maps(sigma_a, sigma_b)
        next_pos = mccann_pushforward(x, src_idx, dst_idx, mu_a, mu_b, affine)
        next_pos = self._navigator.clip_bounds(next_pos)
        next_pos = self._navigator.project_to_free(next_pos)

        # --- Step 4: ESDF A* micro trajectories ---
        _, snapshots = self._navigator.plan_swarm(x, next_pos)

        self._component_idx = dst_idx.astype(int)
        terminals = snapshots[-1] if snapshots else next_pos
        return terminals, snapshots
