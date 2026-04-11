import numpy as np
import scipy
from scipy.stats import norm
from qpsolvers import solve_qp

def _normalize_weights(weights, eps=1e-12):
    weights = np.asarray(weights, dtype=float).reshape(-1)
    s = np.sum(weights)
    if s <= eps:
        return np.ones_like(weights) / len(weights)
    return weights / s

def _safe_gradient(esdf_map, pos):
    grad = esdf_map.compute_gradient(pos)
    if grad is None:
        return np.zeros(3, dtype=float)
    grad = np.asarray(grad, dtype=float).reshape(-1)
    if grad.shape[0] < 3:
        out = np.zeros(3, dtype=float)
        out[:grad.shape[0]] = grad
        return out
    return grad[:3]


def _safe_esdf(esdf_map, pos, default_dist=5.0):
    try:
        val = esdf_map.get_esdf(pos)
        return float(val)
    except Exception:
        return float(default_dist)


# ESDF-CVaR 计算
# ============================================================

def compute_cvar_from_esdf(
    current_means,
    current_covs,
    current_weights,
    esdf_map,
    alpha,
    tol=1e-12,
    max_iter=100,
    min_var=1e-10,
):
    """
    基于整张 3D ESDF 地图计算当前 GMM 的 CVaR
    兼容接口:
        esdf_map.get_esdf(pos)
        esdf_map.compute_gradient(pos)

    参数
    ----
    current_means : (N, 3)
    current_covs : (N, 3, 3)
    current_weights : (N,)
    esdf_map : ESDF adapter
    alpha : float
    """
    current_means = np.asarray(current_means, dtype=float)
    current_covs = np.asarray(current_covs, dtype=float)
    weights = _normalize_weights(current_weights)

    if current_means.ndim != 2 or current_means.shape[1] != 3:
        raise ValueError("current_means must have shape (N, 3)")
    if current_covs.ndim != 3 or current_covs.shape[1:] != (3, 3):
        raise ValueError("current_covs must have shape (N, 3, 3)")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    n_comp = len(weights)

    # 单分量风险高斯参数
    # Y_j = -phi(X_j) ~ N(Smu_j, Ssig_j)
    Smu = np.zeros(n_comp)
    Ssig = np.zeros(n_comp)

    for j in range(n_comp):
        pos = np.asarray(current_means[j], dtype=float).reshape(3,)
        phi_j = _safe_esdf(esdf_map, pos)
        grad_j = _safe_gradient(esdf_map, pos).reshape(3, 1)

        Smu[j] = -phi_j
        Ssig[j] = float(grad_j.T @ current_covs[j] @ grad_j)

    Ssig = np.maximum(Ssig, min_var)
    Sstd = np.sqrt(Ssig)

    # 用单分量 VaR 作为 GMM VaR 的二分上下界
    z_alpha = norm.ppf(1.0 - alpha)
    v = Smu + Sstd * z_alpha
    vmin = np.min(v)
    vmax = np.max(v)

    if np.isclose(vmin, vmax):
        pad = max(1e-6, 1e-3 * (abs(vmin) + 1.0))
        vmin -= pad
        vmax += pad

    # 二分法求 GMM 的 VaR
    VaR = 0.5 * (vmin + vmax)
    for _ in range(max_iter):
        VaR = 0.5 * (vmin + vmax)
        alphaj_tmp = 1.0 - norm.cdf(VaR, loc=Smu, scale=Sstd)
        alpha0 = float(weights @ alphaj_tmp)
        diff = alpha - alpha0

        if abs(diff) < tol:
            break

        # alpha0 > alpha 说明阈值太小
        if diff < 0:
            vmin = VaR
        else:
            vmax = VaR

    # 最终 alphaj
    alphaj = 1.0 - norm.cdf(VaR, loc=Smu, scale=Sstd)
    alphaj = np.clip(alphaj, 1e-12, 1.0 - 1e-12)

    # GMM 的 CVaR
    pdf_value = norm.pdf(norm.ppf(1.0 - alphaj))
    gra = (alphaj * Smu + pdf_value * Sstd) / alpha
    CVaR_curr = float(weights @ gra)

    return CVaR_curr


def linearize_cvar_constraint_fd(
    means,
    covs,
    weights0,
    esdf_map,
    alpha,
    fd_eps=1e-5,
):
    """
    在当前权重 weights0 处，对 CVaR(weights) 做有限差分线性化:
        CVaR(w) ≈ c0 + grad^T (w - w0)
    """
    weights0 = _normalize_weights(weights0)
    c0 = compute_cvar_from_esdf(means, covs, weights0, esdf_map, alpha)

    n = len(weights0)
    grad = np.zeros(n)

    for i in range(n):
        w1 = weights0.copy()
        w1[i] += fd_eps
        w1 = _normalize_weights(w1)
        c1 = compute_cvar_from_esdf(means, covs, w1, esdf_map, alpha)

        denom = w1[i] - weights0[i]
        if abs(denom) < 1e-12:
            grad[i] = 0.0
        else:
            grad[i] = (c1 - c0) / denom

    return c0, grad

