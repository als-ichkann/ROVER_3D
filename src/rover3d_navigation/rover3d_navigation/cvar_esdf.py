import numpy as np
import scipy
from scipy.optimize import fsolve


def CVaR_for_single_node_esdf(gj, covj, esdf_map, alpha):
    """
    基于 ESDF 地图计算单个高斯节点的 VaR 和 CVaR

    输入：
        gj : array_like, shape (3,)
            高斯节点均值
        covj : array_like, shape (3,3)
            高斯节点协方差矩阵
        esdf_map : object
            已知环境的 ESDF 地图，需提供：
                - esdf_map.get_esdf(x)  # 返回点 x 的 signed distance
                - esdf_map.compute_gradient(x)  # 返回点 x 的梯度向量
        alpha : float
            CVaR 置信水平

    输出：
        mean_proj : float
            投影后的均值（沿梯度方向）
        sigma_proj : float
            投影后的方差
        VaR : float
            投影后的 VaR
        CVaR : float
            投影后的 CVaR
    """
    mu = -float(esdf_map.get_esdf(gj))
    grad = esdf_map.compute_gradient(gj)
    if grad is None:
        return mu, 0.0, mu, mu
    grad = np.asarray(grad, dtype=float).flatten()[:3]
    gn = np.linalg.norm(grad)
    if gn < 1e-12:
        return mu, 0.0, mu, mu
    grad = grad / gn

    covj = np.asarray(covj, dtype=float).reshape(3, 3)
    sigma_proj = float(grad @ covj @ grad)
    mean_proj = mu

    def percentile_of_point(cdf_func, p, mu_p, sigma):
        def func(x):
            return cdf_func(x, mu_p, sigma) - p

        result = fsolve(func, mu_p)
        return float(result[0])

    if sigma_proj <= 0.0:
        return mean_proj, sigma_proj, mean_proj, mean_proj

    cdf_func = scipy.stats.norm.cdf
    VaR = percentile_of_point(cdf_func, 1 - alpha, mean_proj, np.sqrt(sigma_proj))
    pdf_value = scipy.stats.norm.pdf(VaR, mean_proj, np.sqrt(sigma_proj))
    CVaR = mean_proj + sigma_proj * pdf_value / alpha

    return mean_proj, sigma_proj, VaR, CVaR
