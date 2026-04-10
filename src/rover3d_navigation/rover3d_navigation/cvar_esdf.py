import numpy as np
from scipy.stats import norm


def CVaR_for_single_node_esdf(gj, covj, esdf_map, alpha):
    """
    基于 ESDF：将 3D 高斯沿 ESDF 梯度方向投影为 1D 高斯，再计算 VaR / CVaR（闭式）。

    一维随机变量 X ~ N(mean_proj, sigma^2)，sigma^2 = g^T Σ g（投影方差）：
      VaR_alpha    = mean_proj + sigma * Phi^{-1}(1 - alpha)
      CVaR_alpha   = mean_proj + sigma * phi(z_alpha) / alpha
    其中 z_alpha = Phi^{-1}(1 - alpha)，phi 为标准正态 pdf。

    输入：
        gj : array_like, shape (3,)
            高斯节点均值
        covj : array_like, shape (3,3)
            高斯节点协方差矩阵
        esdf_map : object
            需提供 get_esdf(x)、compute_gradient(x)
        alpha : float
            尾部概率水平（与上式一致，常用较小正数如 0.05）

    输出：
        mean_proj, sigma_proj, VaR, CVaR（sigma_proj 为投影后方差 g^TΣg）
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
    # 理论上 sigma_proj >= 0；数值上夹紧避免负 tiny 与开方问题
    sigma_proj = max(sigma_proj, 1e-12)

    mean_proj = mu
    sigma = float(np.sqrt(sigma_proj))

    z = float(norm.ppf(1.0 - alpha))
    VaR = mean_proj + sigma * z
    pdf_val = float(norm.pdf(z))
    CVaR = mean_proj + sigma * pdf_val / alpha

    return mean_proj, sigma_proj, VaR, CVaR
