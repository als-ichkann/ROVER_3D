import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.optimize import linprog
from scipy.stats import norm
try:
    from .cvar_esdf import compute_cvar_from_esdf, linearize_cvar_constraint_fd, _normalize_weights  
    from .gen_path_table import find_mean_index, notgreedy_genPathTable, shortest_path
except ImportError:
    from cvar_esdf import compute_cvar_from_esdf, linearize_cvar_constraint_fd, _normalize_weights
    from gen_path_table import find_mean_index, notgreedy_genPathTable, shortest_path
import math
import pandas as pd
import time
import warnings
from qpsolvers import solve_qp
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")



def _to_pruned_csc(mat: np.ndarray, zero_eps: float = 1e-9) -> scipy.sparse.csc_matrix:
    """将近零元素裁剪后转为 CSC，避免 MOSEK 的 zeros-in-sparse-col 警告刷屏。"""
    arr = np.asarray(mat, dtype=float).copy()
    arr[np.abs(arr) < zero_eps] = 0.0
    sp = scipy.sparse.csc_matrix(arr)
    sp.eliminate_zeros()
    return sp


def build_weight_map(path_table, node_indices, col_id, num_variable):
    """
    构造线性映射矩阵 M，使得:
        node_weights = M @ PI
    col_id = 1 表示 path_table[:,1] 对应的下一层节点
    col_id = 2 表示 path_table[:,2] 对应的 k+2 层节点
    """
    node_indices = np.asarray(node_indices, dtype=int)
    M = np.zeros((len(node_indices), num_variable))
    for r, node_idx in enumerate(node_indices):
        idx = np.where(path_table[:, col_id].astype(int) == int(node_idx))[0]
        if len(idx) > 0:
            M[r, idx] = 1.0
    return M


def initialize_path_flow(path_table, current_weights, fweights):
    """
    给 PIa 一个稳定的初值：
    对每个当前分量 i，把其权重按目标分量权重分摊到从 i 出发的路径上
    """
    num_variable = path_table.shape[0]
    PIa = np.zeros(num_variable)

    current_weights = np.asarray(current_weights, dtype=float).reshape(-1)
    fweights = _normalize_weights(fweights)

    num_src = len(current_weights)

    for i in range(num_src):
        idx = np.where(path_table[:, 0].astype(int) == i)[0]
        if len(idx) == 0:
            continue

        dst_target_idx = path_table[idx, 3].astype(int)
        local_target_weights = np.array([fweights[k] for k in dst_target_idx], dtype=float)
        s = np.sum(local_target_weights)

        if s <= 1e-12:
            PIa[idx] = current_weights[i] / len(idx)
        else:
            PIa[idx] = current_weights[i] * (local_target_weights / s)

    return PIa


def Wasserstein_distance(mean1, cov1, mean2, cov2):
    """Wasserstein-2 distance between two Gaussian components."""
    mean1 = np.asarray(mean1, dtype=float)
    mean2 = np.asarray(mean2, dtype=float)
    cov1 = np.asarray(cov1, dtype=float)
    cov2 = np.asarray(cov2, dtype=float)
    add1 = np.linalg.norm(mean1 - mean2)
    if np.array_equal(cov1, cov2):
        return add1
    add2 = np.trace(
        cov1 + cov2 - 2 * scipy.linalg.sqrtm(scipy.linalg.sqrtm(cov1) @ cov2 @ scipy.linalg.sqrtm(cov1))
    )
    return float((add1 ** 2 + add2) ** 0.5)


def calWGMetric_speedUp(means1, covs1, weights1, means2, covs2, weights2):
    """Solve OT between two GMMs and return (WG_sq, plan, WG)."""
    num_comp_p, num_comp_q = len(means1), len(means2)
    total_mass = min(sum(weights1), sum(weights2))
    weights1 = [w / sum(weights1) * total_mass for w in weights1]
    weights2 = [w / sum(weights2) * total_mass for w in weights2]

    C = np.zeros((num_comp_p, num_comp_q), dtype=float)
    for i in range(num_comp_p):
        for j in range(num_comp_q):
            C[i, j] = Wasserstein_distance(means1[i], covs1[i], means2[j], covs2[j]) ** 2
    f = C.T.flatten()

    Aeq = np.zeros((num_comp_p + num_comp_q, num_comp_p * num_comp_q), dtype=float)
    for i in range(num_comp_q):
        Aeq[i, i * num_comp_p: (i + 1) * num_comp_p] = 1
    for i in range(num_comp_q, num_comp_p + num_comp_q):
        Aeq[i, i - num_comp_q::num_comp_p] = 1
    beq = np.array(weights2 + weights1)

    result = linprog(
        f,
        A_eq=Aeq,
        b_eq=beq,
        bounds=[(0, 1)] * (num_comp_p * num_comp_q),
        method="highs-ds",
    )
    if not result.success:
        raise RuntimeError(f"Linear programming did not succeed: {result.message}")

    W, fval = result.x, float(result.fun)
    return fval, W, float(np.sqrt(fval))


def Optimization_SLP(
    current_means,
    current_covs,
    current_weights,
    fmeans,
    fcovs,
    fweights,
    conbinedmeans_list,
    conbinedcovs_list,
    esdf_map,
    alpha,
    current_goal_means,
    current_goal_covs,
    current_goal_weights,
    Graph_GC,
    Wasserstein_table,
    Node_PDF_table,
    robots_positions,
    epsilon=0.2,
    min_weight=0.002,
    tau=1e-3,
    gamma0=1.0,
    aa=None,
    MaxPDF=1e-1,
    ConvThredhold=0.003,
    max_slp_iter=30,
    fd_eps=1e-5,
    qp_solver="mosek",
):
    """
    说明
    ----
    这版函数假设以下依赖已经在你的工程里存在：
        - calWGMetric_speedUp
        - notgreedy_genPathTable
        - qpsolvers.solve_qp
    """

    print("[0] 开始 Optimization_SLP")

    # --------------------------------------------------------
    # Step 0: 基于当前 global/odom 的集群位置做 EM 拟合
    # --------------------------------------------------------
    robots_positions = np.asarray(robots_positions, dtype=float).reshape(-1, 3)
    if robots_positions.shape[0] == 0:
        raise ValueError("robots_positions 为空，无法进行 STEP0-EM")

    current_means_in = np.asarray(current_means, dtype=float).reshape(-1, 3)
    k_hint = int(current_means_in.shape[0])
    k = max(1, min(k_hint, robots_positions.shape[0]))

    em = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=0,
    )
    em.fit(robots_positions)

    current_means = np.asarray(em.means_, dtype=float)
    current_covs = np.asarray(em.covariances_, dtype=float)
    current_weights = _normalize_weights(em.weights_)
    print(
        f"[0] STEP0-EM done: samples={robots_positions.shape[0]}, "
        f"K={k}, weights={current_weights.tolist()}"
    )

    fmeans = np.asarray(fmeans, dtype=float)
    fcovs = np.asarray(fcovs, dtype=float)
    fweights = _normalize_weights(fweights)

    if aa is None:
        aa = 1.0 / np.e**3

    flag = 0

    print(current_means)
    print(current_covs)
    print(current_weights)
    print(f"[0] target fmeans = {fmeans.tolist() if hasattr(fmeans, 'tolist') else fmeans}")
    print(f"[0] target fweights = {fweights.tolist() if hasattr(fweights, 'tolist') else fweights}")
    print(f"[0] slp epsilon = {epsilon}")

    # --------------------------------------------------------
    # Step 1: 当前 GMM 的 ESDF-CVaR 安全检查
    # --------------------------------------------------------
    try:
        phi_now = np.asarray(esdf_map.get_esdf(current_means), dtype=float).reshape(-1)
        neg_cnt = int(np.sum(phi_now < 0.0))
        print(
            f"[1] signed_sdf stats: min={float(np.min(phi_now)):.6f}, "
            f"max={float(np.max(phi_now)):.6f}, neg_count={neg_cnt}/{len(phi_now)}"
        )
    except Exception as ex:
        print(f"[1] signed_sdf stats unavailable: {ex}")

    CVaR_curr = compute_cvar_from_esdf(
        current_means=current_means,
        current_covs=current_covs,
        current_weights=current_weights,
        esdf_map=esdf_map,
        alpha=alpha,
    )
    print(f"[1] current GMM CVaR = {CVaR_curr:.6f}")

    if CVaR_curr > epsilon:
        print("[1] current_gmm_est is larger than epsilon, fallback to current_goal")
        current_means = np.asarray(current_goal_means, dtype=float)
        current_covs = np.asarray(current_goal_covs, dtype=float)
        current_weights = _normalize_weights(current_goal_weights)

    # --------------------------------------------------------
    # Step 2: Wasserstein 终止判据
    # --------------------------------------------------------
    WG_sq, W, _ = calWGMetric_speedUp(
        current_means, current_covs, current_weights,
        fmeans, fcovs, fweights
    )
    print(f"[2] Wasserstein distance squared: {WG_sq:.4f}")

    if 0 <= WG_sq <= 1:
        Next_means = fmeans
        Next_covs = fcovs
        Next_weights = fweights
        TransferMatrix = W
        flag = 1
        return (
            Next_means, Next_covs, Next_weights,
            current_means, current_covs, current_weights,
            TransferMatrix, flag
        )

    # --------------------------------------------------------
    # Step 3: 过滤低权重分量
    # --------------------------------------------------------
    indexList = np.where(np.asarray(current_weights) >= min_weight)[0]
    if len(indexList) < len(current_means):
        current_means = current_means[indexList]
        current_covs = current_covs[indexList]
        current_weights = _normalize_weights(current_weights[indexList])

    # --------------------------------------------------------
    # Step 4: 路径表
    # --------------------------------------------------------
    print("[3] 路径表开始")
    start = time.time()
    path_table = notgreedy_genPathTable(
        current_means, current_covs, current_weights,
        fmeans, fcovs, fweights,
        conbinedmeans_list, conbinedcovs_list,
        esdf_map, Graph_GC, Wasserstein_table
    )
    print(f"[4] 路径表完成，耗时 {time.time() - start:.2f}s")

    if path_table.shape[0] == 0:
        print("[4] 路径表为空（无可达路径），回退为直达目标")
        _, W, _ = calWGMetric_speedUp(
            current_means, current_covs, current_weights,
            fmeans, fcovs, fweights
        )
        return (
            fmeans, fcovs, fweights,
            current_means, current_covs, current_weights,
            W, 1
        )

    # --------------------------------------------------------
    # Step 5: 路径图节点
    # --------------------------------------------------------
    index_next_gc = np.unique(path_table[:, 1]).astype(int)
    index_k2_gc = np.unique(path_table[:, 2]).astype(int)
    num_next_gc = len(index_next_gc)
    num_k2_gc = len(index_k2_gc)

    next_means = np.array([conbinedmeans_list[idx] for idx in index_next_gc], dtype=float)
    next_covs = np.array([conbinedcovs_list[idx] for idx in index_next_gc], dtype=float)

    k2_means = np.array([conbinedmeans_list[idx] for idx in index_k2_gc], dtype=float)
    k2_covs = np.array([conbinedcovs_list[idx] for idx in index_k2_gc], dtype=float)

    numVariable = path_table.shape[0]
    f = np.asarray(path_table[:, 7], dtype=float)

    # 线性映射：PI -> 节点权重
    M_next = build_weight_map(path_table, index_next_gc, col_id=1, num_variable=numVariable)
    M_k2 = build_weight_map(path_table, index_k2_gc, col_id=2, num_variable=numVariable)

    # --------------------------------------------------------
    # Step 6: SLP 初始化
    # --------------------------------------------------------
    iter_count = 0
    ConvFlag = False
    ConvCounter = 0
    soludiff = []
    cost = []
    gamma = gamma0

    PIa = initialize_path_flow(path_table, current_weights, fweights)
    PIa = np.clip(PIa, 0.0, 1.0)

    print("[5] CVaR-SLP 迭代开始")

    while (not ConvFlag) and (iter_count < max_slp_iter):
        # ----------------------------------------------------
        # 当前迭代点对应的下一层 / k+2 层节点权重
        # ----------------------------------------------------
        Wa = M_next @ PIa
        Wak2 = M_k2 @ PIa

        if np.sum(Wa) > 0:
            Wa = _normalize_weights(Wa)
        if np.sum(Wak2) > 0:
            Wak2 = _normalize_weights(Wak2)

        if np.sum(Wa) > 0:
            macro_idx_local = int(np.argmax(Wa))
            macro_idx_global = int(index_next_gc[macro_idx_local])
            macro_point = np.asarray(next_means[macro_idx_local]).tolist()
            print(
                f"[SLP] step={iter_count}, macro_node={macro_idx_global}, "
                f"macro_point={macro_point}, weight={Wa[macro_idx_local]:.6f}"
            )

        # ----------------------------------------------------
        # 风险约束线性化
        # CVaR(w) ≈ c0 + grad^T (w - w0) <= epsilon
        # 且 w = M @ PI
        # => grad^T M PI <= epsilon - c0 + grad^T w0
        # ----------------------------------------------------
        cvar_next_0, grad_next = linearize_cvar_constraint_fd(
            means=next_means,
            covs=next_covs,
            weights0=Wa,
            esdf_map=esdf_map,
            alpha=alpha,
            fd_eps=fd_eps,
        )
        cvar_k2_0, grad_k2 = linearize_cvar_constraint_fd(
            means=k2_means,
            covs=k2_covs,
            weights0=Wak2,
            esdf_map=esdf_map,
            alpha=alpha,
            fd_eps=fd_eps,
        )

        A_risk_next = grad_next.reshape(1, -1) @ M_next
        b_risk_next = np.array([epsilon - cvar_next_0 + float(grad_next @ Wa)])

        A_risk_k2 = grad_k2.reshape(1, -1) @ M_k2
        b_risk_k2 = np.array([epsilon - cvar_k2_0 + float(grad_k2 @ Wak2)])

        print(
            f"[5] iter={iter_count}, "
            f"CVaR_next={cvar_next_0:.6f}, "
            f"CVaR_k2={cvar_k2_0:.6f}"
        )

        # ----------------------------------------------------
        # PDF 约束
        # ----------------------------------------------------
        res = getattr(esdf_map, "resolution", 1.0)

        A_PDF = np.zeros((num_next_gc, numVariable))
        for n in range(numVariable):
            A_PDF[:, n] = Node_PDF_table[index_next_gc, int(path_table[n, 1])] * (res ** 3)

        A_PDF_k2 = np.zeros((num_k2_gc, numVariable))
        for n in range(numVariable):
            A_PDF_k2[:, n] = Node_PDF_table[index_k2_gc, int(path_table[n, 2])] * (res ** 3)

        # ----------------------------------------------------
        # 不等式约束 Gx <= h
        # ----------------------------------------------------
        A = np.vstack([
            A_PDF,
            A_PDF_k2,
            A_risk_next,
            A_risk_k2,
        ])

        b_PDF1 = np.full((num_next_gc,), MaxPDF)
        b_PDF2 = np.full((num_k2_gc,), MaxPDF)

        b = np.concatenate([
            b_PDF1,
            b_PDF2,
            b_risk_next,
            b_risk_k2,
        ], axis=0)

        # ----------------------------------------------------
        # 等式约束 Aeq x = beq
        # 保质量守恒 + 目标边缘
        # ----------------------------------------------------
        Aeq = []
        beq = []

        # 每个当前分量流出的总量 = 当前权重
        for n in range(len(current_means)):
            idx = np.where(path_table[:, 0].astype(int) == n)[0]
            a = np.zeros(numVariable)
            a[idx] = 1.0
            Aeq.append(a)
            beq.append(current_weights[n])

        # 每个目标分量流入的总量 = 目标权重
        for n in range(len(fmeans)):
            idx = np.where(path_table[:, 3].astype(int) == n)[0]
            a = np.zeros(numVariable)
            a[idx] = 1.0
            Aeq.append(a)
            beq.append(fweights[n])

        Aeq = np.asarray(Aeq, dtype=float)
        beq = np.asarray(beq, dtype=float)

        # ----------------------------------------------------
        # QP: 0.5 x^T H x + q^T x
        # 其中 0.5*tau*||x - PIa||^2 + f^T x
        # => H = tau I, q = f - tau PIa
        # ----------------------------------------------------
        H = tau * np.eye(numVariable)
        q = f - tau * PIa

        # sanitize
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        Aeq = np.nan_to_num(Aeq, nan=0.0, posinf=0.0, neginf=0.0)
        beq = np.nan_to_num(beq, nan=0.0, posinf=0.0, neginf=0.0)

        H = np.ascontiguousarray(H, dtype=np.float64)
        q = np.ascontiguousarray(q.flatten(), dtype=np.float64)
        A = np.ascontiguousarray(A, dtype=np.float64)
        b = np.ascontiguousarray(b.flatten(), dtype=np.float64)
        Aeq = np.ascontiguousarray(Aeq, dtype=np.float64)
        beq = np.ascontiguousarray(beq.flatten(), dtype=np.float64)

        # ----------------------------------------------------
        # 解 QP
        # ----------------------------------------------------
        time1 = time.time()
        solver_kwargs = {}
        if qp_solver == "mosek":
            solver_kwargs["mosek_params"] = {
                "MSK_IPAR_LOG": 0,
                "MSK_IPAR_MAX_NUM_WARNINGS": 0,
            }
        solution = solve_qp(
            P=_to_pruned_csc(H),
            q=q,
            G=_to_pruned_csc(A),
            h=b,
            A=_to_pruned_csc(Aeq) if Aeq.size > 0 else None,
            b=beq if beq.size > 0 else None,
            lb=np.zeros(numVariable),
            ub=np.ones(numVariable),
            solver=qp_solver,
            verbose=False,
            **solver_kwargs,
        )

        if solution is None:
            raise RuntimeError(
                "QP 求解失败：solve_qp 返回 None，问题可能不可行。"
                "请检查 PDF / CVaR / 边缘约束是否冲突。"
            )

        fval = 0.5 * np.dot(solution.T, np.dot(H, solution)) + np.dot(q, solution)
        time2 = time.time()

        print(f"[6] QP fval = {fval:.6f}, solve time = {time2 - time1:.4f}s")

        # ----------------------------------------------------
        # 收敛判断 + 阻尼更新
        # ----------------------------------------------------
        cost.append(fval)
        diff_norm = np.linalg.norm(solution - PIa)
        soludiff.append(diff_norm)

        if diff_norm < ConvThredhold:
            ConvCounter += 1
        else:
            ConvCounter = 0

        if ConvCounter > 2:
            ConvFlag = True

        if iter_count > 0:
            gamma = gamma * (1.0 - aa * gamma)

        PIa = PIa + gamma * (solution - PIa)
        PIa = np.clip(PIa, 0.0, 1.0)

        iter_count += 1

    # --------------------------------------------------------
    # Step 7: 根据最终流量构建 TransferMatrix
    # --------------------------------------------------------
    W = solution
    TransferMatrix = np.zeros((len(current_means), len(conbinedmeans_list)))

    for n in range(len(index_next_gc)):
        index = index_next_gc[n]
        indexList = np.where(path_table[:, 1].astype(int) == index)[0]
        for j in range(len(indexList)):
            src_i = int(path_table[indexList[j], 0])
            TransferMatrix[src_i, index] += W[indexList[j]]

    # 清理过小项并按行归一
    indexlist2 = np.where(TransferMatrix < min_weight)
    ppp = np.sum(TransferMatrix, axis=1)
    TransferMatrix[indexlist2] = 0.0

    row_sum = np.sum(TransferMatrix, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        TransferMatrix = (TransferMatrix.T / row_sum).T * ppp[:, np.newaxis]
    TransferMatrix[np.isinf(TransferMatrix) | np.isnan(TransferMatrix)] = 0.0

    qqq = np.sum(TransferMatrix, axis=0)
    indexlist3 = np.where(qqq > 0)[0]

    print(f"[7] TransferMatrix 计算完成，非零列数: {len(indexlist3)}")
    for i in range(TransferMatrix.shape[0]):
        for j in range(TransferMatrix.shape[1]):
            if abs(TransferMatrix[i, j]) > 1e-9:
                print(f"({i}, {j}) = {TransferMatrix[i, j]:.6f}")

    # --------------------------------------------------------
    # Step 8: 提取下一步 GMM 参数
    # --------------------------------------------------------
    next_mu = []
    next_sigma = []
    next_weight = []

    for n in range(len(index_next_gc)):
        index = index_next_gc[n]
        if index in indexlist3.tolist():
            next_mu.append(conbinedmeans_list[index])
            next_sigma.append(conbinedcovs_list[index])
            next_weight.append(qqq[index])

    total_weights = np.sum(next_weight)
    if total_weights <= 1e-12:
        # 兜底：若优化后无有效下一步分量，则退回目标
        print("[8] 无有效下一步 GMM，fallback to target")
        next_mu = fmeans
        next_sigma = fcovs
        next_weight = fweights
        flag = 1
    else:
        next_weight = [w / total_weights for w in next_weight]

    print(f"[8] 最终 GMM 均值: {next_mu}")
    print(f"[8] 最终 GMM 方差: {next_sigma}")
    print(f"[8] 最终 GMM 权重: {next_weight}")

    return (
        next_mu,
        next_sigma,
        next_weight,
        current_means,
        current_covs,
        current_weights,
        TransferMatrix,
        flag,
    )


def interpGMM_PRM(means1, covs1, weights1, means2, covs2, weights2, TransferMatrix, flag):
    delDist = 0.2
    dimW = 3
    WG_sq, W, _ = calWGMetric_speedUp(means1, covs1, weights1, means2, covs2, weights2)
    d = np.sqrt(WG_sq)
    numPoint = math.ceil(d / delDist)
    W = TransferMatrix
    if flag == 1:
        W = W.reshape((len(means1), len(means2)), order='F')
        W = np.delete(W, np.where(np.all(W == 0, axis=0))[0], axis=1)
    GMM = []
    WStack = []
    if numPoint <= 1:
        GMM.append([means2, covs2, weights2])
        WStack.append(W)
        return GMM, WStack
    t = np.linspace(0, 1, numPoint + 1)
    t = t[1:]
    W = np.delete(W, np.where(np.all(W == 0, axis=0))[0], axis=1)
    idxListW = np.where(W.flatten(order='F') > 0)[0]
    numComponent_p = idxListW.shape[0]
    W_0 = np.zeros((len(means1), numComponent_p))
    W_1 = np.zeros((numComponent_p, len(means2)))
    Weight = np.zeros(numComponent_p)
    for k in range(numComponent_p):
        n, m = np.unravel_index(idxListW[k], (len(means1), len(means2)), order='F')
        Weight[k] = W[n, m]
        W_0[n, k] = W[n, m]
        W_1[k, m] = W[n, m]
    W_0_sum = np.sum(W_0, axis=1, keepdims=True)
    W_0 = W_0 / W_0_sum
    W_1_sum = np.sum(W_1, axis=1, keepdims=True)
    W_1 = W_1 / W_1_sum

    for i in range(t.shape[0] - 1):
        t_i = t[i]
        Mu = np.zeros((numComponent_p, dimW))
        Sigma = np.zeros((numComponent_p, dimW, dimW))
        for k in range(numComponent_p):
            n, m = np.unravel_index(idxListW[k], (len(means1), len(means2)), order='F')
            sigma_p0_sqrt = scipy.linalg.sqrtm(np.array(covs1[n]))
            Mu[k, :] = (1 - t_i) * np.array(means1[n]) + t_i * np.array(means2[m])
            temp = scipy.linalg.sqrtm(sigma_p0_sqrt @ np.array(covs2[m]) @ sigma_p0_sqrt)
            Sigma[k, :, :] = np.linalg.inv(sigma_p0_sqrt) @ ((1 - t_i) * np.array(covs1[n]) + t_i * temp) @ (
                    (1 - t_i) * np.array(covs1[n]) + t_i * temp) @ np.linalg.inv(sigma_p0_sqrt)
            Sigma[k, 0, 1] = Sigma[k, 1, 0]
        GMM.append([Mu.tolist(), Sigma.tolist(), Weight.tolist()])
        if i == 1:
            WStack.append(W_0)
        else:
            WStack.append(np.eye(numComponent_p))
    GMM.append([means2, covs2, weights2])
    print(f"[8] 插值 GMM 生成完成: {len(GMM)} 步")
    WStack.append(W_1)
    return GMM, WStack

