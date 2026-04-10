from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


def _find_mean_index(means_list, mean):
    """容差匹配：在 means_list 中查找与 mean 最接近的索引，避免 list.index 的浮点精度问题。"""
    arr = np.asarray(mean).flatten()[:3]
    for i, m in enumerate(means_list):
        if np.allclose(np.asarray(m).flatten()[:3], arr):
            return i
    raise ValueError(f"Mean {list(arr)} not found in means_list (len={len(means_list)})")


# 供 Planning_3D 等模块使用（与 _find_mean_index 相同）
find_mean_index = _find_mean_index


def shortest_path(Graph):
    all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path(Graph, weight="weight"))
    all_pairs_shortest_path_length = dict(nx.all_pairs_dijkstra_path_length(Graph, weight="weight"))
    path_existence = {}
    path_lengths = {}
    for node in Graph.nodes():
        path_existence[node] = {}
        path_lengths[node] = {}
        for target in Graph.nodes():
            if node != target:
                if target in all_pairs_shortest_path[node]:
                    path = all_pairs_shortest_path[node][target]
                    weighted_length = sum(Graph[u][v]["weight"] for u, v in zip(path[:-1], path[1:]))
                    path_lengths[node][target] = weighted_length
                    path_existence[node][target] = True
                else:
                    path_lengths[node][target] = float("nan")
                    path_existence[node][target] = False
            else:
                path_lengths[node][target] = 0
                path_existence[node][target] = True
    return path_existence, path_lengths


def _seg_cache_key(p0, p1) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    a = tuple(np.round(np.asarray(p0, dtype=float).flatten()[:3], 4))
    b = tuple(np.round(np.asarray(p1, dtype=float).flatten()[:3], 4))
    return (a, b) if a <= b else (b, a)


def _collision_cached(
    cache: Dict[Tuple[Tuple[float, float, float], Tuple[float, float, float]], bool],
    esdf_map,
    p0,
    p1,
) -> bool:
    k = _seg_cache_key(p0, p1)
    if k not in cache:
        q0 = [float(x) for x in np.asarray(p0, dtype=float).flatten()[:3]]
        q1 = [float(x) for x in np.asarray(p1, dtype=float).flatten()[:3]]
        cache[k] = bool(esdf_map.is_collision_line_segment(q0, q1))
    return cache[k]


def _sorted_neighbor_indices(
    w_row: np.ndarray,
    k: int,
    max_dist: float,
) -> List[int]:
    """按 Wasserstein 距离升序取前 k 个且不超过 max_dist 的节点下标。"""
    d = np.asarray(w_row, dtype=float).reshape(-1)
    order = np.argsort(d, kind="mergesort")
    out: List[int] = []
    for j in order:
        if len(out) >= k:
            break
        if d[int(j)] > max_dist:
            break
        out.append(int(j))
    return out


def notgreedy_genPathTable(
    current_means,
    current_covs,
    current_weights,
    fmeans,
    fcovs,
    fweights,
    conbinedmeans_list,
    conbinedcovs_list,
    esdf_map,
    Graph_GC,
    Wasserstein_table,
    *,
    knn_k: int = 16,
    max_wasserstein: float = 1.5,
    w_tf: float = 3.0,
    w2_cost_power: float = 2.0,
    debug: bool = False,
):
    """
    枚举 2-hop 路径 (current_i → n → m → target_j)，供 SLP/QP 使用。

    Graph_GC: 全对最短路长度 dict，Graph_GC[u][v] 为标量（与 ROVER_3D 预计算一致）。
    复杂度由全节点三重循环降为 O(|current| · K^2 · |fmeans|) 量级（K=knn_k）。

    :param knn_k: 从当前 GC、从中间节点 n 各只保留 Wasserstein 意义下最近的 K 个邻居。
    :param max_wasserstein: 两段 hop 允许的最大 Wasserstein 距离（与表中元素同量纲）。
    :param w_tf: 图距离惩罚系数。
    :param w2_cost_power: 段代价为 d**power；默认 2 即 d^2，比 ceil 离散化更光滑。
    """
    _ = current_covs, current_weights, fcovs, fweights, conbinedcovs_list

    W = np.asarray(Wasserstein_table, dtype=float)
    n_nodes = W.shape[0]
    if W.shape[1] != n_nodes:
        raise ValueError("Wasserstein_table 须为方阵")

    collision_cache: Dict[Tuple[Tuple[float, float, float], Tuple[float, float, float]], bool] = {}
    rows: List[List[float]] = []

    for i in range(len(current_means)):
        current_mu = current_means[i]
        current_mu_i = _find_mean_index(conbinedmeans_list, current_mu)
        if debug:
            print(f"\n[gen_path] i={i}, current_gc={current_mu_i}, mu={current_mu}")

        neighbors_n = _sorted_neighbor_indices(W[current_mu_i], knn_k, max_wasserstein)

        for n in neighbors_n:
            node_mu = conbinedmeans_list[n]
            d = float(W[current_mu_i, n])
            p_cur = np.asarray(current_mu, dtype=float).flatten()[:3]
            p_n = np.asarray(node_mu, dtype=float).flatten()[:3]
            if _collision_cached(collision_cache, esdf_map, p_cur, p_n):
                continue

            if d < 1e-8:
                lag1 = 0.0
            else:
                lag1 = float(d ** w2_cost_power)

            neighbors_m = _sorted_neighbor_indices(W[n], knn_k, max_wasserstein)

            for m in neighbors_m:
                node_mu_m = conbinedmeans_list[m]
                d_nm = float(W[n, m])
                p_m = np.asarray(node_mu_m, dtype=float).flatten()[:3]
                if _collision_cached(collision_cache, esdf_map, p_n, p_m):
                    continue

                if d_nm < 1e-8:
                    lag2 = 0.0
                else:
                    lag2 = float(d_nm ** w2_cost_power)

                inner = Graph_GC.get(n, {})
                gdist = float(inner.get(m, float("nan")))
                if gdist != gdist:  # NaN：图上不可达，不生成路径
                    continue

                total = lag1 + lag2 + w_tf * gdist
                for j in range(len(fmeans)):
                    rows.append([i, n, m, j, lag1, lag2, gdist, total])

    if not rows:
        return np.zeros((0, 8), dtype=float)

    return np.asarray(rows, dtype=float)
