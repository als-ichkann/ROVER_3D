import math

import networkx as nx
import numpy as np
import pandas as pd


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
):
    LinearConnectFlag = 1
    delDist = 0.05
    W_tf = 3
    NodePairTable = []
    for i in range(len(current_means)):
        current_mu = current_means[i]
        current_mu_i = _find_mean_index(conbinedmeans_list, current_mu)
        print(f"\n[Level 1] 当前GMM组分 i={i}, mu={current_mu}, index={current_mu_i}")
        for n in range(len(conbinedmeans_list)):
            node_mu = conbinedmeans_list[n]
            d = Wasserstein_table[current_mu_i, n]

            if d >= np.sqrt(0) and d <= np.sqrt(4):
                LinearConnectFlag = 1
                point1 = [current_mu[0], current_mu[1], current_mu[2]]
                point2 = [node_mu[0], node_mu[1], node_mu[2]]

                if esdf_map.is_collision_line_segment(point1, point2):
                    LinearConnectFlag = 0
                if LinearConnectFlag == 1:
                    if d < 1e-5:
                        Lagrangian = 0
                    else:
                        Dist_sq = math.ceil(d / delDist) * (delDist**2)
                        Lagrangian = Dist_sq

                    for m in range(len(conbinedmeans_list)):
                        node_mu_m = conbinedmeans_list[m]
                        d_nm = Wasserstein_table[n, m]
                        if d_nm >= np.sqrt(0) and d_nm <= np.sqrt(4):
                            LinearConnectFlag = 1
                            point3 = np.array([node_mu_m[0], node_mu_m[1], node_mu_m[2]])
                            if esdf_map.is_collision_line_segment(point2, point3):
                                LinearConnectFlag = 0

                            if LinearConnectFlag == 1:
                                if d_nm < 1e-5:
                                    Lagrangian2 = 0
                                else:
                                    Lagrangian2 = math.ceil(d_nm / delDist) * (delDist**2)

                                table_line = [i, n, m, 0, Lagrangian, Lagrangian2]
                                table_lines = [table_line[:] for _ in range(len(fmeans))]
                                for ii in range(len(fmeans)):
                                    table_lines[ii][3] = ii
                                NodePairTable.extend(table_lines)
                                print(
                                    f"[Add] i={i}, n={n}, m={m}, d={d:.3f}, d_nm={d_nm:.3f}, "
                                    f"L1={Lagrangian}, L2={Lagrangian2}, total added={len(NodePairTable)}"
                                )
    path_table = np.zeros((len(NodePairTable), 8))
    path_table[:, :6] = np.array(NodePairTable)
    subTable = path_table[:, 2:4]

    df = pd.DataFrame(subTable)
    unique_df = df.drop_duplicates()
    unique_df_sorted = unique_df.sort_values(by=unique_df.columns.tolist()).reset_index(drop=True)
    PathTable_Unique = unique_df_sorted.to_numpy()
    idx_Table_Unique = np.array(
        [np.where(np.all(PathTable_Unique == row, axis=1))[0][0] for row in subTable]
    )

    numPathTable_Unique = PathTable_Unique.shape[0]
    dist_n_j_Unique = np.zeros(numPathTable_Unique)
    indexList_n = PathTable_Unique[:, 0]
    indexList_j = PathTable_Unique[:, 1]
    all_pairs_shortest_path_length = Graph_GC
    for m in range(numPathTable_Unique):
        n = int(indexList_n[m])
        j = int(indexList_j[m])
        inner = all_pairs_shortest_path_length.get(n, {})
        dist_n_j_Unique[m] = inner.get(j, float("nan"))
    dist_n_j = dist_n_j_Unique[idx_Table_Unique]
    path_table[:, 6] = dist_n_j
    path_table[:, 7] = path_table[:, 4] + path_table[:, 5] + W_tf * path_table[:, 6]
    path_table = path_table[~np.isnan(dist_n_j), :]
    return path_table
