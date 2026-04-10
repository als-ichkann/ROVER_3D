"""
ROVER_3D ：主入口与规划进程
PlanningAPFProcess: GMM 宏观规划 + APF 轨迹生成，单步式接口供 ROS2 节点调用。
与 MPC 分离：仅发布轨迹，控制由 mpc_control 订阅执行。
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np

from . import control_law_3D
from . import Planning_3D
import networkx as nx


def _mean_to_key(m) -> tuple:
    """将均值转为可哈希的 tuple，用于索引匹配。"""
    arr = np.asarray(m).flatten()
    return tuple(float(x) for x in arr[:3])


def _find_mean_index(means_list, mean) -> int:
    """在 means_list 中查找与 mean 匹配的索引（坐标容差匹配）。"""
    key = _mean_to_key(mean)
    for i, m in enumerate(means_list):
        if np.allclose(np.asarray(m).flatten()[:3], np.asarray(key)):
            return i
    raise ValueError(f"Mean {mean} not found in means_list")


def _map_goals_to_nearest_gc(
    goal_means: list,
    goal_covs: list,
    goal_weights: list,
    GC_means: list,
    GC_covs: list,
) -> tuple:
    """
    将发布的目标点映射到最近的离散 GC 节点。
    多个目标映射到同一 GC 时合并权重。
    :return: (fmeans, fcovs, fweights) 目标均已在 GC 节点集合内
    """
    if not goal_means or not GC_means:
        return [], [], []

    gm_arr = np.asarray(goal_means, dtype=float).reshape(-1, 3)
    gc_arr = np.asarray(GC_means, dtype=float).reshape(-1, 3)
    weights = list(goal_weights) if goal_weights else [1.0] * len(goal_means)
    while len(weights) < len(goal_means):
        weights.append(1.0)
    weights = weights[: len(goal_means)]

    from collections import defaultdict
    merged: dict = defaultdict(float)
    for i, gm in enumerate(gm_arr):
        dists = np.linalg.norm(gc_arr - gm, axis=1)
        best_idx = int(np.argmin(dists))
        merged[best_idx] += weights[i]

    idx_order = sorted(merged.keys())
    fmeans = [list(GC_means[i]) for i in idx_order]
    fcovs = [np.asarray(GC_covs[i]).reshape(3, 3) if np.size(GC_covs[i]) == 9 else GC_covs[i] for i in idx_order]
    total_w = sum(merged.values())
    fweights = [merged[i] / total_w for i in idx_order]
    return fmeans, fcovs, fweights


def _adj_to_graph(adj: np.ndarray, n: int) -> "nx.DiGraph":
    """将邻接矩阵转为 networkx 有向图。"""
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] != 0 and not np.isnan(adj[i, j]):
                G.add_edge(i, j, weight=float(adj[i, j]))
    return G


def _require_config_dir(config_dir: Optional[str]) -> str:
    if not config_dir or not str(config_dir).strip():
        raise ValueError(
            "PlanningAPFProcess 需要非空 config_dir，目录中须包含预计算文件："
            "GC_means_3D.json, GC_covs_3D.json, Wasserstein_table_3D.npy, "
            "Node_PDF_table_3D.npy, Graph_GC_3D.npy（可用 scripts/precompute_config_prior.py 生成）"
        )
    return os.path.abspath(os.path.expanduser(str(config_dir).strip()))


def _load_precomputed_tables(config_dir: str) -> Tuple[list, list, np.ndarray, np.ndarray, np.ndarray]:
    """仅从磁盘加载 GC 与 Wasserstein / Node_PDF / Graph 邻接，不做运行时重算。"""
    files = (
        ("GC_means_3D.json", "json"),
        ("GC_covs_3D.json", "json"),
        ("Wasserstein_table_3D.npy", "npy"),
        ("Node_PDF_table_3D.npy", "npy"),
        ("Graph_GC_3D.npy", "npy"),
    )
    paths = [(name, os.path.join(config_dir, name), kind) for name, kind in files]
    missing = [name for name, p, _ in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"预计算文件缺失（目录 {config_dir!r}）：{', '.join(missing)}"
        )
    path_means = paths[0][1]
    path_covs = paths[1][1]
    with open(path_means, "r", encoding="utf-8") as f:
        gc_means = json.load(f)
    with open(path_covs, "r", encoding="utf-8") as f:
        raw_covs = json.load(f)
    gc_covs = [
        np.asarray(c).reshape(3, 3) if np.size(c) == 9 else np.asarray(c)
        for c in raw_covs
    ]
    num = len(gc_means)
    if num == 0:
        raise ValueError("GC_means_3D.json 为空，至少需要 1 个 GC 节点")
    if len(gc_covs) != num:
        raise ValueError(
            f"GC_covs 条数 ({len(gc_covs)}) 与 GC_means ({num}) 不一致"
        )
    w = np.load(paths[2][1])
    pdf = np.load(paths[3][1])
    graph_adj = np.load(paths[4][1])
    exp = (num, num)
    if tuple(w.shape) != exp or tuple(pdf.shape) != exp:
        raise ValueError(
            f"Wasserstein / Node_PDF 形状须为 {exp}，实际 W={w.shape}, PDF={pdf.shape}"
        )
    if tuple(graph_adj.shape) != exp:
        raise ValueError(f"Graph_GC_3D 形状须为 {exp}，实际 {graph_adj.shape}")
    return gc_means, gc_covs, w, pdf, graph_adj


class PlanningAPFProcess:
    """
    ROS2 兼容规划进程：单步式 run_one_cycle 接口。
    无多进程、无 MPC：仅做 GMM 优化 + APF 轨迹生成。
    """

    def __init__(
        self,
        num_robots: int,
        esdf_map,
        xa: float,
        xb: float,
        ya: float,
        yb: float,
        za: float,
        zb: float,
        goal_means: list,
        goal_covs: list,
        goal_weights: list,
        gmm_interp_steps: int = 5,
        max_apf_try: int = 10,
        use_gmm_trajectory_slp: bool = True,
        config_dir: Optional[str] = None,
    ):
        self.num_robots = num_robots
        self.esdf_map = esdf_map
        self.xa, self.xb = xa, xb
        self.ya, self.yb = ya, yb
        self.za, self.zb = za, zb
        self.gmm_interp_steps = gmm_interp_steps
        self.max_apf_try = max_apf_try
        self.use_gmm_trajectory_slp = use_gmm_trajectory_slp
        self.alpha = 0.05

        cfg = _require_config_dir(config_dir)
        gc_means, gc_covs, w_load, pdf_load, graph_load = _load_precomputed_tables(cfg)
        self.GC_means = gc_means
        self.GC_covs = gc_covs

        # 发布的目标点自动归并到最近的离散 GC 节点
        self.fmeans, self.fcovs, self.fweights = _map_goals_to_nearest_gc(
            goal_means, goal_covs, goal_weights, self.GC_means, self.GC_covs
        )
        # 节点列表仅含 GC 离散节点，不含目标点（目标已映射到 GC）
        self.conbinedmeans_list = list(self.GC_means)
        self.conbinedcovs_list = [
            np.asarray(c).reshape(3, 3) if np.size(c) == 9 else np.asarray(c)
            for c in self.GC_covs
        ]
        Numnode = len(self.conbinedmeans_list)

        self.Wasserstein_table = w_load
        self.Node_PDF_table = pdf_load
        _, self.Graph_GC = Planning_3D.shortest_path(
            _adj_to_graph(graph_load, Numnode)
        )

        # 当前分布：从初始机器人位置估计
        self.current_means = list(self.fmeans)
        self.current_covs = list(self.fcovs)
        self.current_weights = list(self.fweights)
        self.optimization_k = 0
        self.goalFlag = 1
        self.StopFlag = 0
        self.flag = 0
        self.GMM: List[list] = []
        self.WStack: List = []
        self.step = 0
        self.robots_positions_expected: Optional[np.ndarray] = None
        self.stop_flag = False

    def _gmm_score_samples(self, means, covs, weights, points: np.ndarray) -> np.ndarray:
        """用 GMM 对点集计算 log 概率。"""
        from sklearn.mixture import GaussianMixture
        n_comp = len(means)
        gmm = GaussianMixture(n_components=n_comp, covariance_type='full')
        gmm.means_ = np.array(means)
        gmm.covariances_ = np.array([np.asarray(c).reshape(3, 3) for c in covs])
        gmm.weights_ = np.array(weights)
        gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))
        return gmm.score_samples(points)

    def run_one_cycle(
        self, robots_positions: np.ndarray
    ) -> Optional[Tuple[List[np.ndarray], bool]]:
        """
        执行一次规划循环。
        :param robots_positions: (N, 3) 当前机器人位置
        :return: (trajectories, stop_flag) 或 None（本步无新轨迹）
        """
        robots_positions = np.asarray(robots_positions, dtype=float).reshape(-1, 3)
        if robots_positions.shape[0] != self.num_robots:
            return None

        if self.robots_positions_expected is None:
            self.robots_positions_expected = np.array(robots_positions, copy=True)

        # 若需要宏观优化
        if self.goalFlag and not self.StopFlag:
            if self.flag == 1:
                self.StopFlag = 1
                self.stop_flag = True
                return None
            self.robots_positions_expected = np.array(robots_positions, copy=True)
            rpe = self.robots_positions_expected
            # 检查是否需要重新估计当前 GMM
            if len(self.GMM) > 0 and self.step > 0:
                curr = self.GMM[self.step - 1] if self.step > 0 else [self.current_means, self.current_covs, self.current_weights]
                if len(curr) >= 3 and len(curr[0]) > 0:
                    try:
                        scores = self._gmm_score_samples(curr[0], curr[1], curr[2], rpe)
                        if np.min(scores) < np.log(1e-4):
                            self.current_means, self.current_covs, self.current_weights = control_law_3D.estimate_swarm_GMM_3D(
                                self.conbinedmeans_list, self.conbinedcovs_list, rpe
                            )
                        else:
                            self.current_means, self.current_covs, self.current_weights = curr[0], curr[1], curr[2]
                    except Exception:
                        self.current_means, self.current_covs, self.current_weights = control_law_3D.estimate_swarm_GMM_3D(
                            self.conbinedmeans_list, self.conbinedcovs_list, rpe
                        )
                else:
                    self.current_means, self.current_covs, self.current_weights = control_law_3D.estimate_swarm_GMM_3D(
                        self.conbinedmeans_list, self.conbinedcovs_list, rpe
                    )
            else:
                self.current_means, self.current_covs, self.current_weights = control_law_3D.estimate_swarm_GMM_3D(
                    self.conbinedmeans_list, self.conbinedcovs_list, rpe
                )

            if not self.use_gmm_trajectory_slp:
                self.GMM = [[self.fmeans, self.fcovs, self.fweights]]
                self.WStack = [np.eye(1)]
                self.goalFlag = 0
                self.optimization_k += 1
                self.step = 0
            else:
                current_goal = [self.current_means, self.current_covs, self.current_weights]
                if self.step < len(self.GMM) and len(self.GMM) > 0:
                    prev = self.GMM[self.step - 1] if self.step > 0 else [self.current_means, self.current_covs, self.current_weights]
                    if len(prev) >= 3:
                        current_goal = prev

                (
                    self.goal_means,
                    self.goal_covs,
                    self.goal_weights,
                    self.current_means,
                    self.current_covs,
                    self.current_weights,
                    TransferMatrix,
                    self.flag,
                ) = Planning_3D.Optimization_SLP(
                    self.current_means,
                    self.current_covs,
                    self.current_weights,
                    self.fmeans,
                    self.fcovs,
                    self.fweights,
                    self.conbinedmeans_list,
                    self.conbinedcovs_list,
                    self.esdf_map,
                    self.alpha,
                    current_goal[0],
                    current_goal[1],
                    current_goal[2],
                    self.Graph_GC,
                    self.Wasserstein_table,
                    self.Node_PDF_table,
                )
                self.GMM, self.WStack = Planning_3D.interpGMM_PRM(
                    self.current_means,
                    self.current_covs,
                    self.current_weights,
                    self.goal_means,
                    self.goal_covs,
                    self.goal_weights,
                    TransferMatrix,
                    self.flag,
                )
                self.goalFlag = 0
                self.optimization_k += 1
                self.step = 0

        if len(self.GMM) == 0:
            return None
        if self.step >= len(self.GMM):
            self.goalFlag = 1
            self.step = len(self.GMM) - 1
        next_m, next_c, next_w = self.GMM[self.step]
        next_means = [list(m) for m in next_m]
        next_covs = [np.asarray(c).reshape(3, 3) if np.size(c) == 9 else c for c in next_c]
        next_weights = list(next_w)

        # 每周期用实时 odom 作为 APF 起点。若沿用上一拍 APF 输出，多步开环会使
        # 「等效位置」漂移，吸引势相对虚点计算，易出现全体朝某一象限偏航（与目标均值无关）。
        self.robots_positions_expected = np.array(robots_positions, copy=True)

        # APF 一步
        self.robots_positions_expected, robots_positions_list, _, _, _ = control_law_3D.APF(
            next_means,
            next_covs,
            next_weights,
            self.robots_positions_expected,
            self.esdf_map,
            MaxNumTry=self.max_apf_try,
        )

        trajectories = []
        for i in range(self.num_robots):
            traj = np.array([pos[i] for pos in robots_positions_list])
            trajectories.append(traj)

        if self.step == len(self.GMM) - 1:
            self.goalFlag = 1
        self.step += 1
        return trajectories, self.stop_flag
