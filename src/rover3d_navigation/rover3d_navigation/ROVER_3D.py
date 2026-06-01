"""
ROVER_3D ：主入口与规划进程
PlanningProcess: GMM 宏观规划 + 微观轨迹生成，单步式接口供 ROS2 节点调用。
与 MPC 分离：仅发布轨迹，控制由 mpc_control 订阅执行。
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from . import apf_control_law
from . import density_control_law
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
            "PlanningProcess 需要非空 config_dir，目录中须包含预计算文件："
            "GC_means_3D.json, GC_covs_3D.json, Wasserstein_table_3D.npy, "
            "Node_PDF_table_3D.npy, Graph_GC_3D.npy（可用 scripts/precompute_config_prior.py 生成）"
        )
    return os.path.abspath(os.path.expanduser(str(config_dir).strip()))


def _normalize_weights(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w[w < 0.0] = 0.0
    s = float(np.sum(w))
    if s <= eps:
        return np.ones_like(w) / max(len(w), 1)
    return w / s


def _estimate_swarm_gmm_em(
    positions: np.ndarray,
    n_components: int,
    reg_covar: float = 1e-6,
) -> tuple:
    pts = np.asarray(positions, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise ValueError("positions is empty")
    k = max(1, min(int(n_components), pts.shape[0]))
    em = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=float(reg_covar),
        random_state=0,
    )
    em.fit(pts)
    means = np.asarray(em.means_, dtype=float)
    covs = np.asarray(em.covariances_, dtype=float)
    weights = _normalize_weights(np.asarray(em.weights_, dtype=float))
    return means, covs, weights


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


def _weighted_gmm_center(means: list, weights: list) -> np.ndarray:
    means_np = np.asarray(means, dtype=float).reshape(-1, 3)
    weights_np = np.asarray(weights, dtype=float).reshape(-1)
    if means_np.shape[0] == 0:
        return np.zeros(3, dtype=float)
    if np.sum(weights_np) > 1e-12 and means_np.shape[0] == weights_np.shape[0]:
        weights_np = weights_np / np.sum(weights_np)
        return np.average(means_np, axis=0, weights=weights_np)
    return np.mean(means_np, axis=0)


class PlanningProcess:
    """
    ROS2 兼容规划进程：单步式 run_one_cycle 接口。
    无多进程、无 MPC：仅做 GMM 优化 + 微观轨迹生成。
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
        max_micro_try: int = 10,
        micro_goal_lock_radius: float = 0.25,
        macro_replan_radius: float = 0.6,
        density_traj_steps: int = 8,
        density_use_astar: bool = True,
        density_astar_safe_margin: float = 0.4,
        density_astar_robot_radius: float = 0.18,
        slp_epsilon: float = 0.2,
        use_gmm_trajectory_slp: bool = True,
        micro_controller: str = "apf",
        apf_use_esdf: bool = True,
        config_dir: Optional[str] = None,
        initial_positions: Optional[np.ndarray] = None,
    ):
        self.num_robots = num_robots
        self.esdf_map = esdf_map
        self.xa, self.xb = xa, xb
        self.ya, self.yb = ya, yb
        self.za, self.zb = za, zb
        self.gmm_interp_steps = gmm_interp_steps
        self.max_micro_try = int(max_micro_try)
        self.micro_goal_lock_radius = max(0.0, float(micro_goal_lock_radius))
        self.macro_replan_radius = max(0.05, float(macro_replan_radius))
        self.density_traj_steps = max(1, int(density_traj_steps))
        self.density_use_astar = bool(density_use_astar)
        self.density_astar_safe_margin = max(0.0, float(density_astar_safe_margin))
        self.density_astar_robot_radius = max(0.05, float(density_astar_robot_radius))
        self.slp_epsilon = float(slp_epsilon)
        self.use_gmm_trajectory_slp = use_gmm_trajectory_slp
        self.micro_controller = str(micro_controller).lower().strip()
        self.apf_use_esdf = bool(apf_use_esdf)
        if self.micro_controller not in ("apf", "density"):
            raise ValueError(
                f"Unsupported micro_controller={micro_controller!r}, expected 'apf' or 'density'"
            )
        self.alpha = 0.05
        self._density_controller = density_control_law.DensityController3D(
            seed=0,
            traj_steps=self.density_traj_steps,
            esdf_map=self.esdf_map,
            use_astar=self.density_use_astar,
            astar_safe_margin=self.density_astar_safe_margin,
            astar_robot_radius=self.density_astar_robot_radius,
        )

        cfg = _require_config_dir(config_dir)
        gc_means, gc_covs, w_load, pdf_load, graph_load = _load_precomputed_tables(cfg)
        self.GC_means = gc_means
        self.GC_covs = gc_covs

        # 发布的目标点自动归并到最近的离散 GC 节点
        self.fmeans, self.fcovs, self.fweights = _map_goals_to_nearest_gc(
            goal_means, goal_covs, goal_weights, self.GC_means, self.GC_covs
        )
        print(
            f"[GOAL] raw_goal_means={goal_means}, raw_goal_weights={goal_weights}"
        )
        print(
            f"[GOAL] mapped_fmeans={self.fmeans}, mapped_fweights={self.fweights}"
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

        # 当前分布：从初始机器人位置 EM 估计并投射到 GC 节点
        if initial_positions is not None:
            pts = np.asarray(initial_positions, dtype=float).reshape(-1, 3)
            k = max(1, min(self.num_robots, len(self.fmeans)))
            em_means, em_covs, em_weights = _estimate_swarm_gmm_em(pts, n_components=k)
            self.current_means, self.current_covs, self.current_weights = _map_goals_to_nearest_gc(
                em_means.tolist(),
                [em_covs[i] for i in range(len(em_means))],
                em_weights.tolist(),
                self.GC_means,
                self.GC_covs,
            )
            print(
                f"[INIT] current GMM from swarm EM+GC: "
                f"means={self.current_means}, weights={self.current_weights}"
            )
        else:
            self.current_means = []
            self.current_covs = []
            self.current_weights = []
        self.optimization_k = 0
        self.goalFlag = 1
        self.StopFlag = 0
        self.flag = 0
        self.GMM: List[list] = []
        # WStack: OT 传输矩阵栈，预留供 density OT 等下游使用
        self.WStack: List = []
        self.step = 0
        self.robots_positions_expected: Optional[np.ndarray] = None
        self.stop_flag = False

    def _macro_state_for_slp(self) -> Tuple[list, list, list]:
        """CVaR 回退 / SLP 输入参考：优先上一段插值末态，否则用最近一次宏观输出。"""
        if len(self.GMM) > 0:
            last_m, last_c, last_w = self.GMM[-1]
            return [list(m) for m in last_m], list(last_c), list(last_w)
        if getattr(self, "goal_means", None):
            return list(self.goal_means), list(self.goal_covs), list(self.goal_weights)
        return list(self.current_means), list(self.current_covs), list(self.current_weights)

    def _hold_trajectories(self, robots_positions: np.ndarray) -> List[np.ndarray]:
        """发布单点轨迹，令 MPC 保持当前位置。"""
        pos = np.asarray(robots_positions, dtype=float).reshape(-1, 3)
        return [np.array([pos[i]], dtype=float) for i in range(self.num_robots)]

    def _esdf_usable_for_planning(self) -> bool:
        """ESDF 须已就绪且栅格中存在自由空间（非全零空图）。"""
        if self.esdf_map is None:
            return False
        if hasattr(self.esdf_map, "refresh"):
            self.esdf_map.refresh()
        if hasattr(self.esdf_map, "is_ready") and not self.esdf_map.is_ready:
            return False
        grid = getattr(self.esdf_map, "_dist_grid", None)
        if grid is not None:
            try:
                if not bool(np.any(np.asarray(grid, dtype=float) > 0.0)):
                    return False
            except Exception:
                pass
        return True

    def _current_gmm_segment_target(self) -> Optional[np.ndarray]:
        """当前宏观插值段末态的加权中心。"""
        if len(self.GMM) == 0:
            return None
        last_m, _, last_w = self.GMM[-1]
        return _weighted_gmm_center(last_m, last_w)

    def _swarm_reached_macro_segment_end(self, robots_positions: np.ndarray) -> bool:
        """集群中心是否已到达当前宏观段终点（才允许触发下一轮 SLP）。"""
        target = self._current_gmm_segment_target()
        if target is None:
            return True
        center = np.mean(np.asarray(robots_positions, dtype=float).reshape(-1, 3), axis=0)
        dist = float(np.linalg.norm(center - target))
        return dist <= self.macro_replan_radius

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
            if not self._esdf_usable_for_planning():
                print("[MACRO] ESDF 未就绪或为空图，跳过 SLP，原地 hold")
                return self._hold_trajectories(robots_positions), self.stop_flag
            self.robots_positions_expected = np.array(robots_positions, copy=True)
            rpe = self.robots_positions_expected

            if not self.use_gmm_trajectory_slp:
                self.GMM = [[self.fmeans, self.fcovs, self.fweights]]
                self.WStack = [np.eye(1)]
                self.goalFlag = 0
                self.optimization_k += 1
                self.step = 0
                self._density_controller.reset()
            else:
                # CVaR 超限时回退用的宏观参考；每拍 Step0 EM 会用实时 odom 重估 current
                macro_means, macro_covs, macro_weights = self._macro_state_for_slp()
                current_goal = [macro_means, macro_covs, macro_weights]
                init_means = np.asarray(macro_means, dtype=float)
                init_covs = np.asarray(macro_covs, dtype=float)
                init_weights = np.asarray(macro_weights, dtype=float)
                em_k = 1

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
                    init_means,
                    init_covs,
                    init_weights,
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
                    robots_positions=rpe,
                    epsilon=self.slp_epsilon,
                    use_em_from_odom=True,
                    em_n_components=em_k,
                )
                if self.flag == -1:
                    print(
                        "[MACRO] SLP 路径表为空 (flag=-1)，保持宏观态并原地 hold，"
                        "等待 ESDF/地图恢复后重试"
                    )
                    self.goalFlag = 0
                    self.step = 0
                    if not self.GMM:
                        cm = [list(m) for m in np.asarray(self.current_means, dtype=float).reshape(-1, 3)]
                        cc = [
                            np.asarray(c).reshape(3, 3) if np.size(c) == 9 else np.asarray(c)
                            for c in self.current_covs
                        ]
                        cw = list(self.current_weights)
                        self.GMM = [[cm, cc, cw]]
                        self.WStack = [np.eye(max(1, len(cm)))]
                    return self._hold_trajectories(robots_positions), self.stop_flag
                self.GMM, self.WStack = Planning_3D.interpGMM_PRM(
                    self.current_means,
                    self.current_covs,
                    self.current_weights,
                    self.goal_means,
                    self.goal_covs,
                    self.goal_weights,
                    TransferMatrix,
                    self.flag,
                    interp_steps=self.gmm_interp_steps,
                )
                self.goalFlag = 0
                self.optimization_k += 1
                self.step = 0
                self._density_controller.reset()

        if len(self.GMM) == 0:
            return None
        if self.step >= len(self.GMM):
            if not self._swarm_reached_macro_segment_end(robots_positions):
                target = self._current_gmm_segment_target()
                center = np.mean(robots_positions, axis=0)
                dist = float(np.linalg.norm(center - target)) if target is not None else -1.0
                print(
                    f"[MACRO] GMM 段已播完但集群未到位: center_dist={dist:.3f}, "
                    f"need<={self.macro_replan_radius:.3f}, hold"
                )
                return self._hold_trajectories(robots_positions), self.stop_flag
            self.goalFlag = 1
            return None
        next_m, next_c, next_w = self.GMM[self.step]
        next_means = [list(m) for m in next_m]
        next_covs = [np.asarray(c).reshape(3, 3) if np.size(c) == 9 else c for c in next_c]
        next_weights = list(next_w)

        # 每周期用实时 odom 作为微观轨迹起点。若沿用上一拍输出，多步开环会使
        # 「等效位置」漂移，吸引势相对虚点计算，易出现全体朝某一象限偏航（与目标均值无关）。
        self.robots_positions_expected = np.array(robots_positions, copy=True)

        # 仅在插值末步且集群中心接近最终目标 GC 时 hold，避免在中间宏观步误触发下一拍 SLP。
        if (
            self.micro_goal_lock_radius > 0
            and self.step == len(self.GMM) - 1
            and len(self.fmeans) > 0
        ):
            final_center = _weighted_gmm_center(self.fmeans, self.fweights)
            swarm_center = np.mean(self.robots_positions_expected, axis=0)
            center_dist = float(np.linalg.norm(swarm_center - final_center))
            if center_dist <= self.micro_goal_lock_radius:
                trajectories = [
                    np.array([self.robots_positions_expected[i]], dtype=float)
                    for i in range(self.num_robots)
                ]
                print(
                    f"[MICRO] lock at final goal: step={self.step}, "
                    f"center_dist={center_dist:.4f}, radius={self.micro_goal_lock_radius:.4f}"
                )
                self.step += 1
                self.goalFlag = 1
                return trajectories, self.stop_flag

        if self.micro_controller == "apf":
            self.robots_positions_expected, robots_positions_list, _, _, _ = apf_control_law.APF(
                next_means,
                next_covs,
                next_weights,
                self.robots_positions_expected,
                self.esdf_map,
                MaxNumTry=self.max_micro_try,
                apf_use_esdf=self.apf_use_esdf,
            )
        else:
            if self.step > 0 and self.step - 1 < len(self.GMM):
                src_m, src_c, src_w = self.GMM[self.step - 1]
            else:
                src_m, src_c, src_w = self.current_means, self.current_covs, self.current_weights
            next_pos, robots_positions_list = self._density_controller.step(
                src_m,
                src_c,
                src_w,
                next_means,
                next_covs,
                next_weights,
                self.robots_positions_expected,
            )
            # 约束到地图边界；目标点需满足 ESDF 安全裕度
            esdf_margin = self.density_astar_safe_margin + self.density_astar_robot_radius
            next_pos[:, 0] = np.minimum(np.maximum(next_pos[:, 0], self.xa + 1e-6), self.xb - 1e-6)
            next_pos[:, 1] = np.minimum(np.maximum(next_pos[:, 1], self.ya + 1e-6), self.yb - 1e-6)
            next_pos[:, 2] = np.minimum(np.maximum(next_pos[:, 2], self.za + 1e-6), self.zb - 1e-6)
            try:
                d = self.esdf_map.get_esdf(next_pos)
                d = np.asarray(d, dtype=float).reshape(-1)
                bad = np.where(d < esdf_margin)[0]
                if len(bad) > 0:
                    next_pos[bad] = self.robots_positions_expected[bad]
            except Exception:
                pass
            if robots_positions_list:
                self.robots_positions_expected = np.array(robots_positions_list[-1], copy=True)
            else:
                self.robots_positions_expected = next_pos

        trajectories = []
        for i in range(self.num_robots):
            traj = np.array([pos[i] for pos in robots_positions_list])
            trajectories.append(traj)

        self.step += 1
        return trajectories, self.stop_flag
