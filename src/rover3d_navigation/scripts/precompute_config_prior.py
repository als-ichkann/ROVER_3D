#!/usr/bin/env python3
"""
预计算 rover3d_navigation 的 config 先验文件。

根据地图配置生成纯离散 GC 节点图，不包含目标点。
发布的目标点将在运行时自动归并到最近的离散 GC 节点。

输出：GC_means_3D.json、GC_covs_3D.json、Wasserstein_table_3D.npy、
      Node_PDF_table_3D.npy、Graph_GC_3D.npy。

用法（从 workspace 根目录）:
  cd /path/to/fishbot_multirobot_sim
  PYTHONPATH=src/rover3d_navigation python3 src/rover3d_navigation/scripts/precompute_config_prior.py

或使用 colcon 构建后:
  source install/setup.bash
  python3 src/rover3d_navigation/scripts/precompute_config_prior.py

可选参数:
  --fiesta-config  FIESTA fiesta.yaml 路径
  --planning-config planning.yaml 路径（默认优先使用）
  --map-source     地图参数来源：auto/planning/fiesta
  --output         输出目录
  --grid-step      GC 节点网格步长 (默认 2.0，需与 planning.yaml 一致)
  --graph-grid-res 构图体素分辨率（默认取 esdf_resolution）
  --interp-step    GMM 插值步长（默认与 graph-grid-res 相同）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import scipy
from scipy.stats import multivariate_normal

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.dirname(os.path.dirname(_script_dir))
_pkg_dir = os.path.join(_pkg_root, "rover3d_navigation")
if os.path.isdir(_pkg_dir):
    sys.path.insert(0, os.path.abspath(_pkg_root))

try:
    import yaml
except ImportError:
    yaml = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

def load_yaml(path: str) -> dict:
    if yaml is None:
        raise ImportError("PyYAML required. Install: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_map_params_from_fiesta(fiesta_config_path: str) -> dict:
    """从 FIESTA fiesta.yaml 读取地图参数 (lx, ly, lz, rx, ry, rz)。"""
    data = load_yaml(fiesta_config_path)
    params = data.get("fiesta_node", {}).get("ros__parameters", data)
    lx = float(params.get("lx", -30.0))
    ly = float(params.get("ly", -30.0))
    lz = float(params.get("lz", 0.0))
    rx = float(params.get("rx", 30.0))
    ry = float(params.get("ry", 30.0))
    rz = float(params.get("rz", 6.0))
    return {
        "map_origin_x": lx,
        "map_origin_y": ly,
        "map_origin_z": lz,
        "map_size_x": rx - lx,
        "map_size_y": ry - ly,
        "map_size_z": rz - lz,
        "esdf_resolution": float(params.get("resolution", 0.2)),
    }


def get_map_params_from_planning(planning_config_path: str) -> dict:
    """从 planning.yaml 读取地图参数（map_origin/map_size/esdf_resolution）。"""
    data = load_yaml(planning_config_path)
    params = data.get("planning_node", {}).get("ros__parameters", data)
    return {
        "map_origin_x": float(params.get("map_origin_x", -5.0)),
        "map_origin_y": float(params.get("map_origin_y", -7.5)),
        "map_origin_z": float(params.get("map_origin_z", 0.0)),
        "map_size_x": float(params.get("map_size_x", 22.0)),
        "map_size_y": float(params.get("map_size_y", 17.0)),
        "map_size_z": float(params.get("map_size_z", 6.0)),
        "esdf_resolution": float(params.get("esdf_resolution", 0.2)),
    }


def Wasserstein_distance(mean1, cov1, mean2, cov2):
    """Wasserstein-2 distance between two Gaussian components."""
    mean1 = np.asarray(mean1, dtype=float)
    mean2 = np.asarray(mean2, dtype=float)
    cov1 = np.asarray(cov1, dtype=float)
    cov2 = np.asarray(cov2, dtype=float)
    add1 = np.linalg.norm(mean1 - mean2)
    if np.array_equal(cov1, cov2):
        return float(add1)
    s0 = scipy.linalg.sqrtm(cov1)
    add2 = np.trace(cov1 + cov2 - 2 * scipy.linalg.sqrtm(s0 @ cov2 @ s0))
    return float((add1 ** 2 + add2) ** 0.5)


def interpGC_speedUp(mean1, cov1, mean2, cov2, gridXYZ, delDist):
    d = Wasserstein_distance(mean1, cov1, mean2, cov2)
    mean1 = np.array(mean1)
    cov1 = np.array(cov1)
    mean2 = np.array(mean2)
    cov2 = np.array(cov2)
    numPoint = math.ceil(d / delDist)

    if numPoint <= 1:
        PDF = multivariate_normal.pdf(gridXYZ, mean=mean2, cov=cov2).astype(np.float32)
        PDF_Vectors = (PDF.reshape(-1, 1) * 0.1 ** 3).astype(np.float32)
        return PDF_Vectors, []

    t = np.linspace(0, 1, numPoint + 1)[1:]
    N = gridXYZ.shape[0]
    PDF_Vectors = np.zeros((N, numPoint), dtype=np.float32)
    Dist_sq_Vector = np.full(numPoint, delDist ** 2, dtype=np.float32)

    cov_equal = np.array_equal(cov1, cov2)
    if not cov_equal:
        Sigma0_sqrt = scipy.linalg.sqrtm(cov1)
        temp_common = scipy.linalg.sqrtm(Sigma0_sqrt @ cov2 @ Sigma0_sqrt)
        Sigma0_sqrt_inv = np.linalg.inv(Sigma0_sqrt)

    for i in range(numPoint):
        t_i = t[i]
        mu = (1 - t_i) * mean1 + t_i * mean2
        if not cov_equal:
            Sigma = Sigma0_sqrt_inv @ ((1 - t_i) * cov2 + t_i * temp_common) @ Sigma0_sqrt_inv
        else:
            Sigma = cov1
        PDF = multivariate_normal.pdf(gridXYZ, mean=mu, cov=Sigma).astype(np.float32)
        PDF_Vectors[:, i] = PDF * (0.1 ** 3)

    return PDF_Vectors, Dist_sq_Vector


def init_Graph_GC(
    conbinedmeans_list,
    conbinedcovs_list,
    Wasserstein_table,
    xa=0,
    ya=0,
    za=0,
    xb=20,
    yb=16,
    zb=2,
    graph_grid_res=0.2,
    delDist=0.2,
    edge_w_max=2.0,
):
    print("Start to create the graph...")
    dx = dy = dz = graph_grid_res
    numGridX = int((xb - xa) / dx + 1)
    numGridY = int((yb - ya) / dy + 1)
    numGridZ = int((zb - za) / dz + 1)
    numGrid = numGridX * numGridY * numGridZ
    xGrid = np.linspace(xa, xb, numGridX)
    yGrid = np.linspace(ya, yb, numGridY)
    zGrid = np.linspace(za, zb, numGridZ)

    gridX, gridY, gridZ = np.meshgrid(xGrid, yGrid, zGrid, indexing="ij")
    gridX = np.round(gridX, decimals=1)
    gridY = np.round(gridY, decimals=1)
    gridZ = np.round(gridZ, decimals=1)
    gridXYZ = np.column_stack(
        (gridX.flatten(order="F"), gridY.flatten(order="F"), gridZ.flatten(order="F"))
    )

    D_stack = []
    total_edges = 0
    n_nodes = len(conbinedmeans_list)
    for i in tqdm(range(n_nodes)):
        mu_i = conbinedmeans_list[i]
        Sigma_i = conbinedcovs_list[i]
        D = np.zeros((1, n_nodes))
        edge_pdf_columns = []
        edge_indices = []
        for j in range(n_nodes):
            mu_j = conbinedmeans_list[j]
            Sigma_j = conbinedcovs_list[j]
            d = Wasserstein_table[i, j]
            idx = j * n_nodes + i
            if 0 <= d <= edge_w_max:
                l_sq = math.ceil(d / delDist) * (delDist ** 2)
                D[:, j] = l_sq
                PDF_vector, _ = interpGC_speedUp(mu_i, Sigma_i, mu_j, Sigma_j, gridXYZ, delDist)
                PDF_vector = np.sum(PDF_vector, axis=1)
                edge_pdf_columns.append(PDF_vector.astype(np.float32, copy=False))
                edge_indices.append(idx)
        D_stack.append(D)
        # 按边逐列累积，避免为所有节点预分配 (numGrid, n_nodes) 超大矩阵。
        if edge_pdf_columns:
            _pdf_cols_i = np.column_stack(edge_pdf_columns)
            _edge_idx_i = np.asarray(edge_indices, dtype=np.int64)
            total_edges += int(_edge_idx_i.size)
            del _pdf_cols_i, _edge_idx_i
        del edge_pdf_columns, edge_indices
    D = np.array(D_stack).reshape(n_nodes, n_nodes)
    print(f"Graph edges kept: {total_edges}")
    GraphA = D
    return GraphA


def init_GC_Nodes(mean_table, grid_step):
    GC_means = []
    GC_covs = []
    sigma_diag = float(grid_step) / 2.0
    for i in range(len(mean_table)):
        mu = mean_table[i]
        Sigma = [[sigma_diag, 0, 0], [0, sigma_diag, 0], [0, 0, sigma_diag]]
        GC_means.append(mu)
        GC_covs.append(Sigma)
    return GC_means, GC_covs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="预计算 rover3d_navigation config 先验文件"
    )
    # 默认路径：从 workspace 根目录运行 (cwd = fishbot_multirobot_sim)
    _cwd = os.getcwd()
    _fiesta_candidates = [
        os.path.join(_cwd, "src", "FIESTA", "config", "fiesta.yaml"),
        os.path.join(_pkg_root, "..", "FIESTA", "config", "fiesta.yaml"),
    ]
    _fiesta_default = next((p for p in _fiesta_candidates if os.path.exists(p)), _fiesta_candidates[0])
    parser.add_argument(
        "--fiesta-config",
        default=_fiesta_default,
        help="FIESTA fiesta.yaml 路径",
    )
    _planning_default = os.path.join(_pkg_root, "config", "planning.yaml")
    if not os.path.isfile(_planning_default):
        _planning_default = os.path.join(_cwd, "src", "rover3d_navigation", "config", "planning.yaml")
    parser.add_argument(
        "--planning-config",
        default=_planning_default,
        help="planning.yaml 路径（默认优先读取地图参数）",
    )
    parser.add_argument(
        "--map-source",
        choices=["auto", "planning", "fiesta"],
        default="auto",
        help="地图参数来源：auto(优先 planning)->fiesta",
    )
    _out_default = os.path.join(_pkg_root, "config")
    if not os.path.isdir(_out_default):
        _out_default = os.path.join(_cwd, "src", "rover3d_navigation", "config")
    parser.add_argument(
        "--output",
        default=_out_default,
        help="输出目录",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=2.0,
        help="GC 节点网格步长 [m]（须与生成 config 时所用步长一致）",
    )
    parser.add_argument(
        "--graph-grid-res",
        type=float,
        default=None,
        help="构图体素分辨率 [m]（默认取 esdf_resolution）",
    )
    parser.add_argument(
        "--interp-step",
        type=float,
        default=None,
        help="GC 插值步长 [m]（默认与 graph-grid-res 一致）",
    )
    args = parser.parse_args()

    # 规范化路径（支持脚本从任意目录运行）
    def _norm(p: str) -> str:
        if not os.path.isabs(p):
            p = os.path.join(os.getcwd(), p)
        return os.path.normpath(p)

    fiesta_path = _norm(args.fiesta_config)
    planning_path = _norm(args.planning_config)
    output_dir = _norm(args.output)
    grid_step = args.grid_step

    if grid_step <= 0:
        print("错误: --grid-step 必须 > 0")
        sys.exit(1)

    planning_exists = os.path.exists(planning_path)
    fiesta_exists = os.path.exists(fiesta_path)
    if args.map_source == "planning":
        if not planning_exists:
            print(f"错误: planning 配置不存在: {planning_path}")
            sys.exit(1)
        map_p = get_map_params_from_planning(planning_path)
        map_source = "planning"
    elif args.map_source == "fiesta":
        if not fiesta_exists:
            print(f"错误: FIESTA 配置不存在: {fiesta_path}")
            sys.exit(1)
        map_p = get_map_params_from_fiesta(fiesta_path)
        map_source = "fiesta"
    else:
        if planning_exists:
            map_p = get_map_params_from_planning(planning_path)
            map_source = "planning(auto)"
        elif fiesta_exists:
            map_p = get_map_params_from_fiesta(fiesta_path)
            map_source = "fiesta(auto-fallback)"
        else:
            print("错误: planning 与 FIESTA 配置均不存在")
            print(f"  planning: {planning_path}")
            print(f"  fiesta  : {fiesta_path}")
            sys.exit(1)

    graph_grid_res = (
        float(args.graph_grid_res)
        if args.graph_grid_res is not None
        else float(map_p.get("esdf_resolution", 0.2))
    )
    interp_step = (
        float(args.interp_step)
        if args.interp_step is not None
        else graph_grid_res
    )
    if graph_grid_res <= 0 or interp_step <= 0:
        print("错误: --graph-grid-res 和 --interp-step 必须 > 0")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取地图参数
    xa = map_p["map_origin_x"]
    ya = map_p["map_origin_y"]
    za = map_p["map_origin_z"]
    xb = xa + map_p["map_size_x"]
    yb = ya + map_p["map_size_y"]
    zb = za + map_p["map_size_z"]
    print(f"地图参数来源: {map_source}")
    print(f"地图边界: x=[{xa}, {xb}], y=[{ya}, {yb}], z=[{za}, {zb}]")
    print(f"构图分辨率: {graph_grid_res}, 插值步长: {interp_step}, GC步长: {grid_step}")

    mean_table = []
    center_offset = grid_step / 2.0
    for i in np.arange(xa, xb, grid_step):
        for j in np.arange(ya, yb, grid_step):
            for k in np.arange(za, zb, grid_step):
                mean_table.append(
                    [float(i + center_offset), float(j + center_offset), float(k + center_offset)]
                )
    if len(mean_table) == 0:
        mean_table = [[(xa + xb) / 2, (ya + yb) / 2, (za + zb) / 2]]

    GC_means, GC_covs = init_GC_Nodes(mean_table, grid_step)
    print(f"GC 节点数: {len(GC_means)} （纯离散节点，不含目标点）")

    # 3. 使用 GC 节点构建图（目标点运行时归并到最近 GC）
    conbinedmeans_list = list(GC_means)
    conbinedcovs_list = []
    for c in GC_covs:
        arr = np.array(c) if not hasattr(c, "shape") else c
        conbinedcovs_list.append(arr.reshape(3, 3) if np.size(arr) == 9 else arr)

    Numnode = len(conbinedmeans_list)
    print(f"总节点数: {Numnode}")

    # 4. 计算 Wasserstein 与 Node_PDF 表
    print("计算 Wasserstein 与 Node_PDF 表...")
    Wasserstein_table = np.zeros((Numnode, Numnode))
    Node_PDF_table = np.zeros((Numnode, Numnode))
    for i in range(Numnode):
        for j in range(Numnode):
            m1 = conbinedmeans_list[i]
            c1 = conbinedcovs_list[i]
            if not hasattr(c1, "shape"):
                c1 = np.array(c1).reshape(3, 3)
            m2 = conbinedmeans_list[j]
            c2 = conbinedcovs_list[j]
            if not hasattr(c2, "shape"):
                c2 = np.array(c2).reshape(3, 3)
            Wasserstein_table[i, j] = Wasserstein_distance(m1, c1, m2, c2)
            Node_PDF_table[i, j] = multivariate_normal.pdf(
                m1, mean=np.array(m2), cov=c2
            )

    # 5. 构建 Graph_GC（邻接矩阵）
    print("构建 Graph_GC...")
    Graph_adj = init_Graph_GC(
        conbinedmeans_list, conbinedcovs_list, Wasserstein_table,
        xa=xa, ya=ya, za=za, xb=xb, yb=yb, zb=zb,
        graph_grid_res=graph_grid_res,
        delDist=interp_step,
    )

    # 6. 保存
    path_means = os.path.join(output_dir, "GC_means_3D.json")
    path_covs = os.path.join(output_dir, "GC_covs_3D.json")
    path_w = os.path.join(output_dir, "Wasserstein_table_3D.npy")
    path_pdf = os.path.join(output_dir, "Node_PDF_table_3D.npy")
    path_graph = os.path.join(output_dir, "Graph_GC_3D.npy")

    with open(path_means, "w", encoding="utf-8") as f:
        json.dump(GC_means, f, indent=None)
    with open(path_covs, "w", encoding="utf-8") as f:
        covs_flat = [
            np.array(c).reshape(3, 3).flatten().tolist()
            if hasattr(c, "shape") else np.array(c).flatten().tolist()
            for c in GC_covs
        ]
        json.dump(covs_flat, f, indent=None)
    np.save(path_w, Wasserstein_table)
    np.save(path_pdf, Node_PDF_table)
    np.save(path_graph, Graph_adj)

    print(f"已保存到 {output_dir}:")
    print(f"  - GC_means_3D.json ({len(GC_means)} 节点)")
    print(f"  - GC_covs_3D.json")
    print(f"  - Wasserstein_table_3D.npy ({Numnode}x{Numnode})")
    print(f"  - Node_PDF_table_3D.npy ({Numnode}x{Numnode})")
    print(f"  - Graph_GC_3D.npy ({Numnode}x{Numnode})")
    print("完成。")


if __name__ == "__main__":
    main()
