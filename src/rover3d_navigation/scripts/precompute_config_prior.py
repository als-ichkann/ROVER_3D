#!/usr/bin/env python3
"""
预计算 rover3d_navigation 的 config 先验文件。

根据 FIESTA 的地图配置（lx, ly, lz, rx, ry, rz）生成纯离散 GC 节点图，不包含目标点。
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
  --output         输出目录
  --grid-step      GC 节点网格步长 (默认 2.0，需与 planning_apf.yaml 一致)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
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


def load_yaml(path: str) -> dict:
    if yaml is None:
        raise ImportError("PyYAML required. Install: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_map_params_from_fiesta(fiesta_config_path: str) -> dict:
    """从 FIESTA fiesta.yaml 读取地图参数 (lx, ly, lz, rx, ry, rz)。"""
    data = load_yaml(fiesta_config_path)
    params = data.get("fiesta_node", {}).get("ros__parameters", data)
    lx = float(params.get("lx", -5.0))
    ly = float(params.get("ly", -7.5))
    lz = float(params.get("lz", 0.0))
    rx = float(params.get("rx", 17.0))
    ry = float(params.get("ry", 9.5))
    rz = float(params.get("rz", 6.0))
    return {
        "map_origin_x": lx,
        "map_origin_y": ly,
        "map_origin_z": lz,
        "map_size_x": rx - lx,
        "map_size_y": ry - ly,
        "map_size_z": rz - lz,
    }


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
        help="GC 节点网格步长 [m]，与 planning_apf grid_step 一致",
    )
    args = parser.parse_args()

    # 规范化路径（支持脚本从任意目录运行）
    def _norm(p: str) -> str:
        if not os.path.isabs(p):
            p = os.path.join(os.getcwd(), p)
        return os.path.normpath(p)

    fiesta_path = _norm(args.fiesta_config)
    output_dir = _norm(args.output)
    grid_step = args.grid_step

    if not os.path.exists(fiesta_path):
        print(f"错误: FIESTA 配置不存在: {fiesta_path}")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 从 FIESTA 配置读取地图参数
    map_p = get_map_params_from_fiesta(fiesta_path)
    xa = map_p["map_origin_x"]
    ya = map_p["map_origin_y"]
    za = map_p["map_origin_z"]
    xb = xa + map_p["map_size_x"]
    yb = ya + map_p["map_size_y"]
    zb = za + map_p["map_size_z"]
    print(f"地图边界: x=[{xa}, {xb}], y=[{ya}, {yb}], z=[{za}, {zb}]")

    # 2. 生成离散 GC 节点（不包含目标点，目标在运行时归并到最近 GC）
    from rover3d_navigation import init_Graph_CVaR_3D

    mean_table = []
    for i in np.arange(xa, xb, grid_step):
        for j in np.arange(ya, yb, grid_step):
            for k in np.arange(za, zb, grid_step):
                mean_table.append([float(i + 0.5), float(j + 0.5), float(k + 0.5)])
    if len(mean_table) == 0:
        mean_table = [[(xa + xb) / 2, (ya + yb) / 2, (za + zb) / 2]]

    GC_means, GC_covs = init_Graph_CVaR_3D.init_GC_Nodes(mean_table)
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
    from rover3d_navigation import control_law_3D

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
            Wasserstein_table[i, j] = control_law_3D.Wasserstein_distance(m1, c1, m2, c2)
            Node_PDF_table[i, j] = multivariate_normal.pdf(
                m1, mean=np.array(m2), cov=c2
            )

    # 5. 构建 Graph_GC（邻接矩阵）
    print("构建 Graph_GC...")
    Graph_adj = init_Graph_CVaR_3D.init_Graph_GC(
        conbinedmeans_list, conbinedcovs_list, Wasserstein_table,
        xa=xa, ya=ya, za=za, xb=xb, yb=yb, zb=zb,
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
