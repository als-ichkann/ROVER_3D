# rover3d_navigation

多机三维规划包：GMM 宏观轨迹 + 微观轨迹生成，发布 `nav_msgs/Path`（与底层跟踪控制解耦）。

## 目录结构

- `src/` — `planning_node.py`；GMM 目标可用 `scripts/publish_gmm_goal.py` 测试发布
- `rover3d_navigation/` — 库：`ROVER_3D`、`Planning_3D`、`apf_control_law`、`esdf_adapter` 等
- `config/` — `planning.yaml`、`gmm_goal_publisher.yaml` 等
- `launch/` — `planning.launch.py`

## 规划流程

1. 外部发布目标 GMM（`navigation_msgs/GMM`）
2. `PlanningProcess` 做 SLP 宏观优化 + GMM 插值
3. 微观轨迹生成器（`micro_controller`: `apf` / `density`）输出每机器人轨迹

## 启动

```bash
source install/setup.bash
ros2 launch rover3d_navigation planning.launch.py
```

## 依赖

`navigation_msgs`、`esdf_map`、`python3-numpy`、`python3-scipy`、`python3-sklearn`
