# Planner Utilities

当前目录已移除 RRT 规划实现，保留通用规划工具组件：

- `collision_checker.py`：MuJoCo 碰撞检测
- `path_smoother.py`：路径平滑与轨迹参数化
- `joint_interpolator.py`：关节插值（cubic/quintic）
- `planner_utils.py`：通用路径工具函数（距离、插值、重采样等）

## 说明

- 不再提供 `RRT` / `RRT*` / `RRT-Connect` 规划器实现与示例。
- 若后续需要重新引入全局路径规划器，建议新增独立模块，避免与当前控制链路耦合。
