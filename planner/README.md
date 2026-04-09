# RRT Path Planning Library for Robot Arm

机械臂RRT路径规划算法库 - 基于纯Python实现，完美适配MuJoCo仿真环境

## 📚 功能特性

### 核心算法
- ✅ **RRT** - 基础快速探索随机树
- ✅ **RRT*** - 渐进最优路径规划
- ✅ **RRT-Connect** - 双向快速连接
- ✅ **Informed RRT*** - 椭球采样优化
- ✅ **Bidirectional RRT** - 双向搜索

### 关键功能
- ✅ 基于MuJoCo的精确碰撞检测
- ✅ 高效KD-Tree最近邻搜索
- ✅ 多种路径平滑算法
- ✅ 轨迹时间参数化
- ✅ 安全边距碰撞检测
- ✅ 路径优化和简化

## 🏗️ 项目结构

```
mujoco_env/robot/planner/
├── __init__.py              # 模块导出
├── planner_utils.py         # 工具函数（配置空间、RRT树、距离度量等）
├── collision_checker.py     # MuJoCo碰撞检测器
├── rrt_base.py             # 基础RRT实现
├── rrt_variants.py         # RRT变体（RRT*、RRT-Connect等）
├── path_smoother.py        # 路径平滑和轨迹参数化
├── example_usage.py        # 使用示例
└── README.md              # 本文档
```

## 🚀 快速开始

### 1. 基础RRT路径规划

```python
import numpy as np
import mujoco
from planner import (
    ConfigurationSpace,
    MuJoCoCollisionChecker,
    RRTPlanner
)

# 加载MuJoCo模型
model = mujoco.MjModel.from_xml_path("path/to/robot.xml")
data = mujoco.MjData(model)

# 定义关节空间（6自由度机械臂）
joint_limits = np.array([
    [-2.96706, 2.96706],  # Joint 1
    [-2.09440, 2.09440],  # Joint 2
    [-2.96706, 2.96706],  # Joint 3
    [-2.09440, 2.09440],  # Joint 4
    [-2.96706, 2.96706],  # Joint 5
    [-2.09440, 2.09440],  # Joint 6
])

config_space = ConfigurationSpace(joint_limits)

# 初始化碰撞检测器
collision_checker = MuJoCoCollisionChecker(
    model=model,
    data=data,
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    exclude_bodies=['floor']  # 排除地面
)

# 创建RRT规划器
planner = RRTPlanner(
    config_space=config_space,
    collision_checker=collision_checker,
    max_step_size=0.3,
    goal_bias=0.1,
    goal_tolerance=0.15
)

# 规划路径
start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
goal = np.array([1.5, 0.5, -0.5, 0.3, 0.2, 0.1])

path, info = planner.plan(
    start_config=start,
    goal_config=goal,
    max_iterations=5000,
    max_time=30.0
)

if info['success']:
    print(f"✓ 路径规划成功！")
    print(f"  - 规划时间: {info['planning_time']:.3f}s")
    print(f"  - 路径长度: {info['path_length']:.3f} rad")
    print(f"  - 路径点数: {len(path)}")
else:
    print(f"✗ 规划失败: {info.get('error')}")
```

### 2. 使用RRT*获得最优路径

```python
from planner import RRTStarPlanner

planner = RRTStarPlanner(
    config_space=config_space,
    collision_checker=collision_checker,
    max_step_size=0.25,
    rewire_factor=2.0  # 重连接半径因子
)

path, info = planner.plan(start, goal, max_iterations=5000)

if info['success']:
    print(f"✓ 最优路径cost: {info['path_cost']:.3f}")
```

### 3. 使用RRT-Connect快速规划

```python
from planner import RRTConnectPlanner

planner = RRTConnectPlanner(
    config_space=config_space,
    collision_checker=collision_checker,
    max_step_size=0.3
)

# RRT-Connect通常更快
path, info = planner.plan(start, goal, max_iterations=2000, max_time=10.0)
```

### 4. 路径平滑

```python
from planner import PathSmoother, compute_path_statistics

# 创建平滑器
smoother = PathSmoother(collision_checker=collision_checker)

# 快捷平滑（移除不必要的路径点）
smoothed = smoother.smooth_shortcut(
    path,
    max_iterations=100,
    collision_check=True
)

# 样条平滑
spline_smoothed = smoother.smooth_cubic_spline(
    path,
    num_samples=100
)

# B样条平滑
bspline_smoothed = smoother.smooth_b_spline(
    path,
    num_samples=100,
    smoothness=0.0
)

# 计算路径统计
stats = compute_path_statistics(smoothed)
print(f"路径长度: {stats['length']:.3f} rad")
print(f"平均段长: {stats['avg_segment_length']:.3f} rad")
```

### 5. 轨迹参数化

```python
from planner import TrajectoryParameterizer

# 创建参数化器
parameterizer = TrajectoryParameterizer()

# 梯形速度规划
traj_path, timestamps = parameterizer.parameterize_trapezoidal(
    path=smoothed,
    max_vel=1.0,      # rad/s
    max_acc=2.0       # rad/s^2
)

print(f"轨迹总时长: {timestamps[-1]:.3f}s")

# 插值到高分辨率（如100Hz）
interp_path, interp_times = parameterizer.interpolate_trajectory(
    path=traj_path,
    timestamps=timestamps,
    dt=0.01  # 10ms采样
)

print(f"插值后路径点: {len(interp_path)}")
```

## 📊 算法对比

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **RRT** | 快速、简单 | 路径非最优 | 快速原型、简单场景 |
| **RRT*** | 渐进最优、路径质量高 | 速度较慢 | 需要高质量路径 |
| **RRT-Connect** | 速度最快、双向搜索 | 路径非最优 | 简单场景、实时规划 |
| **Informed RRT*** | 收敛快、路径最优 | 需要初始解 | 复杂场景优化 |

## 🔧 高级功能

### 安全边距碰撞检测

```python
from planner import SafetyMarginCollisionChecker

# 添加安全边距（例如1cm）
safe_checker = SafetyMarginCollisionChecker(
    base_checker=collision_checker,
    safety_margin=0.01  # meters
)

planner = RRTPlanner(
    config_space=config_space,
    collision_checker=safe_checker,
    # ...
)
```

### 自定义距离度量

```python
from planner import distance_metric

# 加权距离（给不同关节不同权重）
weights = np.array([1.0, 1.0, 0.8, 0.6, 0.5, 0.5])

dist = distance_metric(config1, config2, weights=weights)
```

### 获取碰撞详情

```python
# 检查特定配置的碰撞信息
info = collision_checker.get_collision_info_at_config(config)

print(f"是否碰撞: {info['in_collision']}")
print(f"碰撞点数: {info['num_contacts']}")

for contact in info['contacts']:
    print(f"  - {contact['body1']} <-> {contact['body2']}")
    print(f"    穿透深度: {contact['penetration']:.4f}m")
```

### 获取最小间隙

```python
# 获取与障碍物的最小距离
clearance = collision_checker.get_clearance(config)

if clearance < 0:
    print("配置发生碰撞！")
elif clearance < 0.05:
    print(f"警告：距离障碍物仅 {clearance*100:.1f}cm")
```

## 🎯 完整示例

运行示例脚本查看所有功能：

```bash
cd mujoco_env/robot/planner
python example_usage.py
```

示例包括：
1. 基础RRT规划
2. RRT*最优规划
3. RRT-Connect快速规划
4. 路径平滑和参数化
5. 所有算法对比

## 📖 API参考

### ConfigurationSpace
```python
config_space = ConfigurationSpace(joint_limits)
random_config = config_space.sample_random_config()
is_valid = config_space.is_config_valid(config)
```

### MuJoCoCollisionChecker
```python
checker = MuJoCoCollisionChecker(
    model=model,
    data=data,
    joint_names=["joint1", ...],
    exclude_bodies=['floor']
)

collision = checker.check_collision_at_config(config)
collision_free = checker.check_segment_collision_free(config1, config2)
```

### RRTPlanner / RRTStarPlanner / RRTConnectPlanner
```python
planner = RRTPlanner(
    config_space=config_space,
    collision_checker=collision_checker,
    max_step_size=0.3,
    goal_bias=0.1,
    goal_tolerance=0.15
)

path, info = planner.plan(
    start_config=start,
    goal_config=goal,
    max_iterations=5000,
    max_time=30.0
)
```

### PathSmoother
```python
smoother = PathSmoother(collision_checker=collision_checker)

# 方法
smoothed = smoother.smooth_shortcut(path, max_iterations=100)
smoothed = smoother.smooth_cubic_spline(path, num_samples=100)
smoothed = smoother.smooth_b_spline(path, num_samples=100)
smoothed = smoother.smooth_bezier(path, num_samples=100)
```

## 🔍 性能优化建议

1. **调整步长**：较大的`max_step_size`可以加快探索，但可能错过狭窄通道
2. **目标偏向**：提高`goal_bias`可以更快找到解，但可能降低路径质量
3. **树节点数**：RRT*需要更多节点才能收敛到最优解
4. **碰撞检测分辨率**：减小`step_size`参数提高精度但降低速度
5. **算法选择**：
   - 简单场景 → RRT-Connect
   - 复杂场景需要好路径 → RRT*
   - 已有解需要优化 → Informed RRT*

## 🐛 调试技巧

```python
# 1. 检查配置空间是否合理
print(f"Joint limits: {config_space.joint_limits}")

# 2. 验证起点和终点
print(f"Start valid: {config_space.is_config_valid(start)}")
print(f"Goal valid: {config_space.is_config_valid(goal)}")

# 3. 检查起点/终点是否有碰撞
print(f"Start collision: {collision_checker.check_collision_at_config(start)}")
print(f"Goal collision: {collision_checker.check_collision_at_config(goal)}")

# 4. 可视化树节点（用于理解探索过程）
tree_nodes = planner.get_tree_nodes()
tree_edges = planner.get_tree_edges()
```

## 🎓 与MoveIt对比

| 功能 | 本库 | MoveIt |
|------|------|--------|
| RRT算法 | ✅ | ✅ |
| RRT* | ✅ | ✅ |
| RRT-Connect | ✅ | ✅ |
| 碰撞检测 | ✅ MuJoCo | ✅ FCL |
| 轨迹优化 | ✅ | ✅ |
| 依赖 | 仅numpy+scipy | ROS + 多个库 |
| 安装难度 | ⭐ 简单 | ⭐⭐⭐⭐ 复杂 |
| Python兼容 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 可定制性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 📦 依赖项

- `numpy` - 数值计算
- `scipy` - KD-Tree和样条插值
- `mujoco` - 物理仿真和碰撞检测

全部都已包含在你的环境中！

## 🤝 集成到你的项目

在你的机器人控制代码中使用：

```python
from mujoco_env.robot.planner import (
    ConfigurationSpace,
    MuJoCoCollisionChecker,
    RRTConnectPlanner,
    PathSmoother
)

# 在Robot类中添加路径规划方法
class Robot:
    def plan_path_to_config(self, goal_config):
        # 使用RRT规划
        planner = RRTConnectPlanner(...)
        path, info = planner.plan(
            start_config=self.get_joint_positions(),
            goal_config=goal_config
        )

        # 平滑路径
        smoother = PathSmoother()
        smoothed = smoother.smooth_shortcut(path)

        return smoothed, info
```

## 📝 许可证

本库作为MuJoCo Robot Arm项目的一部分，遵循项目许可证。

## 🙏 致谢

- 基于RRT、RRT*、RRT-Connect等经典路径规划算法
- 使用MuJoCo物理引擎进行高精度碰撞检测
- 参考了MoveIt和OMPL的设计理念

---

**需要帮助？** 查看 `example_usage.py` 获取完整示例代码！
