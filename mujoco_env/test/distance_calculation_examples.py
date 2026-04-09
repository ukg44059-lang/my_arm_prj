"""
机械臂与障碍物距离计算 - 使用示例
Robot-Obstacle Distance Calculation - Usage Example

这个文档展示如何使用新添加的距离计算功能
"""

import numpy as np
from mujoco_env.robot.robot import Robot

# ============================================================
# 示例 1: 基本使用
# ============================================================

def example_basic_usage():
    """基本距离计算示例"""
    print("示例 1: 基本使用")
    print("=" * 60)

    # 初始化机器人
    robot = Robot("../robot_model/exp/env_robot_torque.xml",
                  enable_cameras=False)

    # 设置关节位置
    robot.data.qpos[:6] = np.deg2rad([0, 0, -90, 0, -90, 0])
    robot.robot.forward_kinematics()

    # 计算最近距离
    distance_info = robot.get_minimum_distance_to_obstacles()

    # 显示结果
    print(f"最近距离: {distance_info['min_distance']*1000:.2f} mm")
    print(f"最近障碍物: {distance_info['obstacle_name']}")
    print(f"机械臂几何体: {distance_info['robot_geom']}")
    print(f"障碍物几何体: {distance_info['obstacle_geom']}")

    if distance_info['robot_point'] is not None:
        print(f"机械臂最近点: {distance_info['robot_point']}")
        print(f"障碍物最近点: {distance_info['obstacle_point']}")

    robot.cleanup()
    print()


# ============================================================
# 示例 2: 指定特定障碍物
# ============================================================

def example_specific_obstacles():
    """指定特定障碍物的距离计算"""
    print("示例 2: 指定特定障碍物")
    print("=" * 60)

    robot = Robot("mujoco_env/robot_model/exp/env_robot_torque.xml",
                  enable_cameras=False)

    robot.data.qpos[:6] = np.deg2rad([0, 0, -90, 0, -90, 0])
    robot.forward_kinematics()

    # 只计算与obstacle1的距离
    distance_info = robot.get_minimum_distance_to_obstacles(
        obstacle_names=['obstacle1']
    )

    print(f"与obstacle1的最近距离: {distance_info['min_distance']*1000:.2f} mm")

    # 只计算与obstacle2的距离
    distance_info = robot.get_minimum_distance_to_obstacles(
        obstacle_names=['obstacle2']
    )

    print(f"与obstacle2的最近距离: {distance_info['min_distance']*1000:.2f} mm")

    robot.cleanup()
    print()


# ============================================================
# 示例 3: 在仿真循环中使用
# ============================================================

def example_in_simulation_loop():
    """在仿真循环中使用距离计算"""
    print("示例 3: 在仿真循环中使用")
    print("=" * 60)

    robot = Robot("mujoco_env/robot_model/exp/env_robot_torque.xml",
                  enable_cameras=False)

    robot.set_control_mode("zero_gravity")

    # 仿真循环
    for step in range(100):
        # 更新机器人状态
        robot.update_joint_state()

        # 计算控制
        joint_targets = robot.compute_joint_targets(robot.sim_time)
        robot.apply_joint_control(joint_targets)

        # 更新障碍物
        robot.update_obstacles()

        # 执行仿真步
        robot.step()

        # 每10步计算一次距离
        if step % 10 == 0:
            distance_info = robot.get_minimum_distance_to_obstacles()

            print(f"步骤 {step}: 距离 = {distance_info['min_distance']*1000:.1f} mm, "
                  f"障碍物 = {distance_info['obstacle_name']}")

            # 检查安全距离
            if distance_info['min_distance'] < 0.05:  # 小于50mm
                print("  ⚠ 警告: 距离过近!")

    robot.cleanup()
    print()


# ============================================================
# 示例 4: 碰撞预警系统
# ============================================================

def example_collision_warning_system():
    """碰撞预警系统示例"""
    print("示例 4: 碰撞预警系统")
    print("=" * 60)

    robot = Robot("mujoco_env/robot_model/exp/env_robot_torque.xml",
                  enable_cameras=False)

    # 定义安全阈值
    DANGER_THRESHOLD = 0.03   # 30mm - 危险
    WARNING_THRESHOLD = 0.10  # 100mm - 警告
    SAFE_THRESHOLD = 0.20     # 200mm - 安全

    robot.data.qpos[:6] = np.deg2rad([0, 0, -90, 0, -90, 0])
    robot.forward_kinematics()

    # 获取距离信息
    distance_info = robot.get_minimum_distance_to_obstacles()
    distance_m = distance_info['min_distance']
    distance_mm = distance_m * 1000

    # 安全评估
    if distance_m < DANGER_THRESHOLD:
        status = "🔴 危险"
        message = "立即停止! 碰撞风险极高!"
    elif distance_m < WARNING_THRESHOLD:
        status = "🟡 警告"
        message = "小心! 接近障碍物"
    elif distance_m < SAFE_THRESHOLD:
        status = "🟢 注意"
        message = "保持警惕"
    else:
        status = "✅ 安全"
        message = "安全距离"

    print(f"安全状态: {status}")
    print(f"距离: {distance_mm:.1f} mm")
    print(f"最近障碍物: {distance_info['obstacle_name']}")
    print(f"提示: {message}")

    robot.cleanup()
    print()


# ============================================================
# 示例 5: 返回值详解
# ============================================================

def example_return_value_details():
    """返回值详细说明"""
    print("示例 5: 返回值详解")
    print("=" * 60)

    robot = Robot("mujoco_env/robot_model/exp/env_robot_torque.xml",
                  enable_cameras=False)

    robot.data.qpos[:6] = np.deg2rad([0, 0, -90, 0, -90, 0])
    robot.forward_kinematics()

    distance_info = robot.get_minimum_distance_to_obstacles()

    print("返回值字典包含以下键:")
    print(f"  - min_distance: {distance_info['min_distance']} (米)")
    print(f"  - obstacle_name: {distance_info['obstacle_name']}")
    print(f"  - robot_geom: {distance_info['robot_geom']}")
    print(f"  - obstacle_geom: {distance_info['obstacle_geom']}")
    print(f"  - robot_point: {distance_info['robot_point']}")
    print(f"  - obstacle_point: {distance_info['obstacle_point']}")

    print("\n说明:")
    print("  - min_distance: 最小距离,单位为米")
    print("  - obstacle_name: 最近的障碍物名称")
    print("  - robot_geom: 机械臂上最近点所在的几何体名称")
    print("  - obstacle_geom: 障碍物上最近点所在的几何体名称")
    print("  - robot_point: 机械臂上最近点的3D坐标 [x, y, z]")
    print("  - obstacle_point: 障碍物上最近点的3D坐标 [x, y, z]")

    robot.cleanup()
    print()


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("机械臂与障碍物距离计算 - 使用示例")
    print("Robot-Obstacle Distance Calculation - Examples")
    print("=" * 60 + "\n")

    # 运行所有示例
    example_basic_usage()
    example_specific_obstacles()
    example_in_simulation_loop()
    example_collision_warning_system()
    example_return_value_details()

    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
