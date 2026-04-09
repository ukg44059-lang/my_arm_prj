"""
Interactive Joint Space RRT-Connect Planning
交互式关节空间 RRT-Connect 规划

完整流程：
1. 在 MuJoCo Viewer 中拖动 target_body 设置目标位置
2. 自动检测目标位置变化
3. 通过 IK 求解目标关节角度
4. 通过 RRT-Connect 在关节空间搜索路径
5. 插值并执行规划的路径
6. 循环：等待新的目标位置
"""

import numpy as np
import mujoco
import mujoco.viewer
import os
import sys
import time

# Add planner to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rrt_variants import RRTConnectPlanner
from planner_utils import ConfigurationSpace
from collision_checker import MuJoCoCollisionChecker
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'arm_ik'))
from arm_ik import ArmInverseKinematics


def get_ee_pose(model, data, ee_site_name='tools_link'):
    """获取当前末端执行器位姿"""
    site_id = model.site(ee_site_name).id
    pos = data.site_xpos[site_id].copy()
    mat = data.site_xmat[site_id].reshape(3, 3).copy()

    pose = np.eye(4)
    pose[:3, :3] = mat
    pose[:3, 3] = pos
    return pose


def set_joint_positions(model, data, joint_names, positions):
    """设置关节位置"""
    for name, pos in zip(joint_names, positions):
        joint_id = model.joint(name).id
        qpos_addr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_addr] = pos


def compute_fk_trajectory(model, data, joint_path, joint_names, ee_site_name='tools_link'):
    """
    通过正运动学计算路径上每个关节配置对应的末端执行器位置

    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        joint_path: 关节空间路径点列表
        joint_names: 关节名称列表
        ee_site_name: 末端执行器site名称

    Returns:
        ee_positions: 末端执行器位置数组 (N, 3)
        ee_poses: 末端执行器位姿数组 (N, 4, 4)
    """
    ee_positions = []
    ee_poses = []

    for joint_config in joint_path:
        # 设置关节配置
        set_joint_positions(model, data, joint_names, joint_config)
        # 前向运动学
        mujoco.mj_forward(model, data)

        # 获取末端执行器位姿
        pose = get_ee_pose(model, data, ee_site_name)
        ee_poses.append(pose)
        ee_positions.append(pose[:3, 3])

    return np.array(ee_positions), ee_poses


def solve_ik_for_target(ik_solver, target_pose, seed_config, collision_checker, max_attempts=10):
    """
    为目标位姿求解IK，支持多次尝试

    Args:
        ik_solver: IK求解器
        target_pose: 目标位姿 (4x4矩阵)
        seed_config: 初始种子配置
        collision_checker: 碰撞检测器
        max_attempts: 最大尝试次数

    Returns:
        success: 是否成功
        joint_config: 关节配置（如果成功）
    """
    for attempt in range(max_attempts):
        # 添加随机扰动以尝试不同的IK解
        if attempt == 0:
            current_seed = seed_config
        else:
            # 在种子配置周围随机采样
            noise = np.random.uniform(-0.3, 0.3, size=seed_config.shape)
            current_seed = seed_config + noise

        # 求解IK
        success, joint_config = ik_solver.get_ik(target_pose, current_seed)

        if not success:
            continue

        # 检查碰撞
        if collision_checker.check_collision_at_config(joint_config):
            print(f"  ⚠ IK解存在碰撞 (尝试 {attempt + 1}/{max_attempts})")
            continue

        print(f"  ✓ IK求解成功 (尝试 {attempt + 1}/{max_attempts})")
        return True, joint_config

    return False, None


def interactive_target_adjustment(model, data, joint_names, ik_solver, collision_checker,
                                 start_joint_config, target_body_id=None, ee_site_name='tools_link'):
    """
    交互式目标位置调整模式
    允许用户在可视化窗口中移动 target_body，实时显示IK求解状态

    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        joint_names: 关节名称列表
        ik_solver: IK求解器
        collision_checker: 碰撞检测器
        start_joint_config: 起始关节配置
        target_body_id: target_body的ID（如果存在）
        ee_site_name: 末端执行器site名称
    """
    print("\n🎮 交互式目标调整模式")
    print("  - 在MuJoCo Viewer中用鼠标拖动 target_body (红色球体)")
    print("  - 程序会实时检测目标位置是否可达")
    print("  - 绿色 = IK可解且无碰撞 ✓")
    print("  - 红色 = IK失败或有碰撞 ✗")
    print("  - 按 'Ctrl+C' 退出\n")

    # 重置到起始配置
    set_joint_positions(model, data, joint_names, start_joint_config)
    mujoco.mj_forward(model, data)

    last_check_time = time.time()
    check_interval = 0.1  # 每0.1秒检查一次IK

    last_ik_success = False
    last_target_pos = None

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("✓ Viewer已启动，等待目标位置调整...\n")

            step_count = 0

            while viewer.is_running():
                current_time = time.time()

                # 定期检查IK
                if current_time - last_check_time >= check_interval:
                    last_check_time = current_time

                    if target_body_id is not None:
                        # 读取当前目标位置和姿态
                        target_pos = data.xpos[target_body_id].copy()
                        target_quat = data.xquat[target_body_id].copy()

                        # 转换为位姿矩阵
                        target_orientation = np.zeros((3, 3))
                        mujoco.mju_quat2Mat(target_orientation.flatten(), target_quat)
                        target_orientation = target_orientation.reshape(3, 3)

                        goal_pose = np.eye(4)
                        goal_pose[:3, :3] = target_orientation
                        goal_pose[:3, 3] = target_pos

                        # 尝试IK求解
                        ik_success, joint_config = solve_ik_for_target(
                            ik_solver=ik_solver,
                            target_pose=goal_pose,
                            seed_config=start_joint_config,
                            collision_checker=collision_checker,
                            max_attempts=5  # 减少尝试次数以提高响应速度
                        )

                        # 检测位置变化
                        position_changed = (last_target_pos is None or
                                          np.linalg.norm(target_pos - last_target_pos) > 0.001)

                        # 只在状态改变或位置显著变化时打印
                        if (ik_success != last_ik_success) or position_changed:
                            status_symbol = "✓" if ik_success else "✗"
                            status_color = "🟢" if ik_success else "🔴"

                            print(f"{status_color} 目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] | IK: {status_symbol}")

                            if ik_success:
                                # 显示求解的关节配置
                                joint_deg = np.rad2deg(joint_config)
                                print(f"   关节角度(°): [{joint_deg[0]:6.1f}, {joint_deg[1]:6.1f}, {joint_deg[2]:6.1f}, "
                                      f"{joint_deg[3]:6.1f}, {joint_deg[4]:6.1f}, {joint_deg[5]:6.1f}]")

                            last_ik_success = ik_success
                            last_target_pos = target_pos.copy()

                        # 可视化：绘制目标点（颜色表示IK状态）
                        viewer.user_scn.ngeom = 0

                        # 绘制目标球体
                        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                            target_color = [0, 1, 0, 0.8] if ik_success else [1, 0, 0, 0.8]
                            mujoco.mjv_initGeom(
                                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[0.02, 0, 0],
                                pos=target_pos,
                                mat=np.eye(3).flatten(),
                                rgba=target_color
                            )
                            viewer.user_scn.ngeom += 1

                        # 绘制当前末端位置到目标的连接线
                        ee_pos = get_ee_pose(model, data, ee_site_name)[:3, 3]
                        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                            line_color = [0, 1, 0, 0.5] if ik_success else [1, 0, 0, 0.5]
                            mujoco.mjv_connector(
                                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                type=mujoco.mjtGeom.mjGEOM_LINE,
                                width=0.002,
                                from_=ee_pos,
                                to=target_pos
                            )
                            viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = line_color
                            viewer.user_scn.ngeom += 1

                viewer.sync()
                step_count += 1
                time.sleep(0.01)  # 100Hz

    except KeyboardInterrupt:
        print("\n⏹ 用户中断调整模式")

    print("\n✓ 交互式调整结束")
    if last_ik_success and last_target_pos is not None:
        print(f"\n✅ 最后一个有效目标位置: [{last_target_pos[0]:.3f}, {last_target_pos[1]:.3f}, {last_target_pos[2]:.3f}]")
        print("💡 提示: 你可以在XML文件中将 target_body 的位置设置为上述坐标，然后重新运行程序")
    else:
        print("\n⚠️  未找到有效目标位置")


def visualize_trajectory(model, data, joint_path, ee_positions, joint_names, ee_site_name='tools_link'):
    """
    可视化机械臂轨迹

    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        joint_path: 关节空间路径
        ee_positions: 末端执行器位置轨迹
        joint_names: 关节名称列表
        ee_site_name: 末端执行器site名称
    """
    print("\n" + "="*70)
    print("🎨 轨迹可视化")
    print("="*70)

    # 计算路径统计信息
    path_length = 0.0
    for i in range(len(ee_positions) - 1):
        path_length += np.linalg.norm(ee_positions[i+1] - ee_positions[i])

    print(f"\n📊 轨迹统计:")
    print(f"  - 路径点数量: {len(joint_path)}")
    print(f"  - 末端轨迹长度: {path_length:.4f} m")
    print(f"  - 平均段长: {path_length/(len(ee_positions)-1):.4f} m")
    print(f"  - 起点: [{ee_positions[0][0]:.3f}, {ee_positions[0][1]:.3f}, {ee_positions[0][2]:.3f}]")
    print(f"  - 终点: [{ee_positions[-1][0]:.3f}, {ee_positions[-1][1]:.3f}, {ee_positions[-1][2]:.3f}]")

    print("\n🎬 启动可视化窗口...")
    print("机械臂将沿着规划的轨迹运动")
    print("按 'Ctrl+C' 停止\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 重置到起点
            set_joint_positions(model, data, joint_names, joint_path[0])
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0)

            # 动画循环
            for loop in range(3):  # 循环3次
                print(f"\n▶ 循环 {loop + 1}/3")

                # 正向执行
                for i, joint_config in enumerate(joint_path):
                    if not viewer.is_running():
                        break

                    set_joint_positions(model, data, joint_names, joint_config)
                    mujoco.mj_forward(model, data)
                    viewer.sync()

                    # 打印进度
                    if i % max(1, len(joint_path) // 10) == 0:
                        progress = (i / len(joint_path)) * 100
                        current_pos = get_ee_pose(model, data, ee_site_name)[:3, 3]
                        print(f"  进度: {progress:5.1f}% | 位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")

                    time.sleep(0.02)  # 50Hz

                if not viewer.is_running():
                    break

                # 在终点暂停
                time.sleep(1.0)

                # 反向返回起点
                for joint_config in reversed(joint_path):
                    if not viewer.is_running():
                        break
                    set_joint_positions(model, data, joint_names, joint_config)
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(0.01)

                time.sleep(0.5)

            print("\n✓ 动画完成! 保持窗口打开...")
            print("关闭窗口以退出.")

            # 保持在起点
            set_joint_positions(model, data, joint_names, joint_path[0])
            mujoco.mj_forward(model, data)

            while viewer.is_running():
                viewer.sync()
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n⏹ 用户中断可视化")


def interactive_planning_and_control():
    """
    交互式规划与控制主循环
    结合了 interactive_pose_control_simple.py 的交互性和 RRT-Connect 路径规划
    """
    print("\n" + "="*70)
    print("交互式关节空间 RRT-CONNECT 路径规划与控制")
    print("="*70)

    # ========== 初始化环境 ==========
    print("\n[初始化] 加载MuJoCo环境...")

    model_path = os.path.join(
        os.path.dirname(__file__),
        "../../robot_model/exp/env_robot_torque.xml"
    )

    if not os.path.exists(model_path):
        print(f"❌ 模型文件未找到: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    joint_names = [f"joint{i+1}" for i in range(6)]

    print(f"✓ 环境加载成功: {model.nq} DOF 机械臂")

    # 初始化IK求解器
    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "../../robot_model/robot/urdf/arm620.urdf"
    )

    if not os.path.exists(urdf_path):
        print(f"❌ URDF文件未找到: {urdf_path}")
        return

    ik_solver = ArmInverseKinematics(urdf_path, ee_link='tools_link')
    print("✓ IK求解器初始化成功 (末端: tools_link)")

    # 初始化碰撞检测器
    collision_checker = MuJoCoCollisionChecker(
        model=model,
        data=data,
        joint_names=joint_names,
        check_self_collision=True,
        exclude_bodies=['floor']
    )
    print("✓ 碰撞检测器初始化成功")

    # 初始化配置空间
    joint_limits = np.array([
        [-np.pi, np.pi],  # joint1
        [-np.pi, np.pi],  # joint2
        [-np.pi, np.pi],  # joint3
        [-np.pi, np.pi],  # joint4
        [-np.pi, np.pi],  # joint5
        [-np.pi, np.pi],  # joint6
    ])
    config_space = ConfigurationSpace(joint_limits=joint_limits)
    print("✓ 配置空间初始化成功")

    # 初始化 RRT-Connect 规划器
    planner = RRTConnectPlanner(
        config_space=config_space,
        collision_checker=collision_checker,
        max_step_size=0.2,
        goal_tolerance=0.1
    )
    print("✓ RRT-Connect 规划器初始化成功")

    # 获取 target_body
    target_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "target_body"
    )

    if target_body_id == -1:
        print("❌ 找不到 target_body，请检查 XML 文件")
        return

    print(f"✓ 找到 target_body (ID: {target_body_id})")

    # 获取 mocap body ID（用于读取 mocap 位置）
    # MuJoCo 中 mocap body 的位置存储在 data.mocap_pos 中
    mocap_id = model.body_mocapid[target_body_id]
    if mocap_id == -1:
        print("❌ target_body 不是 mocap 物体！")
        return

    print(f"✓ target_body 是 mocap 物体 (mocap_id: {mocap_id})")

    # 设置初始位置
    start_joint_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    set_joint_positions(model, data, joint_names, start_joint_config)
    mujoco.mj_forward(model, data)

    print("\n" + "="*70)
    print("🎮 交互式控制")
    print("="*70)
    print("  📌 如何拖动目标:")
    print("     1. 按住 Ctrl 键")
    print("     2. 按住鼠标右键拖动蓝色球体")
    print("     3. 松开鼠标后自动触发路径规划")
    print("")
    print("  💡 提示:")
    print("     - 绿色球体 = IK可解，可以规划路径")
    print("     - 红色球体 = IK失败，目标不可达")
    print("     - 按 'Ctrl+C' 退出\n")

    # 状态变量
    last_target_pos = None
    current_path = None
    path_index = 0
    is_executing_path = False
    target_changed = False
    position_change_threshold = 0.01  # 1cm

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("✓ Viewer已启动\n")

            # 启用 mocap 物体的交互（可以拖动）
            viewer.cam.fixedcamid = -1
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            while viewer.is_running():
                # 读取当前目标位置（从 mocap 数据）
                target_pos = data.mocap_pos[mocap_id].copy()
                target_quat = data.mocap_quat[mocap_id].copy()

                # 检测目标位置是否变化
                if last_target_pos is None or np.linalg.norm(target_pos - last_target_pos) > position_change_threshold:
                    if not is_executing_path:  # 只在非执行状态下标记变化
                        target_changed = True
                        last_target_pos = target_pos.copy()

                # 如果目标变化且没有在执行路径，触发新的规划
                if target_changed and not is_executing_path:
                    print(f"\n🎯 检测到新目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                    target_changed = False

                    # 转换为位姿矩阵
                    target_orientation = np.zeros((3, 3))
                    mujoco.mju_quat2Mat(target_orientation.flatten(), target_quat)
                    target_orientation = target_orientation.reshape(3, 3)

                    goal_pose = np.eye(4)
                    goal_pose[:3, :3] = target_orientation
                    goal_pose[:3, 3] = target_pos

                    # 获取当前关节配置作为起点
                    current_config = np.array([data.qpos[model.jnt_qposadr[model.joint(name).id]]
                                              for name in joint_names])

                    # 步骤1: IK求解
                    print("  [1/3] IK求解中...")
                    ik_success, goal_joint_config = solve_ik_for_target(
                        ik_solver=ik_solver,
                        target_pose=goal_pose,
                        seed_config=current_config,
                        collision_checker=collision_checker,
                        max_attempts=10
                    )

                    if not ik_success:
                        print("  ❌ IK求解失败，目标不可达")
                        current_path = None
                        continue

                    # 归一化关节角度
                    goal_joint_config = np.arctan2(np.sin(goal_joint_config), np.cos(goal_joint_config))

                    # 步骤2: RRT-Connect路径规划
                    print("  [2/3] RRT-Connect路径规划中...")
                    joint_path, info = planner.plan(
                        start_config=current_config,
                        goal_config=goal_joint_config,
                        max_iterations=3000,
                        max_time=5.0
                    )

                    if not info['success']:
                        print(f"  ❌ 路径规划失败: {info.get('error', '未知')}")
                        current_path = None
                        continue

                    print(f"  ✓ 规划成功 ({info['planning_time']:.2f}s, {len(joint_path)}点)")

                    # 步骤3: 路径插值
                    print("  [3/3] 路径插值中...")
                    from cartesian_rrt_connect import interpolate_joint_path
                    current_path = interpolate_joint_path(joint_path, num_points_per_segment=10)
                    path_index = 0
                    is_executing_path = True
                    print(f"  ✓ 插值完成，共 {len(current_path)} 点")
                    print("  🚀 开始执行路径...\n")

                # 执行路径
                if is_executing_path and current_path is not None:
                    if path_index < len(current_path):
                        # 设置目标关节位置
                        target_joints = current_path[path_index]
                        set_joint_positions(model, data, joint_names, target_joints)
                        mujoco.mj_forward(model, data)

                        path_index += 1

                        # 定期打印进度
                        if path_index % max(1, len(current_path) // 10) == 0:
                            progress = (path_index / len(current_path)) * 100
                            ee_pos = get_ee_pose(model, data, 'tools_link')[:3, 3]
                            print(f"  进度: {progress:5.1f}% | 末端位置: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                    else:
                        # 路径执行完毕
                        print("  ✓ 路径执行完成！")
                        ee_pos = get_ee_pose(model, data, 'tools_link')[:3, 3]
                        distance = np.linalg.norm(target_pos - ee_pos)
                        print(f"  终点误差: {distance*1000:.1f}mm\n")

                        is_executing_path = False
                        current_path = None
                        path_index = 0

                # 可视化
                viewer.user_scn.ngeom = 0

                # 检查当前目标是否可IK求解（用于显示颜色）
                target_orientation = np.zeros((3, 3))
                mujoco.mju_quat2Mat(target_orientation.flatten(), target_quat)
                target_orientation = target_orientation.reshape(3, 3)
                goal_pose = np.eye(4)
                goal_pose[:3, :3] = target_orientation
                goal_pose[:3, 3] = target_pos

                current_config = np.array([data.qpos[model.jnt_qposadr[model.joint(name).id]]
                                          for name in joint_names])

                # 快速IK检查（不打印）
                ik_check, _ = solve_ik_for_target(
                    ik_solver=ik_solver,
                    target_pose=goal_pose,
                    seed_config=current_config,
                    collision_checker=collision_checker,
                    max_attempts=3
                )

                # 绘制目标球体（颜色表示IK状态）
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    target_color = [0, 1, 0, 0.8] if ik_check else [1, 0, 0, 0.8]
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.02, 0, 0],
                        pos=target_pos,
                        mat=np.eye(3).flatten(),
                        rgba=target_color
                    )
                    viewer.user_scn.ngeom += 1

                # 绘制连接线
                ee_pos = get_ee_pose(model, data, 'tools_link')[:3, 3]
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    line_color = [0, 1, 0, 0.4] if ik_check else [1, 0, 0, 0.4]
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LINE,
                        width=0.002,
                        from_=ee_pos,
                        to=target_pos
                    )
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = line_color
                    viewer.user_scn.ngeom += 1

                viewer.sync()
                time.sleep(0.02)  # 50Hz

    except KeyboardInterrupt:
        print("\n⏹ 用户中断")

    print("\n✓ 控制结束")
    print("="*70)


def main_planning_workflow():
    """
    完整的关节空间路径规划工作流
    """
    print("\n" + "="*70)
    print("关节空间 RRT-CONNECT 路径规划")
    print("="*70)

    # ========== 步骤 0: 加载环境 ==========
    print("\n[步骤 0/6] 加载MuJoCo环境...")

    model_path = os.path.join(
        os.path.dirname(__file__),
        "../../robot_model/exp/env_robot_torque.xml"
    )

    if not os.path.exists(model_path):
        print(f"❌ 模型文件未找到: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    joint_names = [f"joint{i+1}" for i in range(6)]

    print(f"✓ 环境加载成功: {model.nq} DOF 机械臂")

    # 初始化IK求解器
    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "../../robot_model/robot/urdf/arm620.urdf"
    )

    if not os.path.exists(urdf_path):
        print(f"❌ URDF文件未找到: {urdf_path}")
        return

    ik_solver = ArmInverseKinematics(urdf_path, ee_link='tools_link')
    print("✓ IK求解器初始化成功 (末端: tools_link)")

    # 初始化碰撞检测器
    collision_checker = MuJoCoCollisionChecker(
        model=model,
        data=data,
        joint_names=joint_names,
        check_self_collision=True,
        exclude_bodies=['floor']
    )
    print("✓ 碰撞检测器初始化成功")

    # 初始化配置空间
    joint_limits = np.array([
        [-np.pi, np.pi],  # joint1
        [-np.pi, np.pi],  # joint2
        [-np.pi, np.pi],  # joint3
        [-np.pi, np.pi],  # joint4
        [-np.pi, np.pi],  # joint5
        [-np.pi, np.pi],  # joint6
    ])
    config_space = ConfigurationSpace(joint_limits=joint_limits)
    print("✓ 配置空间初始化成功")

    # ========== 步骤 1: 获取目标点 ==========
    print("\n[步骤 1/6] 获取目标点...")

    # 设置起始配置
    start_joint_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    set_joint_positions(model, data, joint_names, start_joint_config)
    mujoco.mj_forward(model, data)
    start_pose = get_ee_pose(model, data, ee_site_name='tools_link')

    print(f"起始关节配置: {start_joint_config}")
    print(f"起始末端位置: {start_pose[:3, 3]}")

    # 从 target_body 读取目标位置和姿态
    target_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "target_body"
    )

    if target_body_id == -1:
        print("❌ 找不到 target_body，请检查 XML 文件中是否定义了 target_body")
        print("回退到默认目标位置...")
        target_position = np.array([0.4, 0.2, 0.3])
        target_orientation = start_pose[:3, :3].copy()
    else:
        print(f"✓ 找到 target_body (ID: {target_body_id})")

        # 从 target_body 读取位置和姿态
        target_position = data.xpos[target_body_id].copy()
        target_quat = data.xquat[target_body_id].copy()

        # 将四元数转换为旋转矩阵
        target_orientation = np.zeros((3, 3))
        mujoco.mju_quat2Mat(target_orientation.flatten(), target_quat)
        target_orientation = target_orientation.reshape(3, 3)

        print(f"从 target_body 读取目标位置: {target_position}")
        print(f"从 target_body 读取目标姿态 (四元数): {target_quat}")

    goal_pose = np.eye(4)
    goal_pose[:3, :3] = target_orientation
    goal_pose[:3, 3] = target_position

    print(f"目标末端位置: {target_position}")

    # ========== 步骤 2: 通过IK求解目标关节角度 ==========
    print("\n[步骤 2/6] 通过IK求解目标关节角度...")

    ik_success, goal_joint_config = solve_ik_for_target(
        ik_solver=ik_solver,
        target_pose=goal_pose,
        seed_config=start_joint_config,
        collision_checker=collision_checker,
        max_attempts=20
    )

    if not ik_success:
        print("⚠️ IK求解失败，目标点不可达！")
        print("建议: 在可视化窗口中手动调整 target_body 的位置")
        print("\n尝试的目标位置:", target_position)
        print("可能原因:")
        print("  1. 目标点超出工作空间")
        print("  2. 目标姿态无法实现")
        print("  3. 所有IK解都存在碰撞")
        print("\n" + "="*70)
        print("🎯 进入交互式目标调整模式")
        print("="*70)

        # 启动交互式可视化，让用户调整目标位置
        interactive_target_adjustment(
            model=model,
            data=data,
            joint_names=joint_names,
            ik_solver=ik_solver,
            collision_checker=collision_checker,
            start_joint_config=start_joint_config,
            target_body_id=target_body_id if target_body_id != -1 else None
        )
        return

    print(f"✓ 目标关节配置: {goal_joint_config}")

    # 归一化关节角度到 [-π, π] 范围
    goal_joint_config = np.arctan2(np.sin(goal_joint_config), np.cos(goal_joint_config))
    print(f"  归一化后: {goal_joint_config}")

    # 验证目标配置
    set_joint_positions(model, data, joint_names, goal_joint_config)
    mujoco.mj_forward(model, data)
    actual_goal_pose = get_ee_pose(model, data, ee_site_name='tools_link')
    error = np.linalg.norm(actual_goal_pose[:3, 3] - target_position)
    print(f"  实际到达位置: {actual_goal_pose[:3, 3]}")
    print(f"  位置误差: {error:.6f} m")

    # ========== 步骤 3: 关节空间RRT-Connect路径搜索 ==========
    print("\n[步骤 3/6] 关节空间RRT-Connect路径搜索...")

    planner = RRTConnectPlanner(
        config_space=config_space,
        collision_checker=collision_checker,
        max_step_size=0.2,  # 关节空间步长 (弧度)
        goal_tolerance=0.1   # 关节空间容差 (弧度)
    )

    print("🚀 开始规划...")
    joint_path, info = planner.plan(
        start_config=start_joint_config,
        goal_config=goal_joint_config,
        max_iterations=5000,
        max_time=30.0
    )

    if not info['success']:
        print(f"❌ 路径规划失败!")
        print(f"  原因: {info.get('error', '未知')}")
        if 'planning_time' in info:
            print(f"  耗时: {info['planning_time']:.3f}s")
        return

    print(f"✓ 路径规划成功!")
    print(f"  - 规划时间: {info['planning_time']:.3f}s")
    print(f"  - 迭代次数: {info['num_iterations']}")
    print(f"  - 树节点数: {info['num_nodes']}")
    print(f"  - 路径长度: {info['path_length']:.3f} rad")
    print(f"  - 路径点数: {len(joint_path)}")

    # ========== 步骤 4: 通过FK计算末端执行器轨迹 ==========
    print("\n[步骤 4/6] 通过FK计算末端执行器轨迹...")

    # 先进行路径插值，使轨迹更平滑
    from cartesian_rrt_connect import interpolate_joint_path
    interpolated_joint_path = interpolate_joint_path(joint_path, num_points_per_segment=20)
    print(f"  路径插值: {len(joint_path)} → {len(interpolated_joint_path)} 点")

    # 计算FK轨迹
    ee_positions, ee_poses = compute_fk_trajectory(
        model=model,
        data=data,
        joint_path=interpolated_joint_path,
        joint_names=joint_names,
        ee_site_name='tools_link'
    )

    print(f"✓ FK计算完成: {len(ee_positions)} 个末端位置")

    # ========== 步骤 5: 显示末端点轨迹 ==========
    print("\n[步骤 5/6] 显示末端点轨迹...")

    # 保存轨迹数据
    trajectory_file = os.path.join(os.path.dirname(__file__), "joint_space_trajectory.npz")
    np.savez(
        trajectory_file,
        joint_path=np.array(joint_path),
        interpolated_joint_path=np.array(interpolated_joint_path),
        ee_positions=ee_positions,
        start_config=start_joint_config,
        goal_config=goal_joint_config,
        planning_info=info
    )
    print(f"✓ 轨迹已保存至: {trajectory_file}")

    # 打印轨迹信息
    print("\n📈 末端执行器轨迹信息:")
    cartesian_path_length = 0.0
    for i in range(len(ee_positions) - 1):
        cartesian_path_length += np.linalg.norm(ee_positions[i+1] - ee_positions[i])
    print(f"  - 笛卡尔空间路径长度: {cartesian_path_length:.4f} m")
    print(f"  - 起点: {ee_positions[0]}")
    print(f"  - 终点: {ee_positions[-1]}")

    # ========== 步骤 6: 可视化 (禁用控制，仅演示) ==========
    print("\n[步骤 6/6] 可视化轨迹 (演示模式，禁用控制)...")

    visualize_trajectory(
        model=model,
        data=data,
        joint_path=interpolated_joint_path,
        ee_positions=ee_positions,
        joint_names=joint_names,
        ee_site_name='tools_link'
    )

    print("\n" + "="*70)
    print("✓ 所有步骤完成!")
    print("="*70)


def example_with_custom_target():
    """
    使用自定义目标的示例
    """
    print("\n" + "="*70)
    print("自定义目标示例")
    print("="*70)
    print("\n💡 修改目标位置:")
    print("  在 main_planning_workflow() 函数中修改 target_position 变量")
    print("  例如: target_position = np.array([0.5, -0.1, 0.4])")
    print("\n💡 修改目标姿态:")
    print("  修改 target_orientation 变量或使用旋转矩阵")


if __name__ == "__main__":
    try:
        # 运行交互式规划与控制
        interactive_planning_and_control()

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
