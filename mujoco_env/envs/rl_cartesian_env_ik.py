"""
改进的强化学习环境 - 无IK版本

特性:
1. 使用局部雅可比伪逆更新关节目标
2. 支持位置控制与关节限位约束
3. 保留轨迹插值与可视化能力
"""

import numpy as np
import mujoco
from mujoco import viewer as mujoco_viewer
from typing import Tuple, Dict, Any, Optional, List
import sys
import os

# 添加必要的路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../robot'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../robot/planner'))

# 导入Robot类
from robot import Robot
from joint_interpolator import CubicJointInterpolator

# 添加tools路径并导入可视化工具
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from vizUtils import TrajectoryDrawer


class RLCartesianEnvWithIK:
    """
    无IK模块的强化学习环境（保留原类名用于兼容）

    控制流程:
    1. RL输出笛卡尔增量: action = [Δx, Δy, Δz]
    2. 计算目标末端位置: target_pos = current_pos + delta
    3. 使用局部雅可比伪逆更新目标关节角度
    4. 控制机械臂移动到目标关节角度
    """

    def __init__(
        self,
        model_path: str = None,
        max_episode_steps: int = 500,
        delta_scale: float = 0.01,
        target_reach_threshold: float = 0.05,
        render_mode: Optional[str] = None,
        enable_visualization: bool = False,
        maintain_orientation: bool = True,  # 是否保持末端姿态不变
        sample_mode: str = 'full',  # 采样模式: 'single' (只采样joint1) 或 'full' (全关节采样)
        uniform_sampling: bool = True,  # 是否使用均匀采样（否则使用随机采样）
        initial_joint_positions: Optional[np.ndarray] = None  # 自定义初始关节位置，形状 (6,)
    ):
        """
        初始化环境

        Args:
            model_path: MuJoCo XML模型路径
            max_episode_steps: 每个episode的最大步数
            delta_scale: 动作缩放因子
            target_reach_threshold: 目标到达阈值
            render_mode: 渲染模式
            enable_visualization: 是否启用MuJoCo viewer
            maintain_orientation: 是否保持末端姿态不变
            sample_mode: 采样模式 ('single': 只采样joint1, 'full': 全关节采样)
            uniform_sampling: 是否使用均匀采样 (True: 均匀网格采样, False: 随机采样)
            initial_joint_positions: 自定义初始关节位置 (6,) 数组，单位：弧度
        """
        # 设置模型路径
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "../robot_model/exp/env_robot_torque.xml")

        # 初始化Robot
        self.robot = Robot(
            model_path,
            enable_ros2=True,              # 启用经典ROS2发布功能（摄像头）
            enable_joint_state_ros2=True,  # 启用ROS2关节状态发布
            joint_state_topic="/joint_states_sim"
        )
        self.robot.set_control_mode("zero_gravity")

        # 环境参数
        self.max_episode_steps = max_episode_steps
        self.delta_scale = delta_scale
        self.target_reach_threshold = target_reach_threshold
        self.render_mode = render_mode
        self.enable_visualization = enable_visualization
        self.maintain_orientation = maintain_orientation
        self.sample_mode = sample_mode  # 采样模式
        self.uniform_sampling = uniform_sampling  # 是否均匀采样

        # 关节限位
        self.joint_limits = self.robot.model.jnt_range[:6].copy()  # shape: (6, 2)
        print(f"✓ 关节限位已加载")
        print(f"  - 使用MuJoCo关节限位: {self.joint_limits.flatten()}")

        # 初始化关节插值器
        self.joint_interpolator = CubicJointInterpolator(n_steps=50)

        # Episode状态
        self.current_step = 0
        self.episode_reward = 0.0

        # 目标状态
        self.target_pos = np.array([0.4, 0.3, 0.5])
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # 工作空间边界（已禁用）
        # 仍保留字段以兼容旧代码路径，但不再用于裁剪/终止/采样过滤
        self.workspace_center = np.array([0.3, 0.0, 0.5])
        self.workspace_range = np.array([np.inf, np.inf, np.inf])
        self.workspace_min = np.array([-np.inf, -np.inf, -np.inf])
        self.workspace_max = np.array([np.inf, np.inf, np.inf])

        # 观测和动作空间维度
        self.observation_dim = 47
        self.action_dim = 3

        # 初始关节位姿设置
        if initial_joint_positions is not None:
            if isinstance(initial_joint_positions, np.ndarray) and initial_joint_positions.shape == (6,):
                self.initial_joint_positions = initial_joint_positions.copy()
                print(f"  使用自定义初始关节位置(度): {np.rad2deg(self.initial_joint_positions)}")
            else:
                print(f"⚠ 初始关节位置格式错误，使用默认值")
                self.initial_joint_positions = np.deg2rad([0, 0, -90, 0, -90, 0])
        else:
            self.initial_joint_positions = np.deg2rad([0, 0, -90, 0, -90, 0])
            print(f"  使用默认初始关节位置(度): {np.rad2deg(self.initial_joint_positions)}")

        # Viewer
        self.viewer = None
        if self.enable_visualization:
            self.viewer = mujoco_viewer.launch_passive(self.robot.model, self.robot.data)
            # 关闭坐标轴显示（包括末端工具坐标轴）
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
            
        # 轨迹可视化管理
        self.active_trajectory = None
        self.trajectory_drawer = None

        print("=" * 70)
        print("✓ RLCartesianEnvWithIK initialized")
        print("=" * 70)
        print(f"  - 保持姿态: {maintain_orientation}")
        print(f"  - 采样模式: {sample_mode}")
        print(f"  - 均匀采样: {uniform_sampling}")
        print(f"  - Observation dim: {self.observation_dim}")
        print(f"  - Action dim: {self.action_dim}")
        print("=" * 70)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)

        self.robot.reset_simulation()
        self.robot.data.qpos[:6] = self.initial_joint_positions
        mujoco.mj_forward(self.robot.model, self.robot.data)

        # 采样新目标
        self.target_pos, self.target_quat = self._sample_target()
        self._update_target_visualization()

        self.current_step = 0
        self.episode_reward = 0.0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作

        完整流程:
        1. 获取当前末端位姿
        2. 计算目标末端位置 = 当前位置 + 增量
        3. 使用局部雅可比伪逆计算目标关节角度
        4. 控制机械臂移动
        """
        # 限制动作范围
        action = np.clip(action, -1.0, 1.0)

        # 1. 获取当前末端位姿
        current_ee_pos = self.robot.get_ee_position()
        current_ee_quat = self.robot.get_ee_orientation()

        # 2. 计算目标末端位置
        delta = action * self.delta_scale
        target_ee_pos = current_ee_pos + delta

        # 3. 确定目标姿态
        if self.maintain_orientation:
            # 保持当前姿态不变
            target_ee_quat = current_ee_quat
        else:
            # 使用固定姿态
            target_ee_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # 4. 使用IK求解目标关节角度
        success = self._move_to_position_with_ik(target_ee_pos, target_ee_quat)

        # 更新障碍物
        self.robot.update_obstacles()

        # 执行仿真步
        self.robot.step()

        # 可视化
        if self.enable_visualization and self.viewer is not None:
            # 更新轨迹显示（如果有活跃轨迹）
            if self.trajectory_drawer is not None:
                self._trajectory_render_callback()
            
            self.viewer.sync()

        # 更新状态
        self.current_step += 1
        obs = self._get_observation()
        reward = self._compute_reward()
        self.episode_reward += reward
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_episode_steps

        info = self._get_info()
        info['delta'] = delta
        info['target_ee_pos'] = target_ee_pos
        info['move_success'] = success

        return obs, reward, terminated, truncated, info

    def _move_to_position_with_ik(self, target_pos: np.ndarray,
                                   target_quat: np.ndarray) -> bool:
        """
        使用局部雅可比伪逆移动到目标位置

        Args:
            target_pos: 目标位置 [x, y, z]
            target_quat: 目标姿态 [w, x, y, z]

        Returns:
            success: 目标更新是否成功
        """
        try:
            # 无IK模式：使用局部雅可比伪逆进行关节更新（位置优先）
            current_joints = self.robot.get_joint_positions()
            current_ee_pos = self.robot.get_ee_position()
            pos_error = target_pos - current_ee_pos

            joint_delta = self._compute_jacobian_pseudoinverse(pos_error)
            target_joints = current_joints + joint_delta
            target_joints = np.clip(target_joints, self.joint_limits[:, 0], self.joint_limits[:, 1])

            self.robot.apply_joint_control(target_joints)

            # 位置误差足够小时认为成功
            return float(np.linalg.norm(pos_error)) < 0.01

        except Exception as e:
            print(f"⚠ 关节目标计算失败: {e}")
            return False

    def _compute_jacobian_pseudoinverse(self, pos_error: np.ndarray) -> np.ndarray:
        """数值计算雅可比伪逆"""
        epsilon = 1e-6
        current_joints = self.robot.get_joint_positions()
        current_ee_pos = self.robot.get_ee_position()

        n_joints = len(current_joints)
        jacobian = np.zeros((3, n_joints))

        for i in range(n_joints):
            perturbed_joints = current_joints.copy()
            perturbed_joints[i] += epsilon

            self.robot.data.qpos[:n_joints] = perturbed_joints
            mujoco.mj_forward(self.robot.model, self.robot.data)
            perturbed_ee_pos = self.robot.get_ee_position()

            jacobian[:, i] = (perturbed_ee_pos - current_ee_pos) / epsilon

        # 恢复原始状态
        self.robot.data.qpos[:n_joints] = current_joints
        mujoco.mj_forward(self.robot.model, self.robot.data)

        # 计算伪逆和关节增量
        jacobian_pinv = np.linalg.pinv(jacobian)
        joint_delta = jacobian_pinv @ pos_error
        joint_delta = np.clip(joint_delta, -0.1, 0.1)

        return joint_delta

    def _pose_to_matrix(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """将位置和四元数转换为4x4齐次变换矩阵"""
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    def _get_observation(self) -> np.ndarray:
        """获取观测（与原环境相同）"""
        self.robot.update_joint_state()

        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        ee_pos = self.robot.get_ee_position()
        ee_vel = self.robot.get_ee_velocity()
        ee_quat = self.robot.get_ee_orientation()
        ee_angular_vel = self.robot.get_ee_angular_velocity()

        target_pos = self.robot.get_target_position()
        if target_pos is None:
            target_pos = self.target_pos

        target_quat = self.robot.get_target_orientation()
        if target_quat is None:
            target_quat = self.target_quat

        target_vel = self.robot.get_target_velocity()
        if target_vel is None:
            target_vel = np.zeros(3)

        obs1_pos = self.robot.get_obstacle_position('obstacle1')
        obs1_vel = self.robot.get_obstacle_velocity('obstacle1')
        obs2_pos = self.robot.get_obstacle_position('obstacle2')
        obs2_vel = self.robot.get_obstacle_velocity('obstacle2')

        if obs1_pos is None:
            obs1_pos = np.zeros(3)
        if obs1_vel is None:
            obs1_vel = np.zeros(3)
        if obs2_pos is None:
            obs2_pos = np.zeros(3)
        if obs2_vel is None:
            obs2_vel = np.zeros(3)

        observation = np.concatenate([
            joint_pos, joint_vel, ee_pos, ee_vel, ee_quat, ee_angular_vel,
            target_pos, target_quat, target_vel, obs1_pos, obs1_vel,
            obs2_pos, obs2_vel
        ])

        return observation

    def _compute_reward(self) -> float:
        """计算奖励"""
        c1 = 500.0
        c2 = 15.0

        ee_pos = self.robot.get_ee_position()
        d_T = np.linalg.norm(self.target_pos - ee_pos)
        R_T = 0.5 * d_T ** 2

        distance_info = self.robot.get_minimum_distance_to_obstacles()
        d_O = distance_info['min_distance']
        R_O = (1.0 / (1.0 + d_O)) ** 35

        reward = -c1 * R_T - c2 * R_O
        return reward

    def _check_terminated(self) -> bool:
        """检查终止条件"""
        ee_pos = self.robot.get_ee_position()
        distance = np.linalg.norm(self.target_pos - ee_pos)

        if distance < self.target_reach_threshold:
            return True

        return False

    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        """工作空间检查（已禁用，恒为True）"""
        return True

    def _generate_uniform_samples(self, n_samples: int) -> np.ndarray:
        """
        生成均匀分布的关节角度样本

        Args:
            n_samples: 需要生成的样本数量

        Returns:
            samples: 关节角度样本 (n_samples, 6)
        """
        if not hasattr(self, 'joint_limits'):
            self.joint_limits = self.robot.model.jnt_range[:6].copy()

        samples = []

        if self.sample_mode == 'single':
            # 单关节模式：只在joint1上均匀采样
            joint1_values = np.linspace(
                self.joint_limits[0, 0],
                self.joint_limits[0, 1],
                n_samples
            )
            for value in joint1_values:
                sample = np.zeros(6)
                sample[0] = value
                samples.append(sample)

        elif self.sample_mode == 'full':
            # 全关节模式：使用拉丁超立方采样（Latin Hypercube Sampling）
            # 这比纯网格采样更高效，能更好地覆盖高维空间

            # 为每个关节生成均匀分层的样本点
            for i in range(n_samples):
                sample = np.zeros(6)
                for j in range(6):
                    # 将[0,1]区间分成n_samples个子区间，在第i个子区间内随机采样
                    lower = i / n_samples
                    upper = (i + 1) / n_samples
                    random_in_interval = np.random.uniform(lower, upper)
                    # 映射到关节范围
                    sample[j] = self.joint_limits[j, 0] + random_in_interval * (
                        self.joint_limits[j, 1] - self.joint_limits[j, 0]
                    )
                samples.append(sample)

            # 随机打乱每个维度的顺序（拉丁超立方采样的关键步骤）
            samples = np.array(samples)
            for j in range(6):
                np.random.shuffle(samples[:, j])

        return np.array(samples) if isinstance(samples, list) else samples

    def _sample_target(self, max_attempts: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样目标位置

        根据 sample_mode 参数：
        - 'single': 只随机joint1，其他关节固定为0度（快速，适合平面运动）
        - 'full': 随机所有6个关节（完整3D空间采样）

        直接使用FK计算目标位置，跳过IK验证

        Args:
            max_attempts: 最大采样尝试次数

        Returns:
            target_pos: 目标位置 [x, y, z]
            target_quat: 目标姿态四元数 [w, x, y, z]
        """
        # 获取关节限位
        if not hasattr(self, 'joint_limits'):
            self.joint_limits = self.robot.model.jnt_range[:6].copy()

        for _ in range(max_attempts):
            # 根据采样模式生成关节角度
            if self.sample_mode == 'single':
                # 只随机joint1，其他关节固定为0
                sampled_joints = np.zeros(6)
                sampled_joints[0] = np.random.uniform(
                    self.joint_limits[0, 0],  # joint1下限
                    self.joint_limits[0, 1]   # joint1上限
                )
            elif self.sample_mode == 'full':
                # 随机所有6个关节
                sampled_joints = np.zeros(6)
                for i in range(6):
                    sampled_joints[i] = np.random.uniform(
                        self.joint_limits[i, 0],  # 关节i下限
                        self.joint_limits[i, 1]   # 关节i上限
                    )
            else:
                raise ValueError(f"未知的采样模式: {self.sample_mode}, 必须是 'single' 或 'full'")

            # 保存原始关节状态
            original_joints = self.robot.data.qpos[:6].copy()

            # 设置采样的关节角度并执行正向运动学
            self.robot.data.qpos[:6] = sampled_joints
            mujoco.mj_forward(self.robot.model, self.robot.data)

            # 获取FK计算的末端位置和姿态
            target_pos = self.robot.get_ee_position()
            target_quat = self.robot.get_ee_orientation()

            # 恢复原始关节状态
            self.robot.data.qpos[:6] = original_joints
            mujoco.mj_forward(self.robot.model, self.robot.data)

            # 工作空间过滤已禁用，直接返回采样结果
            return target_pos, target_quat

        # 理论上不会到达这里，保留兜底
        target_pos = self.robot.get_ee_position()
        target_quat = self.robot.get_ee_orientation()
        return target_pos, target_quat

    def _update_target_visualization(self):
        """更新目标可视化"""
        target_body_id = mujoco.mj_name2id(
            self.robot.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target_body"
        )

        if target_body_id != -1:
            mocap_id = self.robot.model.body_mocapid[target_body_id]
            if mocap_id != -1:
                self.robot.data.mocap_pos[mocap_id] = self.target_pos
                self.robot.data.mocap_quat[mocap_id] = self.target_quat
                mujoco.mj_forward(self.robot.model, self.robot.data)

    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        ee_pos = self.robot.get_ee_position()
        distance_to_target = np.linalg.norm(self.target_pos - ee_pos)
        distance_info = self.robot.get_minimum_distance_to_obstacles()

        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'distance_to_target': distance_to_target,
            'ee_position': ee_pos,
            'target_position': self.target_pos,
            'in_workspace': self._is_in_workspace(ee_pos),
            'min_obstacle_distance': distance_info['min_distance'],
            'nearest_obstacle': distance_info['obstacle_name'],
            'nearest_robot_geom': distance_info['robot_geom'],
        }

        return info

    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco_viewer.launch_passive(self.robot.model, self.robot.data)
            self.viewer.sync()

    def close(self):
        """关闭环境"""
        # 清理轨迹可视化
        self.clear_trajectory_visualization()
        
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        print("✓ RLCartesianEnvWithIK closed")

    def generate_trajectory_from_sampling(
        self,
        n_targets: int = 5,
        interpolation_steps: int = 50,
        max_retries_per_target: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        通过传统采样生成轨迹

        流程:
        1. 根据采样模式和类型生成关节角度
        2. 使用FK计算末端位姿
        3. 直接使用采样关节作为目标关节
        4. 使用插值器生成平滑轨迹

        Args:
            n_targets: 采样的新目标点数（不包括起点）
            interpolation_steps: 每段的插值步数
            max_retries_per_target: 每个目标点IK失败时的最大重试次数

        Returns:
            joint_waypoints: 关节路点 (n_waypoints, 6)
            cartesian_waypoints: 笛卡尔路点 (n_waypoints, 7) [x,y,z,qw,qx,qy,qz]
            full_trajectory: 完整插值轨迹 (n_total_steps, 6)
        """
        print(f"\n🎯 采样轨迹生成: {n_targets}个目标点, 插值{interpolation_steps}步/段")

        if n_targets < 1:
            print("⚠ n_targets 必须 >= 1")
            return None, None, None

        # 保存当前关节位置
        original_joints = self.robot.data.qpos[:6].copy()

        joint_waypoints = []
        cartesian_waypoints = []

        # 第一个点使用当前位置
        current_joints = self.robot.get_joint_positions()
        current_ee_pos = self.robot.get_ee_position()
        current_ee_quat = self.robot.get_ee_orientation()

        joint_waypoints.append(current_joints.copy())
        cartesian_waypoints.append(np.concatenate([current_ee_pos, current_ee_quat]))

        print(f"\n📍 起点: [{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}]")

        # 生成均匀采样网格点（如果使用均匀采样）
        if self.uniform_sampling:
            uniform_samples = self._generate_uniform_samples(n_targets)
            uniform_sample_idx = 0

        # 采样新目标点
        successful_targets = 0
        for i in range(n_targets):
            # 对每个目标点进行重试，直到IK成功或达到最大重试次数
            retry_count = 0
            ik_success = False

            while retry_count < max_retries_per_target and not ik_success:
                retry_count += 1

                # 1. 根据采样模式和采样类型生成关节角度
                if self.uniform_sampling and retry_count == 1:
                    # 均匀采样：第一次尝试使用预生成的均匀样本
                    sampled_joints = uniform_samples[uniform_sample_idx]
                    uniform_sample_idx += 1
                else:
                    # 随机采样：或者均匀采样失败后的重试
                    if self.sample_mode == 'single':
                        # 只随机joint1，其他为0
                        sampled_joints = np.zeros(6)
                        sampled_joints[0] = np.random.uniform(
                            self.joint_limits[0, 0],
                            self.joint_limits[0, 1]
                        )
                    elif self.sample_mode == 'full':
                        # 随机所有6个关节
                        sampled_joints = np.zeros(6)
                        for j in range(6):
                            sampled_joints[j] = np.random.uniform(
                                self.joint_limits[j, 0],
                                self.joint_limits[j, 1]
                            )
                    else:
                        raise ValueError(f"未知的采样模式: {self.sample_mode}")

                # 2. 使用FK计算末端位姿
                self.robot.data.qpos[:6] = sampled_joints
                mujoco.mj_forward(self.robot.model, self.robot.data)

                target_pos = self.robot.get_ee_position()
                target_quat = self.robot.get_ee_orientation()

                # 恢复原始状态
                self.robot.data.qpos[:6] = original_joints
                mujoco.mj_forward(self.robot.model, self.robot.data)

                # 3. 无IK模式：直接使用采样关节作为目标关节
                target_joints = sampled_joints.copy()

                # FK验证IK结果
                original_joints = self.robot.data.qpos[:6].copy()
                self.robot.data.qpos[:6] = target_joints
                mujoco.mj_forward(self.robot.model, self.robot.data)
                fk_pos = self.robot.get_ee_position()
                fk_quat = self.robot.get_ee_orientation()
                self.robot.data.qpos[:6] = original_joints
                mujoco.mj_forward(self.robot.model, self.robot.data)
                
                # 计算误差
                pos_error = np.linalg.norm(fk_pos - target_pos)
                
                # 检查关节限位
                joint_violations = []
                for j in range(6):
                    if target_joints[j] < self.joint_limits[j, 0]:
                        joint_violations.append(f"J{j+1}<{self.joint_limits[j, 0]:.3f} ({target_joints[j]:.3f})")
                    elif target_joints[j] > self.joint_limits[j, 1]:
                        joint_violations.append(f"J{j+1}>{self.joint_limits[j, 1]:.3f} ({target_joints[j]:.3f})")
                
                print(f"  🔍 FK验证 目标点{successful_targets + 1}:")
                print(f"     目标位置: [{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}]")
                print(f"     目标姿态: [{target_quat[0]:.6f}, {target_quat[1]:.6f}, {target_quat[2]:.6f}, {target_quat[3]:.6f}]")
                print(f"     FK验证位置: [{fk_pos[0]:.6f}, {fk_pos[1]:.6f}, {fk_pos[2]:.6f}]")
                print(f"     FK验证姿态: [{fk_quat[0]:.6f}, {fk_quat[1]:.6f}, {fk_quat[2]:.6f}, {fk_quat[3]:.6f}]")
                print(f"     位置误差: {pos_error:.6f} m")
                if joint_violations:
                    print(f"     ⚠ 关节限位违反: {', '.join(joint_violations)}")
                else:
                    print(f"     ✓ 关节限位检查通过")

                # IK成功，保存路点
                joint_waypoints.append(target_joints.copy())
                cartesian_waypoints.append(np.concatenate([target_pos, target_quat]))
                current_joints = target_joints
                successful_targets += 1

                print(f"  ✓ 目标点{successful_targets}: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

        if len(joint_waypoints) < 2:
            print("❌ 路点数量不足，无法生成轨迹")
            return None, None, None

        # 4. 使用插值器生成完整轨迹
        original_n_steps = self.joint_interpolator.n_steps
        self.joint_interpolator.n_steps = interpolation_steps

        trajectory_segments = []
        for i in range(len(joint_waypoints) - 1):
            segment = self.joint_interpolator.interpolate(
                joint_waypoints[i],
                joint_waypoints[i + 1]
            )
            trajectory_segments.append(segment)

        full_trajectory = np.vstack(trajectory_segments)
        print(f"\n✓ 总轨迹: {len(full_trajectory)} 步")

        # 恢复插值器步数
        self.joint_interpolator.n_steps = original_n_steps

        # 恢复原始关节状态
        self.robot.data.qpos[:6] = original_joints
        mujoco.mj_forward(self.robot.model, self.robot.data)

        return np.array(joint_waypoints), np.array(cartesian_waypoints), full_trajectory

    def _trajectory_render_callback(self):
        """轨迹渲染函数，在每次渲染时绘制轨迹"""
        if self.trajectory_drawer is not None and self.viewer is not None:
            # 绘制轨迹
            self.trajectory_drawer.draw_trajectory(
                self.viewer,
                color=[0, 1, 0, 1],  # 绿色
                width=0.003,
                fade=False
            )

    def clear_trajectory_visualization(self):
        """清除轨迹可视化"""
        self.active_trajectory = None
        self.trajectory_drawer = None

    def visualize_trajectory(
        self,
        joint_waypoints: np.ndarray,
        cartesian_waypoints: np.ndarray,
        full_trajectory: np.ndarray
    ):
        """
        可视化轨迹，集成到主仿真循环中持续显示
        
        Args:
            joint_waypoints: 关节路点 (n_waypoints, 6)
            cartesian_waypoints: 笛卡尔路点 (n_waypoints, 7) [x,y,z,qw,qx,qy,qz]
            full_trajectory: 完整插值轨迹 (n_total_steps, 6)
        """
        print(f"\n🎬 可视化轨迹执行...")
        
        if self.viewer is None:
            print("⚠ 没有可视化器，跳过轨迹可视化")
            return
        
        # 1. 计算轨迹的所有末端位置
        ee_path = []
        original_joints = self.robot.data.qpos[:6].copy()
        
        for joint_positions in full_trajectory:
            self.robot.data.qpos[:6] = joint_positions
            mujoco.mj_forward(self.robot.model, self.robot.data)
            ee_pos = self.robot.get_ee_position()
            ee_path.append(ee_pos)
        
        ee_path = np.array(ee_path)
        
        # 恢复原始关节位置
        self.robot.data.qpos[:6] = original_joints
        mujoco.mj_forward(self.robot.model, self.robot.data)
        
        # 2. 创建轨迹绘制器并保存
        from vizUtils import TrajectoryDrawer
        self.trajectory_drawer = TrajectoryDrawer(max_segments=len(ee_path), min_distance=0.001)
        
        # 添加所有轨迹点
        for pos in ee_path:
            self.trajectory_drawer.add_point(pos)
        
        print(f"  ✓ 轨迹点数: {len(self.trajectory_drawer.positions)}")
        
        # 3. 设置目标点显示
        self._show_waypoints(cartesian_waypoints)
        
        # 4. 立即绘制一次
        self._trajectory_render_callback()
        self.viewer.sync()
        
        print("  ✓ 轨迹已集成到主循环，将在每次step()时自动更新")
        print("  💡 调用 clear_trajectory_visualization() 清除轨迹显示")
    
    def update_trajectory_display(self):
        """手动更新轨迹显示（用于不支持渲染回调的情况）"""
        if self.trajectory_drawer is not None and self.viewer is not None:
            self._trajectory_render_callback()
            self.viewer.sync()
    
    def _show_waypoints(self, cartesian_waypoints: np.ndarray):
        """显示路点标记"""
        target_body_id = mujoco.mj_name2id(
            self.robot.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target_body"
        )
        
        if target_body_id != -1:
            mocap_id = self.robot.model.body_mocapid[target_body_id]
            if mocap_id != -1:
                # 显示最后一个路点作为目标标记
                final_waypoint = cartesian_waypoints[-1]
                pos = final_waypoint[:3]
                quat = final_waypoint[3:]
                
                self.robot.data.mocap_pos[mocap_id] = pos
                self.robot.data.mocap_quat[mocap_id] = quat
                mujoco.mj_forward(self.robot.model, self.robot.data)
                
                print(f"  📍 目标标记: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    def save_trajectory_to_file(
        self,
        trajectory: np.ndarray,
        cartesian_waypoints: np.ndarray,
        filename: str = None
    ) -> str:
        """
        保存轨迹到文件

        Args:
            trajectory: 关节轨迹 (n_steps, 6)
            cartesian_waypoints: 笛卡尔路点 (n_waypoints, 7)
            filename: 文件名（如果为None则自动生成）

        Returns:
            保存的文件路径
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.txt"

        filepath = os.path.join(os.path.dirname(__file__), filename)

        with open(filepath, 'w') as f:
            # 写入文件头
            f.write("# Cartesian Trajectory Data\n")
            f.write("# Generated by RLCartesianEnvWithIK\n")
            f.write("#" + "=" * 70 + "\n\n")

            # 写入路点信息
            f.write("# Cartesian Waypoints:\n")
            f.write("# Format: x(m) y(m) z(m) qw qx qy qz\n")
            f.write("#" + "-" * 70 + "\n")
            for i, waypoint in enumerate(cartesian_waypoints):
                f.write(f"# Waypoint {i+1}: ")
                f.write(f"{waypoint[0]:.6f} {waypoint[1]:.6f} {waypoint[2]:.6f} ")
                f.write(f"{waypoint[3]:.6f} {waypoint[4]:.6f} {waypoint[5]:.6f} {waypoint[6]:.6f}\n")

            f.write("\n")

            # 写入轨迹数据
            f.write("# Joint Trajectory:\n")
            f.write("# Format: joint1(rad) joint2(rad) joint3(rad) joint4(rad) joint5(rad) joint6(rad)\n")
            f.write("#" + "-" * 70 + "\n")
            for joint_positions in trajectory:
                f.write(" ".join(f"{angle:.6f}" for angle in joint_positions))
                f.write("\n")

        print(f"✓ 轨迹已保存到: {filepath}")
        print(f"  - 路点数: {len(cartesian_waypoints)}")
        print(f"  - 轨迹步数: {len(trajectory)}")

        return filepath

    # ========== 障碍物控制方法 ==========
    
    def enable_obstacles(self):
        """启用所有障碍物运动"""
        self.robot.enable_obstacles()

    def disable_obstacles(self):
        """禁用所有障碍物运动"""
        self.robot.disable_obstacles()

    def toggle_obstacles(self):
        """切换障碍物运动状态"""
        return self.robot.toggle_obstacles()

    def is_obstacles_enabled(self):
        """返回当前障碍物启用状态"""
        return self.robot.is_obstacles_enabled()

    def set_obstacle_frequency(self, obstacle_idx, frequency):
        """设置障碍物运动频率"""
        self.robot.set_obstacle_frequency(obstacle_idx, frequency)

    def set_obstacle_amplitude(self, obstacle_idx, amplitude):
        """设置障碍物运动幅度"""
        self.robot.set_obstacle_amplitude(obstacle_idx, amplitude)
