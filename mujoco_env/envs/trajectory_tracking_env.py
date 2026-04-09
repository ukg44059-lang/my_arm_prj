"""
轨迹跟踪环境
基于rl_cartesian_env_ik.py，添加轨迹跟踪功能和经典轨迹跟踪奖励函数

经典轨迹跟踪奖励组成:
1. 位置跟踪奖励 - 高斯分布奖励
2. 速度跟踪奖励 - 速度匹配奖励  
3. 进度奖励 - 沿轨迹前进奖励
4. 障碍物避碰奖励 - 保持现有避障机制
5. 完成奖励 - 成功完成轨迹的奖励
6. 平滑性奖励 - 鼓励平滑运动
"""

import numpy as np
import mujoco
from typing import Tuple, Dict, Any, Optional, List
from .rl_cartesian_env_ik import RLCartesianEnvWithIK


class TrajectoryTrackingEnv(RLCartesianEnvWithIK):
    """
    轨迹跟踪环境
    
    扩展原有的点到点控制环境，添加轨迹跟踪功能
    RL智能体需要跟踪预定义的轨迹，同时避开障碍物
    """
    
    def __init__(
        self,
        trajectory_mode: str = "generated",  # "generated" 或 "predefined"
        trajectory_length: int = 100,        # 轨迹点数
        trajectory_duration: float = 10.0,   # 轨迹时长(秒)
        # 轨迹跟踪奖励权重
        w_position: float = 100.0,           # 位置跟踪权重
        w_velocity: float = 10.0,            # 速度跟踪权重  
        w_progress: float = 1.0,             # 进度奖励权重
        w_obstacle: float = 15.0,            # 障碍物避碰权重
        w_completion: float = 50.0,          # 完成奖励权重
        w_smoothness: float = 0.1,           # 平滑性奖励权重
        # 奖励函数参数
        sigma_position: float = 0.02,        # 位置高斯奖励标准差(2cm)
        sigma_velocity: float = 0.1,         # 速度高斯奖励标准差
        completion_threshold: float = 0.01,  # 轨迹完成阈值(1cm)
        **kwargs
    ):
        """
        初始化轨迹跟踪环境
        
        Args:
            trajectory_mode: 轨迹模式 ("generated": 自动生成, "predefined": 预定义)
            trajectory_length: 轨迹点数
            trajectory_duration: 轨迹时长
            w_*: 各项奖励权重
            sigma_*: 高斯奖励标准差
            completion_threshold: 完成阈值
        """
        super().__init__(**kwargs)
        
        # 轨迹参数
        self.trajectory_mode = trajectory_mode
        self.trajectory_length = trajectory_length
        self.trajectory_duration = trajectory_duration
        self.dt_trajectory = trajectory_duration / trajectory_length
        
        # 奖励权重
        self.w_position = w_position
        self.w_velocity = w_velocity
        self.w_progress = w_progress
        self.w_obstacle = w_obstacle
        self.w_completion = w_completion
        self.w_smoothness = w_smoothness
        
        # 奖励函数参数
        self.sigma_position = sigma_position
        self.sigma_velocity = sigma_velocity
        self.completion_threshold = completion_threshold
        
        # 轨迹状态
        self.current_trajectory = None       # 当前轨迹 (N, 7) [x,y,z,vx,vy,vz,t]
        self.trajectory_index = 0            # 当前轨迹点索引
        self.trajectory_start_time = 0.0     # 轨迹开始时间
        self.previous_position = None        # 上一步位置（用于平滑性计算）
        
        print(f"✓ 轨迹跟踪环境初始化")
        print(f"  - 轨迹模式: {trajectory_mode}")
        print(f"  - 轨迹点数: {trajectory_length}")
        print(f"  - 轨迹时长: {trajectory_duration:.1f}s")
        print(f"  - 奖励权重: pos={w_position}, vel={w_velocity}, prog={w_progress}")
        
    def _generate_random_sampled_trajectory(self) -> np.ndarray:
        """
        使用test_simple_trajectory.py的方法生成随机采样轨迹
        
        Returns:
            trajectory: (N, 7) [x, y, z, vx, vy, vz, t]
        """
        # 使用环境现有的轨迹生成功能（来自test_simple_trajectory.py）
        result = self.generate_trajectory_from_sampling(
            n_targets=1,  # 一个目标点
            interpolation_steps=self.trajectory_length,
            max_retries_per_target=20
        )
        
        if result[0] is not None:
            joint_waypoints, cartesian_waypoints, joint_trajectory = result
            
            # 将关节轨迹转换为笛卡尔轨迹格式
            cartesian_trajectory = np.zeros((len(joint_trajectory), 7))
            
            # 保存当前关节状态
            original_qpos = self.robot.data.qpos[:6].copy()
            
            for i, joints in enumerate(joint_trajectory):
                # 设置关节位置来计算末端状态
                self.robot.data.qpos[:6] = joints
                mujoco.mj_forward(self.robot.model, self.robot.data)
                
                # 获取末端位置
                ee_pos = self.robot.get_ee_position()
                
                # 计算速度（简单的数值微分）
                if i > 0:
                    prev_pos = cartesian_trajectory[i-1, :3]
                    dt = self.dt_trajectory
                    ee_vel = (ee_pos - prev_pos) / dt
                else:
                    ee_vel = np.zeros(3)
                
                # 时间
                time = i * self.dt_trajectory
                
                cartesian_trajectory[i] = [*ee_pos, *ee_vel, time]
            
            # 恢复原始关节状态
            self.robot.data.qpos[:6] = original_qpos
            mujoco.mj_forward(self.robot.model, self.robot.data)
            
            return cartesian_trajectory
        else:
            # 采样失败，返回None
            return None
    
    def reset(self, seed: Optional[int] = None, trajectory_type: str = "random") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境并生成新轨迹
        
        Args:
            seed: 随机种子
            trajectory_type: 轨迹类型 (忽略，固定使用random sampling)
        """
        # 调用父类reset
        obs, info = super().reset(seed=seed)
        
        # 生成随机采样轨迹（使用test_simple_trajectory.py的方法）
        max_attempts = 5
        for attempt in range(max_attempts):
            self.current_trajectory = self._generate_random_sampled_trajectory()
            if self.current_trajectory is not None:
                break
            print(f"轨迹生成失败，重试 {attempt + 1}/{max_attempts}")
        
        if self.current_trajectory is None:
            print("⚠️ 轨迹生成失败，使用零轨迹")
            # 创建一个简单的零轨迹作为后备
            start_pos = self.robot.get_ee_position()
            self.current_trajectory = np.zeros((self.trajectory_length, 7))
            for i in range(self.trajectory_length):
                self.current_trajectory[i] = [*start_pos, 0, 0, 0, i * self.dt_trajectory]
        
        # 重置轨迹状态
        self.trajectory_index = 0
        self.trajectory_start_time = 0.0
        self.previous_position = self.robot.get_ee_position().copy()
        
        # 更新info
        info['trajectory_length'] = len(self.current_trajectory)
        info['trajectory_type'] = "random_sampled"
        
        return obs, info
    
    def _get_current_trajectory_target(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前时刻的轨迹目标位置和速度
        
        Returns:
            target_pos: 目标位置 [x, y, z]
            target_vel: 目标速度 [vx, vy, vz]
        """
        if self.current_trajectory is None:
            return np.zeros(3), np.zeros(3)
            
        # 基于时间的轨迹跟踪
        current_time = self.current_step * self.robot.model.opt.timestep
        trajectory_time = current_time - self.trajectory_start_time
        
        # 计算轨迹索引（时间插值）
        time_index = trajectory_time / self.dt_trajectory
        
        # 限制在轨迹范围内
        if time_index >= len(self.current_trajectory) - 1:
            # 轨迹结束，使用最后一个点
            self.trajectory_index = len(self.current_trajectory) - 1
            target_pos = self.current_trajectory[-1, :3]
            target_vel = self.current_trajectory[-1, 3:6]
        else:
            # 线性插值获得平滑的目标
            idx = int(time_index)
            alpha = time_index - idx
            
            if idx + 1 < len(self.current_trajectory):
                pos1 = self.current_trajectory[idx, :3]
                pos2 = self.current_trajectory[idx + 1, :3]
                vel1 = self.current_trajectory[idx, 3:6]
                vel2 = self.current_trajectory[idx + 1, 3:6]
                
                target_pos = pos1 * (1 - alpha) + pos2 * alpha
                target_vel = vel1 * (1 - alpha) + vel2 * alpha
            else:
                target_pos = self.current_trajectory[idx, :3]
                target_vel = self.current_trajectory[idx, 3:6]
            
            self.trajectory_index = idx
        
        return target_pos, target_vel
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作（轨迹跟踪版本）
        
        Args:
            action: 动作数组
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # 执行父类step (但不使用其reward)
        obs, _, terminated, truncated, info = super().step(action)
        
        # 计算轨迹跟踪奖励
        reward = self._compute_trajectory_tracking_reward()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_trajectory_tracking_reward(self) -> float:
        """
        计算轨迹跟踪奖励
        
        Returns:
            total_reward: 总奖励值
        """
        # 获取当前状态
        current_pos = self.robot.get_ee_position()
        current_vel = self.robot.get_ee_velocity()
        
        # 获取轨迹目标
        target_pos, target_vel = self._get_current_trajectory_target()
        
        # 1. 位置跟踪奖励 (高斯奖励)
        pos_error = np.linalg.norm(current_pos - target_pos)
        r_position = np.exp(-pos_error**2 / (2 * self.sigma_position**2))
        
        # 2. 速度跟踪奖励
        vel_error = np.linalg.norm(current_vel - target_vel)
        r_velocity = np.exp(-vel_error**2 / (2 * self.sigma_velocity**2))
        
        # 3. 进度奖励
        trajectory_progress = self.trajectory_index / max(1, len(self.current_trajectory) - 1)
        r_progress = trajectory_progress * 0.1  # 小的进度奖励
        
        # 4. 障碍物避碰奖励（保持原有机制）
        distance_info = self.robot.get_minimum_distance_to_obstacles()
        d_O = distance_info['min_distance']
        r_obstacle = -(1.0 / (1.0 + d_O)) ** 35
        
        # 5. 完成奖励
        r_completion = 0.0
        if pos_error < self.completion_threshold and self.trajectory_index >= len(self.current_trajectory) - 1:
            r_completion = 1.0
        
        # 6. 平滑性奖励（惩罚大的位置变化）
        r_smoothness = 0.0
        if self.previous_position is not None:
            position_change = np.linalg.norm(current_pos - self.previous_position)
            # 惩罚过大的位置跳跃
            r_smoothness = -np.maximum(0, position_change - 0.01)  # 超过1cm的跳跃会被惩罚
        
        self.previous_position = current_pos.copy()
        
        # 组装奖励组件（用于信息记录）
        reward_components = {
            'position': r_position,
            'velocity': r_velocity,
            'progress': r_progress,
            'obstacle': r_obstacle,
            'completion': r_completion,
            'smoothness': r_smoothness,
            'pos_error': pos_error,
            'vel_error': vel_error,
            'trajectory_progress': trajectory_progress
        }
        
        # 保存奖励组件到info（用于分析）
        self._last_reward_components = reward_components
        
        # 计算加权总奖励
        total_reward = (
            self.w_position * r_position +
            self.w_velocity * r_velocity + 
            self.w_progress * r_progress +
            self.w_obstacle * r_obstacle +
            self.w_completion * r_completion +
            self.w_smoothness * r_smoothness
        )
        
        return total_reward
    
    def _get_info(self) -> Dict[str, Any]:
        """
        获取环境信息（扩展版本）
        """
        info = super()._get_info()
        
        # 添加轨迹跟踪信息
        if hasattr(self, '_last_reward_components'):
            info.update({
                'trajectory_index': self.trajectory_index,
                'trajectory_progress': self._last_reward_components.get('trajectory_progress', 0.0),
                'pos_error': self._last_reward_components.get('pos_error', 0.0),
                'vel_error': self._last_reward_components.get('vel_error', 0.0),
                'reward_components': self._last_reward_components
            })
        
        # 添加当前轨迹目标
        if self.current_trajectory is not None:
            target_pos, target_vel = self._get_current_trajectory_target()
            info.update({
                'trajectory_target_pos': target_pos,
                'trajectory_target_vel': target_vel
            })
        
        return info
    
    def _check_terminated(self) -> bool:
        """
        检查终止条件（轨迹跟踪版本）
        """
        # 原有终止条件
        if super()._check_terminated():
            return True
            
        # 轨迹完成终止条件
        if self.current_trajectory is not None:
            current_pos = self.robot.get_ee_position()
            target_pos, _ = self._get_current_trajectory_target()
            pos_error = np.linalg.norm(current_pos - target_pos)
            
            # 如果到达轨迹末端且位置误差小于阈值，则成功完成
            if (self.trajectory_index >= len(self.current_trajectory) - 1 and 
                pos_error < self.completion_threshold):
                return True
        
        return False
    
    def set_custom_trajectory(self, trajectory: np.ndarray):
        """
        设置自定义轨迹
        
        Args:
            trajectory: (N, 7) [x, y, z, vx, vy, vz, t]
        """
        self.current_trajectory = trajectory.copy()
        self.trajectory_index = 0
        self.trajectory_start_time = self.current_step * self.robot.model.opt.timestep
        print(f"✓ 设置自定义轨迹，包含 {len(trajectory)} 个点")


if __name__ == "__main__":
    """测试轨迹跟踪环境"""
    print("=" * 70)
    print("测试轨迹跟踪环境")
    print("=" * 70)
    
    # 创建环境
    env = TrajectoryTrackingEnv(
        enable_visualization=True,
        trajectory_length=50,
        trajectory_duration=5.0
    )
    
    # 重置环境
    obs, info = env.reset(trajectory_type="random")  # 总是使用random
    print(f"轨迹类型: {info['trajectory_type']}")
    print(f"轨迹长度: {info['trajectory_length']}")
    
    # 运行几步
    for i in range(100):
        action = np.random.uniform(-0.1, 0.1, 3)  # 小的随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"\n步骤 {i}:")
            print(f"  位置误差: {info['pos_error']:.4f}m")
            print(f"  速度误差: {info['vel_error']:.4f}m/s") 
            print(f"  轨迹进度: {info['trajectory_progress']:.1%}")
            print(f"  总奖励: {reward:.3f}")
            
        if terminated or truncated:
            print(f"\nEpisode结束 - 轨迹完成!")
            break
    
    env.close()
