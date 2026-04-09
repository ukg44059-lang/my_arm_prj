"""
Obstacle Controller Manager
管理动态障碍物的控制和运动
"""

import numpy as np
import mujoco


class ObstacleController:
    """
    动态障碍物控制管理器

    管理多个球形障碍物的往复运动，支持独立的频率、幅度和相位设置
    """

    def __init__(self, num_obstacles=2, model=None):
        """
        初始化障碍物控制器

        Args:
            num_obstacles: 障碍物数量
            model: MuJoCo 模型对象（用于自动推导控制索引）
        """
        self.num_obstacles = num_obstacles

        # 自动推导障碍物控制索引
        if model is not None:
            self.control_indices = self._find_obstacle_control_indices(model, num_obstacles)
        else:
            # 默认假设没有夹爪（机械臂6个关节后直接是障碍物）
            self.control_indices = list(range(6, 6 + num_obstacles))

        # 为每个障碍物存储参数
        self.frequencies = [1.5] * num_obstacles  # 频率 (Hz)
        self.amplitudes = [0.8] * num_obstacles   # 幅度 (0-1)
        self.phases = [0.0] * num_obstacles       # 相位 (弧度)
        
        # 障碍物启用/禁用标志
        self.obstacles_enabled = True  # 全局启用/禁用所有障碍物
        
        # 障碍物原始位置存储（用于恢复）
        self.original_positions = {}
        self.underground_depth = -10.0  # 地平线以下的深度
        
        # 存储模型引用以便操作障碍物位置
        self.model = model

    def store_original_positions(self, data):
        """
        存储障碍物的原始位置和关节状态（在第一次禁用前调用）
        
        Args:
            data: MuJoCo 数据对象
        """
        if self.model is None:
            return
            
        for i in range(self.num_obstacles):
            obstacle_name = f"obstacle{i+1}"
            body_id = mujoco.mj_name2id(
                self.model, 
                mujoco.mjtObj.mjOBJ_BODY, 
                obstacle_name
            )
            
            # 查找对应的关节
            joint_name = f"obstacle{i+1}_x" if i == 0 else f"obstacle{i+1}_y"  # obstacle1使用X轴，obstacle2使用Y轴
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name
            )
            
            if body_id >= 0 and joint_id >= 0 and obstacle_name not in self.original_positions:
                # 存储原始位置和关节状态
                self.original_positions[obstacle_name] = {
                    'body_id': body_id,
                    'joint_id': joint_id,
                    'joint_name': joint_name,
                    'position': data.xpos[body_id].copy(),
                    'joint_pos': data.qpos[joint_id]  # 存储关节位置
                }
                print(f"✓ 存储障碍物 {obstacle_name} 原始状态:")
                print(f"  位置: {data.xpos[body_id]}")
                print(f"  关节位置: {data.qpos[joint_id]}")

    def move_obstacles_underground(self, data):
        """
        将所有障碍物移动到地平线以下（通过修改关节位置）
        
        Args:
            data: MuJoCo 数据对象
        """
        if self.model is None:
            return
            
        # 首先存储原始位置（如果还没存储）
        self.store_original_positions(data)
        
        for obstacle_name, info in self.original_positions.items():
            joint_id = info['joint_id']
            joint_name = info['joint_name']
            
            # 根据关节类型设置位置
            if 'x' in joint_name:
                # X轴滑动关节，移动到很远的X位置
                data.qpos[joint_id] = 2.0  # 移动到远离机械臂的X位置
            elif 'y' in joint_name:
                # Y轴滑动关节，移动到很远的Y位置  
                data.qpos[joint_id] = 2.0  # 移动到远离机械臂的Y位置
            
        # 重新计算正向动力学以更新位置
        mujoco.mj_forward(self.model, data)

    def restore_obstacles_position(self, data):
        """
        恢复障碍物到原始位置
        
        Args:
            data: MuJoCo 数据对象
        """
        if self.model is None:
            return
            
        for obstacle_name, info in self.original_positions.items():
            joint_id = info['joint_id']
            original_joint_pos = info['joint_pos']
            
            # 恢复到原始关节位置
            data.qpos[joint_id] = original_joint_pos
            print(f"🔄 恢复 {obstacle_name} 关节位置: {original_joint_pos}")
            
        # 重新计算正向动力学以更新位置
        mujoco.mj_forward(self.model, data)

    def _find_obstacle_control_indices(self, model, num_obstacles):
        """
        从 MuJoCo 模型中自动查找障碍物控制器的索引

        Args:
            model: MuJoCo 模型对象
            num_obstacles: 障碍物数量

        Returns:
            list: 障碍物控制器的索引列表
        """
        obstacle_indices = []

        # 查找所有以 "obstacle" 开头的执行器
        for i in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name and "obstacle" in actuator_name.lower():
                obstacle_indices.append(i)

        if len(obstacle_indices) >= num_obstacles:
            return obstacle_indices[:num_obstacles]
        else:
            print(f"⚠ Warning: Found {len(obstacle_indices)} obstacle actuators, expected {num_obstacles}")
            print(f"  Using default indices starting from 6")
            return list(range(6, 6 + num_obstacles))

    def update(self, data, sim_time):
        """
        更新障碍物控制信号

        Args:
            data: MuJoCo 数据对象
            sim_time: 当前仿真时间
        """
        if not self.obstacles_enabled:
            # 如果障碍物被禁用，将所有控制信号设为0并移动到地下
            for i in range(self.num_obstacles):
                data.ctrl[self.control_indices[i]] = 0.0
            self.move_obstacles_underground(data)
            return
            
        # 障碍物启用时，先确保它们在正确位置
        if len(self.original_positions) > 0:
            # 检查是否有障碍物在远处，如果有则恢复
            for obstacle_name, info in self.original_positions.items():
                joint_id = info['joint_id']
                original_joint_pos = info['joint_pos']
                current_joint_pos = data.qpos[joint_id]
                
                # 如果关节位置偏离原始位置太远（说明在远处）
                if abs(current_joint_pos - original_joint_pos) > 2.0:
                    self.restore_obstacles_position(data)
                    break
            
        # 正常更新障碍物运动
        for i in range(self.num_obstacles):
            cmd = self.amplitudes[i] * np.sin(
                2 * np.pi * self.frequencies[i] * sim_time + self.phases[i]
            )
            data.ctrl[self.control_indices[i]] = cmd

    def set_frequency(self, obstacle_idx, frequency):
        """
        设置障碍物的往复频率

        Args:
            obstacle_idx: 障碍物索引 (0 或 1)
            frequency: 频率 (Hz)
        """
        if 0 <= obstacle_idx < self.num_obstacles:
            self.frequencies[obstacle_idx] = frequency

    def set_amplitude(self, obstacle_idx, amplitude):
        """
        设置障碍物的运动幅度

        Args:
            obstacle_idx: 障碍物索引 (0 或 1)
            amplitude: 幅度 (0-1)
        """
        if 0 <= obstacle_idx < self.num_obstacles:
            self.amplitudes[obstacle_idx] = np.clip(amplitude, 0, 1)

    def enable_obstacles(self, data=None):
        """
        启用所有障碍物运动
        
        Args:
            data: MuJoCo 数据对象，用于恢复位置
        """
        was_disabled = not self.obstacles_enabled
        self.obstacles_enabled = True
        
        # 如果之前被禁用，恢复位置
        if was_disabled and data is not None:
            self.restore_obstacles_position(data)
            
        print("✓ 障碍物运动已启用")

    def disable_obstacles(self, data=None):
        """
        禁用所有障碍物运动
        
        Args:
            data: MuJoCo 数据对象，用于移动到地下
        """
        was_enabled = self.obstacles_enabled
        self.obstacles_enabled = False
        
        # 如果之前启用，移动到地下
        if was_enabled and data is not None:
            # 首先存储原始位置（如果还没存储）
            self.store_original_positions(data)
            
            # 显示具体的障碍物移动信息
            for obstacle_name, info in self.original_positions.items():
                joint_name = info['joint_name']
                print(f"🕳 将 {obstacle_name} 移动到远处 ({joint_name} = 2.0)")
            
            self.move_obstacles_underground(data)
            
        print("✓ 障碍物运动已禁用")

    def toggle_obstacles(self, data=None):
        """
        切换障碍物运动状态
        
        Args:
            data: MuJoCo 数据对象，用于位置操作
        """
        if self.obstacles_enabled:
            self.disable_obstacles(data)
        else:
            self.enable_obstacles(data)
        return self.obstacles_enabled

    def is_obstacles_enabled(self):
        """返回当前障碍物启用状态"""
        return self.obstacles_enabled

    def set_phase(self, obstacle_idx, phase):
        """
        设置障碍物的相位偏移

        Args:
            obstacle_idx: 障碍物索引 (0 或 1)
            phase: 相位 (弧度)
        """
        if 0 <= obstacle_idx < self.num_obstacles:
            self.phases[obstacle_idx] = phase

    def set_all_frequencies(self, frequency):
        """
        设置所有障碍物的频率

        Args:
            frequency: 频率 (Hz)
        """
        for i in range(self.num_obstacles):
            self.frequencies[i] = frequency

    def set_all_amplitudes(self, amplitude):
        """
        设置所有障碍物的幅度

        Args:
            amplitude: 幅度 (0-1)
        """
        amplitude = np.clip(amplitude, 0, 1)
        for i in range(self.num_obstacles):
            self.amplitudes[i] = amplitude

    def reset_controls(self, data):
        """
        重置所有控制信号为 0

        Args:
            data: MuJoCo 数据对象
        """
        for ctrl_idx in self.control_indices:
            data.ctrl[ctrl_idx] = 0.0

    def get_status(self):
        """
        获取障碍物控制器状态

        Returns:
            dict: 包含所有障碍物的参数信息
        """
        status = {
            "num_obstacles": self.num_obstacles,
            "obstacles": []
        }

        for i in range(self.num_obstacles):
            obstacle_info = {
                "index": i,
                "frequency_hz": self.frequencies[i],
                "amplitude": self.amplitudes[i],
                "phase_rad": self.phases[i],
                "control_index": self.control_indices[i]
            }
            status["obstacles"].append(obstacle_info)

        return status

    def print_status(self):
        """打印障碍物控制器状态信息"""
        status = self.get_status()
        print(f"\n{'='*60}")
        print(f"Obstacle Controller Status")
        print(f"{'='*60}")

        for obs_info in status["obstacles"]:
            print(f"\nObstacle {obs_info['index']}:")
            print(f"  - Frequency: {obs_info['frequency_hz']:.2f} Hz")
            print(f"  - Amplitude: {obs_info['amplitude']:.2f}")
            print(f"  - Phase: {obs_info['phase_rad']:.4f} rad")
            print(f"  - Control Index: {obs_info['control_index']}")

        print(f"\n{'='*60}\n")
