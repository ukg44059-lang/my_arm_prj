"""
扭矩控制器 (Torque Controller)

实现 PID 控制 + 重力补偿的混合控制策略。
输出扭矩 = PID 扭矩 + 重力补偿扭矩
"""

import numpy as np
from typing import Optional
from .gravity_compensation import GravityCompensationController
from .pid_config_loader import load_pid_config, save_pid_config, PIDConfigLoader


class TorqueController:
    """
    混合扭矩控制器：PID + 重力补偿

    该控制器：
    1. 计算位置/速度的 PID 控制扭矩
    2. 计算重力补偿扭矩
    3. 合并两者作为最终的控制扭矩输出
    """

    def __init__(
        self,
        model_path: str,
        kp: float = 500.0,
        kd: float = 30.0,
        ki: float = 120.0,
        use_gravity_compensation: bool = True,
        gravity: Optional[np.ndarray] = None,
        gravity_scale: float = 1.0,
        torque_limits: Optional[np.ndarray] = None,
    ):
        """
        初始化扭矩控制器

        Args:
            model_path: 机器人模型路径 (.urdf)
            kp: 比例增益 (可以是标量或数组[6,])
            kd: 微分增益 (可以是标量或数组[6,])
            ki: 积分增益 (可以是标量或数组[6,], 可选，默认不使用)
            use_gravity_compensation: 是否启用重力补偿
            gravity: 重力加速度向量
            gravity_scale: 重力补偿系数（用于校正模型不匹配，默认1.0）
            torque_limits: 力矩限制 (可以是标量或数组[6,]), 默认为[49, 49, 49, 9, 9, 9]
        """
        self.use_gravity_compensation = use_gravity_compensation
        self.gravity_scale = gravity_scale

        # 初始化重力补偿控制器
        if use_gravity_compensation:
            self.gravity_comp = GravityCompensationController(model_path, gravity)
            self.n_joints = self.gravity_comp.n_control_joints
        else:
            self.gravity_comp = None
            self.n_joints = 6  # 默认6个关节

        # 设置力矩限制（根据执行器规格）
        if torque_limits is None:
            # 默认力矩限制：Joint 1-3: ±49 N·m, Joint 4-6: ±9 N·m
            self.torque_limits = np.array([49.0, 49.0, 49.0, 9.0, 9.0, 9.0])
        else:
            self.torque_limits = np.atleast_1d(torque_limits)
            if self.torque_limits.size == 1:
                self.torque_limits = np.full(self.n_joints, self.torque_limits[0])
            elif self.torque_limits.size != self.n_joints:
                raise ValueError(f"torque_limits must be a scalar or array of size {self.n_joints}")

        # 将PID增益转换为数组形式（支持每个关节不同的增益）
        self.kp = np.atleast_1d(kp)
        if self.kp.size == 1:
            self.kp = np.full(self.n_joints, self.kp[0])
        elif self.kp.size != self.n_joints:
            raise ValueError(f"kp must be a scalar or array of size {self.n_joints}")

        self.kd = np.atleast_1d(kd)
        if self.kd.size == 1:
            self.kd = np.full(self.n_joints, self.kd[0])
        elif self.kd.size != self.n_joints:
            raise ValueError(f"kd must be a scalar or array of size {self.n_joints}")

        if ki is not None:
            self.ki = np.atleast_1d(ki)
            if self.ki.size == 1:
                self.ki = np.full(self.n_joints, self.ki[0])
            elif self.ki.size != self.n_joints:
                raise ValueError(f"ki must be a scalar or array of size {self.n_joints}")
        else:
            self.ki = np.zeros(self.n_joints)

        # 积分误差累积
        self.integrated_error = np.zeros(self.n_joints)

        # 上一步的误差（用于微分计算）
        self.prev_error = np.zeros(self.n_joints)

        print(f"✓ Torque Controller initialized")
        print(f"  - Kp: {self.kp}")
        print(f"  - Kd: {self.kd}")
        print(f"  - Ki: {self.ki}")
        print(f"  - Torque limits (N·m): {self.torque_limits}")
        print(f"  - Gravity compensation: {use_gravity_compensation}")

    def compute_torque(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_target: np.ndarray,
        dq_target: Optional[np.ndarray] = None,
        dt: float = 0.001,
    ) -> np.ndarray:
        """
        计算控制扭矩

        Args:
            q: 当前关节角度 (6,)
            dq: 当前关节速度 (6,)
            q_target: 目标关节角度 (6,)
            dq_target: 目标关节速度（可选），默认为零
            dt: 时间步长

        Returns:
            tau: 控制扭矩 (6,)
        """
        if dq_target is None:
            dq_target = np.zeros(self.n_joints)

        # 位置误差和速度误差
        error = q_target - q
        error_rate = dq_target - dq

        # PID 控制扭矩
        tau_pid = self.kp * error + self.kd * error_rate

        # 积分项（如果启用）
        if np.any(self.ki > 0):
            self.integrated_error += error * dt
            # 积分饱和限制（防止积分风速）
            self.integrated_error = np.clip(self.integrated_error, -1.5, 1.5)
            tau_pid += self.ki * self.integrated_error

        # 添加重力补偿
        if self.use_gravity_compensation and self.gravity_comp is not None:
            # 需要完整的 q 向量（12维），补充手指关节角度（通常为0）
            q_full = np.zeros(self.gravity_comp.n_joints)
            q_full[:self.n_joints] = q

            dq_full = np.zeros(self.gravity_comp.n_joints)
            dq_full[:self.n_joints] = dq

            tau_gravity = self.gravity_comp.compute_gravity_torque(q_full, dq_full)
            # 应用重力补偿系数（用于校正模型不匹配）
            tau_gravity = tau_gravity * self.gravity_scale
        else:
            tau_gravity = np.zeros(self.n_joints)

        # 最终控制扭矩 = PID 扭矩 + 重力补偿扭矩
        tau = tau_pid + tau_gravity

        # 力矩限幅（防止超过执行器限制）
        tau = np.clip(tau, -self.torque_limits, self.torque_limits)

        # 更新误差历史
        self.prev_error = error.copy()

        return tau

    def reset_integral_error(self):
        """重置积分误差"""
        self.integrated_error = np.zeros(self.n_joints)

    def set_gains(self, kp=None, kd=None, ki=None):
        """
        设置控制增益

        Args:
            kp: 比例增益 (可以是标量或数组[6,])
            kd: 微分增益 (可以是标量或数组[6,])
            ki: 积分增益 (可以是标量或数组[6,], 可选)
        """
        if kp is not None:
            self.kp = np.atleast_1d(kp)
            if self.kp.size == 1:
                self.kp = np.full(self.n_joints, self.kp[0])
            elif self.kp.size != self.n_joints:
                raise ValueError(f"kp must be a scalar or array of size {self.n_joints}")

        if kd is not None:
            self.kd = np.atleast_1d(kd)
            if self.kd.size == 1:
                self.kd = np.full(self.n_joints, self.kd[0])
            elif self.kd.size != self.n_joints:
                raise ValueError(f"kd must be a scalar or array of size {self.n_joints}")

        if ki is not None:
            self.ki = np.atleast_1d(ki)
            if self.ki.size == 1:
                self.ki = np.full(self.n_joints, self.ki[0])
            elif self.ki.size != self.n_joints:
                raise ValueError(f"ki must be a scalar or array of size {self.n_joints}")

        print(f"Gains updated:")
        print(f"  - Kp: {self.kp}")
        print(f"  - Kd: {self.kd}")
        print(f"  - Ki: {self.ki}")

    def enable_gravity_compensation(self):
        """启用重力补偿"""
        if self.gravity_comp is not None:
            self.use_gravity_compensation = True
            print("Gravity compensation enabled")

    def disable_gravity_compensation(self):
        """禁用重力补偿"""
        self.use_gravity_compensation = False
        print("Gravity compensation disabled")

    def set_gravity(self, gravity: np.ndarray):
        """设置重力加速度"""
        if self.gravity_comp is not None:
            self.gravity_comp.set_gravity(gravity)

    def set_joint_gains(self, joint_index: int, kp: float = None, kd: float = None, ki: float = None):
        """
        设置单个关节的PID增益

        Args:
            joint_index: 关节索引 (0-5)
            kp: 比例增益
            kd: 微分增益
            ki: 积分增益
        """
        if joint_index < 0 or joint_index >= self.n_joints:
            raise ValueError(f"Joint index must be in range [0, {self.n_joints-1}]")

        if kp is not None:
            self.kp[joint_index] = kp
        if kd is not None:
            self.kd[joint_index] = kd
        if ki is not None:
            self.ki[joint_index] = ki

        print(f"Joint {joint_index} gains updated:")
        print(f"  - Kp: {self.kp[joint_index]:.2f}")
        print(f"  - Kd: {self.kd[joint_index]:.2f}")
        print(f"  - Ki: {self.ki[joint_index]:.2f}")

    def get_joint_gains(self, joint_index: int) -> dict:
        """
        获取单个关节的PID增益

        Args:
            joint_index: 关节索引 (0-5)

        Returns:
            dict: {'kp': float, 'kd': float, 'ki': float}
        """
        if joint_index < 0 or joint_index >= self.n_joints:
            raise ValueError(f"Joint index must be in range [0, {self.n_joints-1}]")

        return {
            'kp': float(self.kp[joint_index]),
            'kd': float(self.kd[joint_index]),
            'ki': float(self.ki[joint_index])
        }

    def get_all_gains(self) -> dict:
        """
        获取所有关节的PID增益

        Returns:
            dict: {
                'kp': np.ndarray,
                'kd': np.ndarray,
                'ki': np.ndarray,
                'joints': list of dict
            }
        """
        joints = []
        for i in range(self.n_joints):
            joints.append({
                'index': i,
                'kp': float(self.kp[i]),
                'kd': float(self.kd[i]),
                'ki': float(self.ki[i])
            })

        return {
            'kp': self.kp.copy(),
            'kd': self.kd.copy(),
            'ki': self.ki.copy(),
            'joints': joints
        }

    def load_gains_from_dict(self, config: dict):
        """
        从字典加载PID增益

        Args:
            config: 包含增益信息的字典
                可以是全局形式: {'kp': [..], 'kd': [..], 'ki': [..]}
                或关节形式: {'joints': [{'index': 0, 'kp': .., 'kd': .., 'ki': ..}, ...]}
        """
        if 'joints' in config:
            # 从关节列表加载
            for joint_cfg in config['joints']:
                idx = joint_cfg['index']
                if idx < 0 or idx >= self.n_joints:
                    print(f"⚠ Warning: Skipping invalid joint index {idx}")
                    continue

                if 'kp' in joint_cfg:
                    self.kp[idx] = joint_cfg['kp']
                if 'kd' in joint_cfg:
                    self.kd[idx] = joint_cfg['kd']
                if 'ki' in joint_cfg:
                    self.ki[idx] = joint_cfg['ki']
        else:
            # 从全局数组加载
            if 'kp' in config:
                kp = np.atleast_1d(config['kp'])
                if len(kp) == self.n_joints:
                    self.kp = kp.copy()
                else:
                    print(f"⚠ Warning: Kp array length mismatch")

            if 'kd' in config:
                kd = np.atleast_1d(config['kd'])
                if len(kd) == self.n_joints:
                    self.kd = kd.copy()
                else:
                    print(f"⚠ Warning: Kd array length mismatch")

            if 'ki' in config:
                ki = np.atleast_1d(config['ki'])
                if len(ki) == self.n_joints:
                    self.ki = ki.copy()
                else:
                    print(f"⚠ Warning: Ki array length mismatch")

        print("✓ Gains loaded from config")
        print(f"  - Kp: {self.kp}")
        print(f"  - Kd: {self.kd}")
        print(f"  - Ki: {self.ki}")

    def load_config_from_file(self, config_path: str):
        """
        从配置文件加载PID参数

        Args:
            config_path: 配置文件路径 (.yaml, .yml, 或 .json)
        """
        config = load_pid_config(config_path)

        # 加载全局设置
        if 'global' in config:
            global_cfg = config['global']
            if 'use_gravity_compensation' in global_cfg:
                self.use_gravity_compensation = global_cfg['use_gravity_compensation']
            if 'gravity_scale' in global_cfg:
                self.gravity_scale = global_cfg['gravity_scale']

        # 加载关节增益
        self.load_gains_from_dict(config)

        # 重置积分误差
        self.reset_integral_error()

        print(f"✓ PID controller configured from file: {config_path}")

    def save_config_to_file(self, file_path: str, joint_names: list = None):
        """
        保存当前PID参数到配置文件

        Args:
            file_path: 保存路径 (.yaml, .yml, 或 .json)
            joint_names: 关节名称列表（可选）
        """
        config = PIDConfigLoader.create_config_from_arrays(
            kp=self.kp,
            kd=self.kd,
            ki=self.ki,
            joint_names=joint_names,
            use_gravity_compensation=self.use_gravity_compensation,
            gravity_scale=self.gravity_scale
        )

        save_pid_config(config, file_path)
        print(f"✓ PID configuration saved to: {file_path}")
