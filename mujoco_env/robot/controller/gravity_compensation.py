"""
重力补偿控制器 (Gravity Compensation Controller)

基于 Pinocchio 库实现的无重力模式。
通过计算每个关节的重力扭矩，使机器人在自由度空间中失重。
"""

import numpy as np
import pinocchio as pin
from typing import Optional


class GravityCompensationController:
    """
    基于Pinocchio的重力补偿控制器

    该控制器计算抵消重力效应所需的关节扭矩。
    """

    def __init__(
        self,
        model_path: str,
        gravity: Optional[np.ndarray] = None,
    ):
        """
        初始化重力补偿控制器

        Args:
            model_path: 机器人模型文件路径 (.urdf)
            gravity: 重力加速度向量 [gx, gy, gz]，默认为 [0, 0, -9.81]
        """
        # 加载模型
        self.model = pin.buildModelFromUrdf(model_path)
        self.data = self.model.createData()
        self.n_joints = self.model.nv

        # 设置重力向量
        if gravity is None:
            self.gravity = np.array([0.0, 0.0, -9.81])
        else:
            self.gravity = np.array(gravity, dtype=float)

        # 计算所有关节的重力补偿（包括夹爪）
        # 前6个是机械臂关节，后面的是夹爪关节
        self.n_control_joints = 6  # 只控制机械臂的6个关节
        self.n_total_joints = self.n_joints  # 但重力补偿需要考虑所有关节（包括夹爪重量）

        print(f"✓ Gravity Compensation Controller initialized")
        print(f"  - Model: {model_path}")
        print(f"  - Total DOFs: {self.n_joints}")
        print(f"  - Control DOFs: {self.n_control_joints} (机械臂关节)")
        print(f"  - 重力补偿包含夹爪重量影响")
        print(f"  - Gravity: {self.gravity}")

    def compute_gravity_torque(
        self,
        q: np.ndarray,
        dq: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        计算重力补偿扭矩（考虑夹爪重量影响）

        使用逆动力学计算平衡重力所需的静态扭矩。
        原理：当加速度为零时，tau = h(q, dq)（重力项和科里奥利项）

        Args:
            q: 关节角度向量，长度为 n_joints (包括夹爪关节)
            dq: 关节速度向量（可选），默认为零

        Returns:
            tau: 机械臂关节的重力补偿扭矩，长度为 n_control_joints (前6个)
        """
        # 如果输入的q只有6个关节（机械臂），需要扩展到包括夹爪关节
        if len(q) == 6:
            # 扩展q到包括夹爪关节，假设夹爪处于中性位置
            q_full = np.zeros(self.n_joints)
            q_full[:6] = q  # 机械臂关节
            q_full[6:] = 0.0  # 夹爪关节设为中性位置
            q = q_full
        elif len(q) != self.n_joints:
            raise ValueError(f"Expected q with {self.n_joints} or 6 elements, got {len(q)}")

        # 如果没有提供速度，使用零速度
        if dq is None:
            dq = np.zeros(self.n_joints)
        else:
            dq = np.array(dq, dtype=float)
            if len(dq) == 6:
                # 扩展dq到包括夹爪关节
                dq_full = np.zeros(self.n_joints)
                dq_full[:6] = dq
                dq_full[6:] = 0.0
                dq = dq_full
            elif len(dq) != self.n_joints:
                raise ValueError(f"Expected dq with {self.n_joints} or 6 elements, got {len(dq)}")

        # 零加速度（静态平衡）
        ddq = np.zeros(self.n_joints)

        # 设置模型的重力
        self.model.gravity.linear = self.gravity

        # 计算所有关节的逆动力学（包括夹爪重量对机械臂关节的影响）
        tau_full = pin.rnea(self.model, self.data, q, dq, ddq)

        # 只返回机械臂关节的扭矩（前6个）
        # 注意：夹爪的重量会通过运动学链影响机械臂关节的重力补偿
        return tau_full[:self.n_control_joints]

    def set_gravity(self, gravity: np.ndarray):
        """
        设置重力加速度

        Args:
            gravity: 重力加速度向量 [gx, gy, gz]
        """
        self.gravity = np.array(gravity, dtype=float)
        print(f"Gravity set to: {self.gravity}")

    def disable_gravity(self):
        """禁用重力（设置为零）"""
        self.gravity = np.array([0.0, 0.0, 0.0])
        print("Gravity disabled (set to [0, 0, 0])")

    def enable_gravity(self, gravity: Optional[np.ndarray] = None):
        """
        启用重力

        Args:
            gravity: 重力加速度向量，默认为 [0, 0, -9.81]
        """
        if gravity is None:
            gravity = np.array([0.0, 0.0, -9.81])
        self.set_gravity(gravity)

    def get_gravity(self) -> np.ndarray:
        """获取当前的重力加速度向量"""
        return self.gravity.copy()
