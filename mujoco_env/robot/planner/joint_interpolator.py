"""
关节角度插值器
Joint Angle Interpolator using Cubic Polynomial
"""

import numpy as np


class CubicJointInterpolator:
    """三次多项式关节插值器"""

    def __init__(self, n_steps: int = 10):
        """
        初始化插值器

        Args:
            n_steps: 插值步数（从当前位置到目标位置的步数）
        """
        self.n_steps = n_steps

    def interpolate(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        dq_start: np.ndarray = None,
        dq_end: np.ndarray = None
    ) -> np.ndarray:
        """
        使用三次多项式插值生成关节轨迹

        三次多项式形式: q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        边界条件:
            q(0) = q_start
            q(1) = q_end
            dq(0) = dq_start (默认为0)
            dq(1) = dq_end (默认为0)

        Args:
            q_start: 起始关节角度 (n_joints,)
            q_end: 目标关节角度 (n_joints,)
            dq_start: 起始关节速度 (n_joints,) 默认为0
            dq_end: 结束关节速度 (n_joints,) 默认为0

        Returns:
            trajectory: 插值轨迹 (n_steps, n_joints)
        """
        n_joints = len(q_start)

        if dq_start is None:
            dq_start = np.zeros(n_joints)
        if dq_end is None:
            dq_end = np.zeros(n_joints)

        # 计算三次多项式系数
        # q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # dq(t) = a1 + 2*a2*t + 3*a3*t^2

        # 边界条件:
        # q(0) = q_start  => a0 = q_start
        # q(1) = q_end    => a0 + a1 + a2 + a3 = q_end
        # dq(0) = dq_start => a1 = dq_start
        # dq(1) = dq_end   => a1 + 2*a2 + 3*a3 = dq_end

        a0 = q_start
        a1 = dq_start
        a2 = 3 * (q_end - q_start) - 2 * dq_start - dq_end
        a3 = -2 * (q_end - q_start) + dq_start + dq_end

        # 生成时间序列 t ∈ [0, 1]
        t = np.linspace(0, 1, self.n_steps)

        # 计算轨迹 q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        trajectory = np.zeros((self.n_steps, n_joints))
        for i, ti in enumerate(t):
            trajectory[i] = a0 + a1 * ti + a2 * ti**2 + a3 * ti**3

        return trajectory

    def interpolate_with_velocity(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        dq_start: np.ndarray = None,
        dq_end: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        生成关节轨迹及对应的速度轨迹

        Args:
            q_start: 起始关节角度
            q_end: 目标关节角度
            dq_start: 起始关节速度
            dq_end: 结束关节速度

        Returns:
            q_trajectory: 位置轨迹 (n_steps, n_joints)
            dq_trajectory: 速度轨迹 (n_steps, n_joints)
        """
        n_joints = len(q_start)

        if dq_start is None:
            dq_start = np.zeros(n_joints)
        if dq_end is None:
            dq_end = np.zeros(n_joints)

        # 三次多项式系数
        a0 = q_start
        a1 = dq_start
        a2 = 3 * (q_end - q_start) - 2 * dq_start - dq_end
        a3 = -2 * (q_end - q_start) + dq_start + dq_end

        # 时间序列
        t = np.linspace(0, 1, self.n_steps)

        # 计算位置和速度轨迹
        q_trajectory = np.zeros((self.n_steps, n_joints))
        dq_trajectory = np.zeros((self.n_steps, n_joints))

        for i, ti in enumerate(t):
            q_trajectory[i] = a0 + a1 * ti + a2 * ti**2 + a3 * ti**3
            dq_trajectory[i] = a1 + 2 * a2 * ti + 3 * a3 * ti**2

        return q_trajectory, dq_trajectory


class QuinticJointInterpolator:
    """五次多项式关节插值器（更平滑）"""

    def __init__(self, n_steps: int = 10):
        """
        初始化插值器

        Args:
            n_steps: 插值步数
        """
        self.n_steps = n_steps

    def interpolate(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        dq_start: np.ndarray = None,
        dq_end: np.ndarray = None,
        ddq_start: np.ndarray = None,
        ddq_end: np.ndarray = None
    ) -> np.ndarray:
        """
        使用五次多项式插值生成关节轨迹

        五次多项式: q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5

        边界条件:
            q(0) = q_start, q(1) = q_end
            dq(0) = dq_start, dq(1) = dq_end
            ddq(0) = ddq_start, ddq(1) = ddq_end

        Args:
            q_start: 起始关节角度
            q_end: 目标关节角度
            dq_start: 起始速度 (默认0)
            dq_end: 结束速度 (默认0)
            ddq_start: 起始加速度 (默认0)
            ddq_end: 结束加速度 (默认0)

        Returns:
            trajectory: 插值轨迹 (n_steps, n_joints)
        """
        n_joints = len(q_start)

        if dq_start is None:
            dq_start = np.zeros(n_joints)
        if dq_end is None:
            dq_end = np.zeros(n_joints)
        if ddq_start is None:
            ddq_start = np.zeros(n_joints)
        if ddq_end is None:
            ddq_end = np.zeros(n_joints)

        # 五次多项式系数
        a0 = q_start
        a1 = dq_start
        a2 = 0.5 * ddq_start
        a3 = 10 * (q_end - q_start) - 6 * dq_start - 4 * dq_end - 1.5 * ddq_start + 0.5 * ddq_end
        a4 = -15 * (q_end - q_start) + 8 * dq_start + 7 * dq_end + 1.5 * ddq_start - ddq_end
        a5 = 6 * (q_end - q_start) - 3 * dq_start - 3 * dq_end - 0.5 * ddq_start + 0.5 * ddq_end

        # 时间序列
        t = np.linspace(0, 1, self.n_steps)

        # 计算轨迹
        trajectory = np.zeros((self.n_steps, n_joints))
        for i, ti in enumerate(t):
            trajectory[i] = (a0 + a1 * ti + a2 * ti**2 + a3 * ti**3 +
                           a4 * ti**4 + a5 * ti**5)

        return trajectory


# 便捷函数
def cubic_interpolate(q_start, q_end, n_steps=10, dq_start=None, dq_end=None):
    """
    便捷函数：三次插值

    Args:
        q_start: 起始关节角度
        q_end: 目标关节角度
        n_steps: 插值步数
        dq_start: 起始速度
        dq_end: 结束速度

    Returns:
        trajectory: 插值轨迹 (n_steps, n_joints)
    """
    interpolator = CubicJointInterpolator(n_steps)
    return interpolator.interpolate(q_start, q_end, dq_start, dq_end)


def quintic_interpolate(q_start, q_end, n_steps=10, dq_start=None, dq_end=None,
                        ddq_start=None, ddq_end=None):
    """
    便捷函数：五次插值

    Args:
        q_start: 起始关节角度
        q_end: 目标关节角度
        n_steps: 插值步数
        dq_start: 起始速度
        dq_end: 结束速度
        ddq_start: 起始加速度
        ddq_end: 结束加速度

    Returns:
        trajectory: 插值轨迹 (n_steps, n_joints)
    """
    interpolator = QuinticJointInterpolator(n_steps)
    return interpolator.interpolate(q_start, q_end, dq_start, dq_end, ddq_start, ddq_end)
