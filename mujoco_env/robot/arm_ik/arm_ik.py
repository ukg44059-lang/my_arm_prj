"""
逆运动学求解器 — 基于 MuJoCo 雅可比的迭代 IK

★ 注意:
  目标坐标是 MuJoCo 世界坐标系下的绝对坐标。
  机械臂基座在世界坐标 base pos="1.0 0 0.96" 处。
  工作空间是以基座为中心的空心球壳（不是实心球）。
  调用 get_workspace_info() 查看具体范围。
"""

import numpy as np
import mujoco
from typing import Optional, Tuple


class MuJoCoIKSolver:

    def __init__(
        self, robot,
        max_iterations: int = 300,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-2,
        num_restarts: int = 8,
    ):
        """
        Args:
            robot: Robot 实例
            max_iterations: 每次尝试的最大迭代次数
            position_tolerance: 位置收敛阈值 (m)
            num_restarts: IK 失败时的随机重启次数
        """
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.ee_site_id = robot.ee_site_id
        self.num_joints = robot.num_joints

        self.max_iterations = max_iterations
        self.pos_tol = position_tolerance
        self.ori_tol = orientation_tolerance
        self.num_restarts = num_restarts

        # ★ 通过 joint name 查找真实地址
        self._qpos_addrs = []
        self._dof_addrs = []
        self._jnt_ranges = []
        for i in range(self.num_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            assert jid >= 0, f"找不到 joint{i+1}"
            self._qpos_addrs.append(self.model.jnt_qposadr[jid])
            self._dof_addrs.append(self.model.jnt_dofadr[jid])
            self._jnt_ranges.append(self.model.jnt_range[jid].copy())

        self._qpos_addrs = np.array(self._qpos_addrs)
        self._dof_addrs = np.array(self._dof_addrs)
        self.joint_limits = np.array(self._jnt_ranges)

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        # 基座位置
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.base_pos = self.data.xpos[bid].copy() if bid >= 0 else np.zeros(3)

        # 工作空间信息（延迟初始化）
        self._ws_info = None

    # ---- 读写关节角 ----
    def _get_arm_qpos(self):
        return self.data.qpos[self._qpos_addrs].copy()

    def _set_arm_qpos(self, q):
        self.data.qpos[self._qpos_addrs] = q

    def _extract_jac_cols(self, full_jac):
        return full_jac[:, self._dof_addrs].copy()

    # ------------------------------------------------------------------
    #  工作空间
    # ------------------------------------------------------------------
    def get_workspace_info(self, n_samples=5000):
        if self._ws_info is not None:
            return self._ws_info

        original = self.data.qpos.copy()
        home_pos = self.data.site_xpos[self.ee_site_id].copy()

        positions = []
        for _ in range(n_samples):
            q = np.array([np.random.uniform(lo, hi) for lo, hi in self.joint_limits])
            self._set_arm_qpos(q)
            mujoco.mj_forward(self.model, self.data)
            positions.append(self.data.site_xpos[self.ee_site_id].copy())

        self.data.qpos[:] = original
        mujoco.mj_forward(self.model, self.data)

        positions = np.array(positions)
        dists = np.linalg.norm(positions - self.base_pos, axis=1)

        self._ws_info = {
            'base_position': self.base_pos.copy(),
            'home_ee_position': home_pos,
            'workspace_min': positions.min(axis=0),
            'workspace_max': positions.max(axis=0),
            'workspace_center': (positions.min(axis=0) + positions.max(axis=0)) / 2,
            'inner_radius': float(np.percentile(dists, 2)),   # 去掉极端值
            'outer_radius': float(np.percentile(dists, 98)),
            'min_z': float(np.percentile(positions[:, 2], 2)),
            'max_z': float(np.percentile(positions[:, 2], 98)),
        }

        ws = self._ws_info
        print("\n" + "=" * 60)
        print("  机械臂工作空间 (世界坐标)")
        print("=" * 60)
        print(f"  基座:   [{ws['base_position'][0]:.3f}, {ws['base_position'][1]:.3f}, {ws['base_position'][2]:.3f}]")
        print(f"  初始EE: [{ws['home_ee_position'][0]:.3f}, {ws['home_ee_position'][1]:.3f}, {ws['home_ee_position'][2]:.3f}]")
        print(f"  可达距离: {ws['inner_radius']:.3f}m ~ {ws['outer_radius']:.3f}m (到基座距离)")
        print(f"  高度范围: Z = [{ws['min_z']:.3f} ~ {ws['max_z']:.3f}]")
        print(f"  坐标范围:")
        print(f"    X: [{ws['workspace_min'][0]:.3f} ~ {ws['workspace_max'][0]:.3f}]")
        print(f"    Y: [{ws['workspace_min'][1]:.3f} ~ {ws['workspace_max'][1]:.3f}]")
        print(f"    Z: [{ws['workspace_min'][2]:.3f} ~ {ws['workspace_max'][2]:.3f}]")
        print("=" * 60)

        return ws

    def check_reachable(self, target_pos):
        """快速检查目标点是否大致可达"""
        ws = self.get_workspace_info()
        d = np.linalg.norm(np.asarray(target_pos) - ws['base_position'])
        if d < ws['inner_radius'] * 0.8:
            return False, f"太近 (距基座{d:.3f}m < 最小{ws['inner_radius']:.3f}m)"
        if d > ws['outer_radius'] * 1.1:
            return False, f"太远 (距基座{d:.3f}m > 最大{ws['outer_radius']:.3f}m)"
        if target_pos[2] < ws['min_z'] - 0.05:
            return False, f"太低 (Z={target_pos[2]:.3f} < {ws['min_z']:.3f})"
        return True, "可达"

    # ------------------------------------------------------------------
    #  单次 IK 尝试
    # ------------------------------------------------------------------
    def _solve_once(self, target_pos, target_quat, q_init, verbose=False):
        """单次 IK 迭代求解"""
        use_ori = target_quat is not None
        original = self.data.qpos.copy()
        q = q_init.copy()
        best_q, best_err = q.copy(), np.inf
        damping = 0.01

        for it in range(self.max_iterations):
            self._set_arm_qpos(q)
            mujoco.mj_forward(self.model, self.data)

            pos = self.data.site_xpos[self.ee_site_id].copy()
            pos_err = target_pos - pos
            pos_err_n = float(np.linalg.norm(pos_err))

            if use_ori:
                ori_err = self._orientation_error(target_quat)
                ori_err_n = float(np.linalg.norm(ori_err))
            else:
                ori_err, ori_err_n = np.zeros(3), 0.0

            total = pos_err_n + ori_err_n
            if total < best_err:
                best_err = total
                best_q = q.copy()

            if pos_err_n < self.pos_tol and ((not use_ori) or ori_err_n < self.ori_tol):
                self.data.qpos[:] = original
                mujoco.mj_forward(self.model, self.data)
                return True, best_q, best_err

            # Jacobian
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.ee_site_id)
            Jp = self._extract_jac_cols(self._jacp)
            if use_ori:
                Jr = self._extract_jac_cols(self._jacr)
                J = np.vstack([Jp, Jr])
                e = np.concatenate([pos_err, ori_err])
            else:
                J, e = Jp, pos_err

            # DLS
            n = J.shape[0]
            try:
                dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(n), e)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(J) @ e

            dq = np.clip(dq, -0.15, 0.15)
            q = np.clip(q + dq, self.joint_limits[:, 0], self.joint_limits[:, 1])

        self.data.qpos[:] = original
        mujoco.mj_forward(self.model, self.data)
        return False, best_q, best_err

    # ------------------------------------------------------------------
    #  主求解接口（含多次随机重启）
    # ------------------------------------------------------------------
    def solve(self, target_pos, target_quat=None, q_init=None, verbose=False):
        target_pos = np.asarray(target_pos, dtype=np.float64).flatten()
        if target_quat is not None:
            target_quat = np.asarray(target_quat, dtype=np.float64).flatten()
            target_quat /= (np.linalg.norm(target_quat) + 1e-12)

        # 第一次用给定初始值
        if q_init is None:
            q_init = self._get_arm_qpos()

        if verbose:
            print(f"    [IK] 从当前关节角开始...")

        ok, best_q, best_err = self._solve_once(target_pos, target_quat, q_init, verbose)
        if ok:
            if verbose: print(f"    [IK] ✓ 首次求解成功, err={best_err*1000:.2f}mm")
            return True, best_q

        # 多次随机重启
        if verbose:
            print(f"    [IK] 首次未收敛 (err={best_err*1000:.1f}mm), 尝试 {self.num_restarts} 次随机重启...")

        for trial in range(self.num_restarts):
            q_rand = np.array([np.random.uniform(lo, hi) for lo, hi in self.joint_limits])
            ok_t, q_t, err_t = self._solve_once(target_pos, target_quat, q_rand)
            if ok_t:
                if verbose: print(f"    [IK] ✓ 重启 #{trial+1} 成功, err={err_t*1000:.2f}mm")
                return True, q_t
            if err_t < best_err:
                best_err = err_t
                best_q = q_t.copy()

        if verbose:
            print(f"    [IK] ⚠ {self.num_restarts}次重启后最优 err={best_err*1000:.2f}mm")

        return best_err < self.pos_tol * 10, best_q

    def solve_and_verify(self, target_pos, target_quat=None, q_init=None, verbose=False):
        success, q = self.solve(target_pos, target_quat, q_init, verbose)

        original = self.data.qpos.copy()
        self._set_arm_qpos(q)
        mujoco.mj_forward(self.model, self.data)
        fk_pos = self.data.site_xpos[self.ee_site_id].copy()
        mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        fk_quat = np.empty(4); mujoco.mju_mat2Quat(fk_quat, mat.flatten())
        self.data.qpos[:] = original
        mujoco.mj_forward(self.model, self.data)

        return {
            'success': success,
            'joint_angles': q.copy(),
            'joint_angles_deg': np.rad2deg(q),
            'fk_position': fk_pos,
            'position_error': float(np.linalg.norm(np.asarray(target_pos).flatten() - fk_pos)),
        }

    # ------------------------------------------------------------------
    #  生成可达的示例目标点
    # ------------------------------------------------------------------
    def generate_reachable_targets(self, n=5):
        """通过 FK 生成保证可达的目标点"""
        original = self.data.qpos.copy()
        targets = []

        for _ in range(n * 3):
            if len(targets) >= n:
                break
            q = np.array([np.random.uniform(lo, hi) for lo, hi in self.joint_limits])
            self._set_arm_qpos(q)
            mujoco.mj_forward(self.model, self.data)
            pos = self.data.site_xpos[self.ee_site_id].copy()
            # 过滤掉太低或太极端的点
            if pos[2] > self.base_pos[2] - 0.1:
                targets.append(pos)

        self.data.qpos[:] = original
        mujoco.mj_forward(self.model, self.data)
        return targets[:n]

    # ---- 内部 ----
    def _orientation_error(self, tq):
        Rc = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        Rt = np.empty(9); mujoco.mju_quat2Mat(Rt, tq); Rt = Rt.reshape(3, 3)
        Re = Rt @ Rc.T
        tr = np.clip((np.trace(Re)-1)/2, -1, 1); a = np.arccos(tr)
        if abs(a) < 1e-8: return np.zeros(3)
        ax = np.array([Re[2,1]-Re[1,2], Re[0,2]-Re[2,0], Re[1,0]-Re[0,1]])
        n = np.linalg.norm(ax)
        return (ax/n*a) if n > 1e-8 else np.zeros(3)

    def get_current_ee_pose(self):
        return self.robot.get_ee_position(), self.robot.get_ee_orientation()

    def print_joint_limits(self):
        print("\n  ARM620 关节限位:")
        for i in range(self.num_joints):
            lo, hi = np.rad2deg(self.joint_limits[i])
            print(f"    Joint{i+1}: [{lo:7.1f}° ~ {hi:6.1f}°]  qpos[{self._qpos_addrs[i]}]")
