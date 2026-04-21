#!/usr/bin/env python3
"""
逆运动学可视化演示 — 在 MuJoCo 仿真窗口中看机械臂运动到目标坐标

    cd mujoco_env
    python3 test/demo_ik_visual.py            # 交互模式
    python3 test/demo_ik_visual.py --demo     # 自动演示
"""

import numpy as np
import mujoco
from mujoco import viewer as mujoco_viewer
import sys, os, time, argparse, threading, importlib.util
from collections import deque


def _find_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(d, 'robot')) and os.path.isdir(os.path.join(d, 'envs')):
            return d
        d = os.path.dirname(d)
    raise FileNotFoundError("找不到项目根目录")

_R = _find_root()
for s in ['robot','robot/planner','robot/controller','tools','sensors','envs']:
    sys.path.insert(0, os.path.join(_R, s))

from robot import Robot
_sp = importlib.util.spec_from_file_location("ik", os.path.join(_R,'robot','arm_ik','arm_ik.py'))
_m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
MuJoCoIKSolver = _m.MuJoCoIKSolver
from joint_interpolator import CubicJointInterpolator


class VisualIKController:

    def __init__(self, model_path):
        print("\n" + "="*60 + "\n  初始化...\n" + "="*60)

        self.robot = Robot(model_path, enable_ros2=False, enable_cameras=False, enable_joint_state_ros2=False)
        self.robot._heartbeat_enabled = False

        self.solver = MuJoCoIKSolver(self.robot, max_iterations=300, position_tolerance=1e-3, num_restarts=10)

        self.initial_joints = np.deg2rad([0, 0, -90, 0, -90, 0])
        self.solver._set_arm_qpos(self.initial_joints)
        mujoco.mj_forward(self.robot.model, self.robot.data)
        self.robot.update_joint_state()

        self.interpolator = CubicJointInterpolator(n_steps=100)

        self.viewer = mujoco_viewer.launch_passive(self.robot.model, self.robot.data)
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        self.viewer.sync()

        self._target_q = self.initial_joints.copy()
        self._traj = deque()

        self.ws = self.solver.get_workspace_info()
        self.solver.print_joint_limits()

        ee = self.robot.get_ee_position()
        print(f"\n  ✓ 就绪! 末端: [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]")

    # ---- sim tick ----
    def _tick(self):
        if self._traj:
            self._target_q = self._traj.popleft()
        self.robot.apply_joint_control(self._target_q)
        self.robot.step()
        self.robot.update_joint_state()
        self.viewer.sync()

    def _run(self, sec):
        for _ in range(int(sec / self.robot.model.opt.timestep)):
            if not self.viewer.is_running(): return
            self._tick()

    def _wait(self, timeout=15):
        t0 = time.time()
        while self._traj and self.viewer.is_running():
            self._tick()
            if time.time()-t0 > timeout: self._traj.clear(); break

    # ---- 运动 ----
    def move_to(self, target_pos, dur=3.0, settle=1.5):
        target_pos = np.asarray(target_pos, dtype=np.float64).flatten()
        print(f"\n  🎯 目标: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")

        # 可达性预检
        ok, msg = self.solver.check_reachable(target_pos)
        if not ok:
            print(f"  ❌ 不可达: {msg}")
            return False

        self._update_marker(target_pos)
        q_cur = self.solver._get_arm_qpos()

        res = self.solver.solve_and_verify(target_pos, q_init=q_cur, verbose=True)
        q_tgt = res['joint_angles']

        s = '✓ 成功' if res['success'] else '⚠ 近似'
        print(f"  IK {s}  误差: {res['position_error']*1000:.2f}mm")
        print(f"  关节(deg): [{', '.join(f'{d:.1f}' for d in res['joint_angles_deg'])}]")

        if not res['success'] and res['position_error'] > 0.05:
            print(f"  ❌ 误差过大, 该点可能不可达")
            return False

        dt = self.robot.model.opt.timestep
        steps = max(int(dur / dt), 500)
        self.interpolator.n_steps = steps
        traj = self.interpolator.interpolate(q_cur, q_tgt)
        self._traj.clear()
        for q in traj: self._traj.append(q)

        print(f"  ▶ 运动中... ({steps}步 ≈ {dur:.1f}s)")
        self._wait(dur + 3)
        self._target_q = q_tgt.copy()
        self._run(settle)

        fp = self.robot.get_ee_position()
        err = np.linalg.norm(target_pos - fp)
        print(f"  ✓ 到达! [{fp[0]:.4f},{fp[1]:.4f},{fp[2]:.4f}] 误差:{err*1000:.2f}mm")
        return True

    def go_home(self, dur=3.0):
        print("\n  🏠 回初始...")
        q_cur = self.solver._get_arm_qpos()
        dt = self.robot.model.opt.timestep
        steps = max(int(dur / dt), 500)
        self.interpolator.n_steps = steps
        traj = self.interpolator.interpolate(q_cur, self.initial_joints)
        self._traj.clear()
        for q in traj: self._traj.append(q)
        self._wait(dur + 3)
        self._target_q = self.initial_joints.copy()
        self._run(1.5)
        ee = self.robot.get_ee_position()
        print(f"  ✓ 末端: [{ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f}]")

    # ---- 自动演示 ----
    def demo(self):
        print("\n" + "="*60)
        print("  🎬 自动演示: 通过FK生成保证可达的5个目标点")
        print("="*60)

        targets = self.solver.generate_reachable_targets(5)
        for i, pos in enumerate(targets):
            if not self.viewer.is_running(): return
            print(f"\n  ── 第{i+1}/5: [{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] ──")
            self.move_to(pos, dur=3.0, settle=1.5)
            if self.viewer.is_running(): self._run(1.0)

        if self.viewer.is_running(): self.go_home()
        print("\n  ✅ 演示完成!")

    # ---- 交互 ----
    def interactive(self):
        ws = self.ws
        base = ws['base_position']

        # 生成3个示例坐标（保证可达）
        examples = self.solver.generate_reachable_targets(3)

        print("\n" + "="*60)
        print("  🎮 交互模式")
        print("="*60)
        print(f"\n  基座位置: [{base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f}]")
        print(f"  可达距离: {ws['inner_radius']:.2f}m ~ {ws['outer_radius']:.2f}m (到基座)")
        print(f"  高度范围: Z = [{ws['min_z']:.2f} ~ {ws['max_z']:.2f}]")
        print(f"  坐标范围:")
        print(f"    X: [{ws['workspace_min'][0]:.2f} ~ {ws['workspace_max'][0]:.2f}]")
        print(f"    Y: [{ws['workspace_min'][1]:.2f} ~ {ws['workspace_max'][1]:.2f}]")
        print(f"    Z: [{ws['workspace_min'][2]:.2f} ~ {ws['workspace_max'][2]:.2f}]")
        print(f"\n  示例 (保证可达):")
        for i, ex in enumerate(examples):
            print(f"    {ex[0]:.2f} {ex[1]:.2f} {ex[2]:.2f}")
        print(f"\n  命令: home | demo | q\n")

        cq = deque()
        stop = threading.Event()
        def _inp():
            while not stop.is_set():
                try:
                    l = input("  🎯 坐标 (x y z): ").strip()
                    if l: cq.append(l)
                except: cq.append("__q"); break
        threading.Thread(target=_inp, daemon=True).start()

        try:
            while self.viewer.is_running():
                if cq:
                    r = cq.popleft()
                    if r == "__q" or r.lower() in ('q','quit','exit'): break
                    elif r.lower() == 'home': self.go_home()
                    elif r.lower() == 'demo': self.demo()
                    else:
                        try:
                            p = r.replace(',', ' ').split()
                            if len(p) != 3: print("  ❌ 请输入3个值")
                            else: self.move_to(np.array([float(x) for x in p]))
                        except ValueError: print("  ❌ 格式错误")
                self._tick()
        except KeyboardInterrupt: pass
        stop.set()

    def _update_marker(self, pos, quat=None):
        try:
            bid = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, "target_body")
            if bid < 0: return
            mid = self.robot.model.body_mocapid[bid]
            if mid < 0: return
            self.robot.data.mocap_pos[mid] = pos
            if quat is not None: self.robot.data.mocap_quat[mid] = quat
            mujoco.mj_forward(self.robot.model, self.robot.data)
        except: pass

    def close(self):
        self.robot._heartbeat_enabled = False
        if self.viewer: self.viewer.close(); self.viewer = None
        self.robot.cleanup()
        print("  ✓ 已退出")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", type=str, default=None)
    pa.add_argument("--demo", action="store_true")
    a = pa.parse_args()

    mp = a.model or os.path.join(_R, "robot_model/exp/env_robot_torque.xml")
    if not os.path.isfile(mp): print(f"❌ {mp}"); return

    c = VisualIKController(mp)
    try:
        if a.demo:
            c.demo()
            print("\n  关闭窗口退出...")
            while c.viewer.is_running(): c._tick()
        else:
            c.interactive()
    except KeyboardInterrupt: pass
    finally: c.close()

if __name__ == "__main__":
    main()
