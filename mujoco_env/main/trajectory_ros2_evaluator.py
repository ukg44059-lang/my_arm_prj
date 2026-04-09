#!/usr/bin/env python3
"""
机械臂任务轨迹评估器
====================
当前主入口为离线评估模式：读取 rosbag2 / mcap，输出评分结果。

离线读取的话题:
    /joint_states 或 /joint_states_sim (sensor_msgs/JointState) - 关节位置/速度/力矩
    /gripper_status (std_msgs/String)                           - idle | grasping | grasped，用于阶段划分

任务阶段划分:
    Phase 1 - approach:   自动记录接近阶段
    Phase 2 - grasping:   gripper_status 变为 "grasping"              → 夹爪开始闭合
    Phase 3 - transport:  gripper_status 变为 "grasped"               → 搬运物体到目标点
    Phase 4 - release:    gripper_status 从 grasped/抓取中 变回 idle   → 放下物体并结束评估

评分公式:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  [每阶段内] 阶段分 = 0.40×平滑度分 + 0.35×稳定性分 + 0.25×力矩效率分    │
    │  [跨阶段]  总体分 = Σ( 阶段权重 × 阶段分 )  （各权重之和 = 1）           │
    │                                                                          │
    │  阶段权重（抓住物体后权重显著提高）:                                       │
    │      approach  (移动到目标): 0.10                                         │
    │      grasping  (抓取中):     0.15                                         │
    │      transport (搬运物体):   0.50   ← 已抓住，最关键                      │
    │      release   (放下物体):   0.25   ← 已抓住，次关键                      │
    └──────────────────────────────────────────────────────────────────────────┘

    子指标计算 (均为 0~100 分):
        平滑度分   = max(0, 100 × (1 - overall_jerk  / 150.0))
        稳定性分   = max(0, 100 × (1 - end_vel_norm  /   0.3))
        力矩效率分 = max(0, 100 × (1 - rms_utilization / 80%))

    参考量含义:
        overall_jerk    : 阶段内所有关节 jerk (Δvel/Δt) 的 RMS 均值
        end_vel_norm    : 阶段结束时刻的关节速度向量范数
        rms_utilization : 阶段内 RMS 力矩 / 力矩限制 的平均利用率

    注意: 若某阶段无数据，其权重按比例重新分配给有数据的阶段。

运行方式:
    # 读取 bag 目录或 mcap 所在目录
    /usr/bin/python3 main/trajectory_ros2_evaluator.py \
      --bag /path/to/rosbag_or_mcap_dir

    # 也可指定输出目录
    /usr/bin/python3 main/trajectory_ros2_evaluator.py \
      --bag /path/to/rosbag_or_mcap_dir \
      --output_dir /tmp/eval_results
"""

import sys
import os
import time
import json
import argparse
import threading
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ==============================================================================
# 常量 & 配置
# ==============================================================================

JOINT_NAMES   = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
TORQUE_LIMITS = np.array([49.0, 49.0, 49.0, 9.0, 9.0, 9.0])  # N·m

# 评分参考值（根据实际机械臂调整）
REF_JERK_MAX       = 150.0   # rad/s³  超过此值 Jerk 评分归 0
REF_END_VEL_MAX    = 0.3     # rad/s   阶段末尾速度超过此值稳定性归 0
REF_EFFORT_MAX_PCT = 80.0    # %       力矩利用率超过此值效率归 0

# 抖动相关阈值（建议由“无抖动理想抓取轨迹”的 3σ 标定得到）
# 当前为默认值，可按机械臂型号与采样频率调整
REF_RMSE_MAX       = 0.12    # rad/s   速度 RMSE 参考上限（默认）
REF_STD_MAX        = 0.08    # rad/s   速度 STD 参考上限（默认）

# 赛题惩罚机制缩放：每次“剧烈抖动”扣分
SEVERE_JITTER_SCORE_THRESHOLD = 50.0   # 任一关节 Jerk 得分 < 50 触发
SEVERE_JITTER_PENALTY_POINTS  = 10.0   # 总分扣 10 分 / 次

# 阶段权重（平滑度 / 稳定性 / 力矩效率）
PHASE_WEIGHTS = {
    'smoothness': 0.40,
    'stability':  0.35,
    'effort':     0.25,
}

# 各阶段在总体评分中的权重（未抓物体权重低，抓住后权重高）
OVERALL_PHASE_WEIGHTS = {
    'approach':  0.10,   # 移动到目标，尚未抓取
    'grasping':  0.15,   # 夹爪闭合中，尚未抓住
    'transport': 0.50,   # 已抓住，搬运最关键
    'release':   0.25,   # 已抓住，放置次关键
}

PHASE_LABELS = {
    'approach':  'Phase 1 · 移动到目标',
    'grasping':  'Phase 2 · 抓取',
    'transport': 'Phase 3 · 搬运',
    'release':   'Phase 4 · 放下',
}

PHASE_ORDER = ['approach', 'grasping', 'transport', 'release']


# ==============================================================================
# 数据结构
# ==============================================================================

@dataclass
class Frame:
    """单步关节状态快照"""
    timestamp:  float
    positions:  np.ndarray   # (6,) rad
    velocities: np.ndarray   # (6,) rad/s
    efforts:    np.ndarray   # (6,) N·m
    phase:      str
    gripper_status: str      # idle | grasping | grasped
    is_grasped_flag: int     # 0/1，便于后处理筛选


@dataclass
class PhaseMetrics:
    """单阶段评价指标"""
    name:            str
    frame_count:     int   = 0
    duration:        float = 0.0   # s

    jerk_per_joint:  np.ndarray = field(default_factory=lambda: np.zeros(6))
    jerk_score_per_joint: np.ndarray = field(default_factory=lambda: np.zeros(6))
    overall_jerk:    float      = 0.0   # rad/s³
    severe_jitter:   bool       = False
    severe_jitter_count: int    = 0

    end_vel_norm:    float      = 0.0   # rad/s  阶段末尾 30 帧速度均值
    effort_rms:      np.ndarray = field(default_factory=lambda: np.zeros(6))
    effort_rms_mean: float      = 0.0   # 各关节 RMS 均值

    score_smoothness: float = 0.0  # 0~100
    score_stability:  float = 0.0
    score_effort:     float = 0.0
    score_total:      float = 0.0


# ==============================================================================
# 评估器核心逻辑
# ==============================================================================

class TrajectoryEvaluator:
    """
    纯计算类，不依赖 ROS2。
    接收帧列表，按阶段切片后计算指标并打印报告。
    """

    def evaluate(self, frames: List[Frame]) -> dict:
        """
        计算所有阶段的指标。

        Args:
            frames: 所有记录帧（按时间顺序）

        Returns:
            dict: phase_name -> PhaseMetrics，以及 'overall_score'
        """
        # 按阶段分组
        phase_frames: dict[str, List[Frame]] = {p: [] for p in PHASE_ORDER}
        for f in frames:
            if f.phase in phase_frames:
                phase_frames[f.phase].append(f)

        results = {}
        for phase_name in PHASE_ORDER:
            pf = phase_frames[phase_name]
            if len(pf) < 5:
                m = PhaseMetrics(name=phase_name, frame_count=len(pf))
                results[phase_name] = m
                continue
            results[phase_name] = self._compute_phase_metrics(phase_name, pf)

        # 综合评分：各阶段按 OVERALL_PHASE_WEIGHTS 加权求和
        # 若某阶段无数据，其权重按比例重新分配给有数据的阶段
        valid_weight_sum = sum(
            OVERALL_PHASE_WEIGHTS[p] for p in PHASE_ORDER
            if results[p].frame_count >= 5
        )
        if valid_weight_sum > 0:
            overall_raw = sum(
                OVERALL_PHASE_WEIGHTS[p] / valid_weight_sum * results[p].score_total
                for p in PHASE_ORDER
                if results[p].frame_count >= 5
            )
        else:
            overall_raw = 0.0

        # 赛题惩罚：任一关节 Jerk 得分 < 50 视为一次“剧烈抖动”（按阶段计次）
        severe_count = sum(
            1 for p in PHASE_ORDER
            if results[p].frame_count >= 5 and results[p].severe_jitter
        )
        penalty = severe_count * SEVERE_JITTER_PENALTY_POINTS
        overall = max(0.0, overall_raw - penalty)

        results['overall_score_raw'] = float(overall_raw)
        results['jitter_penalty_count'] = int(severe_count)
        results['jitter_penalty_points'] = float(penalty)
        results['overall_score'] = float(overall)

        return results

    def _compute_phase_metrics(self, name: str, frames: List[Frame]) -> PhaseMetrics:
        N = len(frames)
        m = PhaseMetrics(name=name, frame_count=N)

        timestamps  = np.array([f.timestamp  for f in frames])
        velocities  = np.array([f.velocities for f in frames])   # (N,6)
        efforts     = np.array([f.efforts    for f in frames])   # (N,6)

        m.duration = float(timestamps[-1] - timestamps[0])

        # ── 平均控制周期
        dts = np.diff(timestamps)
        dt  = float(np.mean(dts)) if len(dts) > 0 else 0.002

        # ── Jerk（速度二阶差分 / dt²）
        if N > 2:
            jerk = np.diff(velocities, n=2, axis=0) / (dt ** 2)   # (N-2, 6)
            m.jerk_per_joint = np.mean(np.abs(jerk), axis=0)
            m.overall_jerk   = float(np.mean(np.abs(jerk)))
            m.jerk_score_per_joint = np.maximum(
                0.0,
                100.0 * (1.0 - m.jerk_per_joint / REF_JERK_MAX)
            )
            severe_mask = m.jerk_score_per_joint < SEVERE_JITTER_SCORE_THRESHOLD
            m.severe_jitter_count = int(np.sum(severe_mask))
            m.severe_jitter = m.severe_jitter_count > 0

        # ── 末尾稳定性（最后 min(30, N//5) 帧的速度范数均值）
        tail = max(5, min(30, N // 5))
        end_vels = velocities[-tail:]
        m.end_vel_norm = float(np.mean(np.linalg.norm(end_vels, axis=1)))

        # ── 力矩 RMS
        m.effort_rms      = np.sqrt(np.mean(efforts ** 2, axis=0))
        utilization       = m.effort_rms / TORQUE_LIMITS * 100.0  # %
        m.effort_rms_mean = float(np.mean(utilization))

        # ── 评分
        m.score_smoothness = max(0.0, 100.0 * (1.0 - m.overall_jerk / REF_JERK_MAX))
        m.score_stability  = max(0.0, 100.0 * (1.0 - m.end_vel_norm / REF_END_VEL_MAX))
        m.score_effort     = max(0.0, 100.0 * (1.0 - m.effort_rms_mean / REF_EFFORT_MAX_PCT))

        m.score_total = (
            PHASE_WEIGHTS['smoothness'] * m.score_smoothness
            + PHASE_WEIGHTS['stability']  * m.score_stability
            + PHASE_WEIGHTS['effort']     * m.score_effort
        )

        return m

    def report(self, results: dict, session_name: str = ""):
        """打印格式化评估报告"""
        sep = '=' * 68
        print(f'\n{sep}')
        print(f'  机械臂任务轨迹评估报告  {session_name}')
        print(sep)

        for phase_name in PHASE_ORDER:
            m: PhaseMetrics = results[phase_name]
            label = PHASE_LABELS[phase_name]

            print(f'\n  [{label}]')
            if m.frame_count < 5:
                print(f'    无数据（帧数: {m.frame_count}）')
                continue

            print(f'    时长: {m.duration:.2f} s    帧数: {m.frame_count}')

            # Jerk
            jerk_str = '  '.join(f'{v:.1f}' for v in m.jerk_per_joint)
            print(f'    Jerk /关节 (rad/s³): {jerk_str}')
            print(f'    综合 Jerk: {m.overall_jerk:.2f} rad/s³'
                  f'   →  平滑度评分: {m.score_smoothness:.1f}/100')
            jerk_score_str = '  '.join(f'{v:.1f}' for v in m.jerk_score_per_joint)
            print(f'    Jerk得分/关节: {jerk_score_str}')
            if m.severe_jitter:
                print(f'    ⚠ 剧烈抖动: 是（低于{SEVERE_JITTER_SCORE_THRESHOLD:.0f}分的关节数: {m.severe_jitter_count}）')
            else:
                print('    剧烈抖动: 否')

            # 稳定性
            print(f'    末尾速度范数: {m.end_vel_norm:.4f} rad/s'
                  f'   →  稳定性评分: {m.score_stability:.1f}/100')

            # 力矩
            rms_str = '  '.join(f'{v:.1f}' for v in m.effort_rms)
            print(f'    力矩 RMS (N·m): {rms_str}')
            print(f'    平均利用率: {m.effort_rms_mean:.1f}%'
                  f'   →  效率评分: {m.score_effort:.1f}/100')

            # 阶段分数
            w = OVERALL_PHASE_WEIGHTS[phase_name]
            bar_len = int(m.score_total / 100 * 30)
            bar = '[' + '#' * bar_len + '-' * (30 - bar_len) + ']'
            print(f'    阶段评分: {m.score_total:.1f}/100  {bar}  (总体权重 {w:.0%})')

        # 总分
        overall_raw = results.get('overall_score_raw', results.get('overall_score', 0.0))
        penalty_count = results.get('jitter_penalty_count', 0)
        penalty_points = results.get('jitter_penalty_points', 0.0)
        overall = results.get('overall_score', 0.0)
        print(f'\n  {"-" * 50}')
        print(f'  抖动惩罚: {penalty_count} 次  × {SEVERE_JITTER_PENALTY_POINTS:.0f} 分 = -{penalty_points:.1f} 分')
        print(f'  惩罚前总分: {overall_raw:.1f}/100')
        bar_len = int(overall / 100 * 30)
        bar = '[' + '#' * bar_len + '-' * (30 - bar_len) + ']'
        if overall >= 90:
            grade = '无抖动，稳定性优秀'
        elif overall >= 70:
            grade = '轻微抖动，稳定性良好'
        elif overall >= 50:
            grade = '中度抖动，稳定性一般'
        else:
            grade = '严重抖动，触发惩罚'
        print(f'  总体评分: {overall:.1f}/100  {bar}  {grade}')
        print(f'{sep}\n')

    def save_metrics(self, results: dict, path: str):
        """保存评估结果到 JSON"""
        out = {}
        for phase_name in PHASE_ORDER:
            m = results[phase_name]
            out[phase_name] = {
                'frame_count':      m.frame_count,
                'duration':         m.duration,
                'overall_jerk':     m.overall_jerk,
                'jerk_per_joint':   m.jerk_per_joint.tolist(),
                'jerk_score_per_joint': m.jerk_score_per_joint.tolist(),
                'severe_jitter':    m.severe_jitter,
                'severe_jitter_count': m.severe_jitter_count,
                'end_vel_norm':     m.end_vel_norm,
                'effort_rms':       m.effort_rms.tolist(),
                'effort_rms_mean':  m.effort_rms_mean,
                'score_smoothness': m.score_smoothness,
                'score_stability':  m.score_stability,
                'score_effort':     m.score_effort,
                'score_total':      m.score_total,
            }
        out['config'] = {
            'REF_JERK_MAX': REF_JERK_MAX,
            'REF_END_VEL_MAX': REF_END_VEL_MAX,
            'REF_EFFORT_MAX_PCT': REF_EFFORT_MAX_PCT,
            'REF_RMSE_MAX': REF_RMSE_MAX,
            'REF_STD_MAX': REF_STD_MAX,
            'SEVERE_JITTER_SCORE_THRESHOLD': SEVERE_JITTER_SCORE_THRESHOLD,
            'SEVERE_JITTER_PENALTY_POINTS': SEVERE_JITTER_PENALTY_POINTS,
        }
        out['overall_score_raw'] = results.get('overall_score_raw', results.get('overall_score', 0.0))
        out['jitter_penalty_count'] = results.get('jitter_penalty_count', 0)
        out['jitter_penalty_points'] = results.get('jitter_penalty_points', 0.0)
        out['overall_score'] = results.get('overall_score', 0.0)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'[Evaluator] 指标已保存: {path}')

    def save_raw_data(self, frames: List[Frame], path: str):
        """保存原始帧到 .npz"""
        if not frames:
            return
        grasped_flags = np.array([f.is_grasped_flag for f in frames], dtype=np.int32)
        np.savez_compressed(
            path,
            timestamps  = np.array([f.timestamp  for f in frames]),
            positions   = np.array([f.positions   for f in frames]),
            velocities  = np.array([f.velocities  for f in frames]),
            efforts     = np.array([f.efforts     for f in frames]),
            phases      = np.array([f.phase       for f in frames]),
            gripper_status = np.array([f.gripper_status for f in frames]),
            is_grasped_flag = grasped_flags,
        )
        n_grasped = int(np.sum(grasped_flags))
        n_free = int(len(frames) - n_grasped)
        print(f'[Evaluator] 原始数据已保存: {path}  ({len(frames)} 帧)')
        print(f'[Evaluator] 轨迹标记统计: free_motion={n_free}, grasped_motion={n_grasped}')


# ==============================================================================
# ROS2 节点
# ==============================================================================

class PhaseState(str, Enum):
    IDLE      = 'idle'
    APPROACH  = 'approach'
    GRASPING  = 'grasping'
    TRANSPORT = 'transport'
    RELEASE   = 'release'
    DONE      = 'done'


# ==============================================================================
# 主函数
# ==============================================================================

def _resolve_bag_uri(input_path: str) -> str:
    p = os.path.abspath(input_path)
    if os.path.isdir(p):
        if os.path.exists(os.path.join(p, 'metadata.yaml')):
            return p
        raise RuntimeError(f'目录 {p} 中无 metadata.yaml，不是有效 rosbag 目录。')
    if os.path.isfile(p):
        if p.endswith(('.mcap', '.mca')):
            parent = os.path.dirname(p)
            if os.path.exists(os.path.join(parent, 'metadata.yaml')):
                return parent
            raise RuntimeError(f'文件 {p} 同目录缺少 metadata.yaml，无法读取。')
        raise RuntimeError(f'文件 {p} 扩展名非 .mcap/.mca，无法识别。')
    raise RuntimeError(f'无效路径（文件或目录不存在）：{p}')


def _load_frames_from_bag(bag_uri: str,
                          joint_topic_candidates: List[str],
                          gripper_topic: str) -> tuple[List[Frame], str, str]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as e:
        print(f"[ERROR] rosbag2_py 导入失败: {e}")
        print("离线读取 rosbag 需要 ROS2 环境，请运行:")
        print("  source /opt/ros/humble/setup.bash")
        raise

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    
    print(f"[Bag Info] Metadata 话题: {list(type_map.keys())}", file=sys.stderr)

    # 先扫描实际 bag 中的话题（可能被重命名）
    actual_topics = set()
    temp_reader = rosbag2_py.SequentialReader()
    temp_reader.open(storage_options, converter_options)
    scan_count = 0
    while temp_reader.has_next() and scan_count < 100:
        topic, _, _ = temp_reader.read_next()
        actual_topics.add(topic)
        scan_count += 1
    
    print(f"[Bag Info] 实际话题（扫描前100条）: {sorted(actual_topics)}", file=sys.stderr)

    # 优先使用实际话题匹配
    joint_topic = None
    for t in joint_topic_candidates:
        if t in actual_topics:
            joint_topic = t
            break
    if joint_topic is None:
        raise RuntimeError(f'未找到关节话题，候选: {joint_topic_candidates}，实际: {sorted(actual_topics)}')
    if gripper_topic not in actual_topics:
        print(f"[Warn] Gripper 话题 {gripper_topic} 不在实际话题中，尝试继续...", file=sys.stderr)

    # 建立实际话题到 type 的映射（处理重命名）
    # 尝试从 metadata 推断：找同类型话题
    actual_type_map = {}
    for actual in actual_topics:
        # 直接匹配
        if actual in type_map:
            actual_type_map[actual] = type_map[actual]
        else:
            # 尝试匹配 JointState 类型（关节话题）
            for meta_topic, meta_type in type_map.items():
                if 'JointState' in meta_type and 'joint' in actual.lower():
                    actual_type_map[actual] = meta_type
                    break
                elif 'String' in meta_type and actual == gripper_topic:
                    actual_type_map[actual] = meta_type
                    break

    print(f"[Bag Info] 实际话题类型映射: {actual_type_map}", file=sys.stderr)

    msg_cache = {}

    def decode(topic: str, payload: bytes):
        if topic not in actual_type_map:
            raise KeyError(f'话题 {topic} 无类型信息')
        if topic not in msg_cache:
            msg_cache[topic] = get_message(actual_type_map[topic])
        return deserialize_message(payload, msg_cache[topic])

    joint_rows = []       # (t, pos, vel, eff)
    gripper_rows = []     # (t, status)
    msg_count = {joint_topic: 0, gripper_topic: 0}

    while reader.has_next():
        topic, payload, t_ns = reader.read_next()
        t = float(t_ns) * 1e-9
        
        # 调试：打印前几个话题以检查匹配
        if msg_count[joint_topic] == 0 and msg_count[gripper_topic] < 3:
            print(f"[Debug] 读取话题 '{topic}', joint_topic='{joint_topic}', match={topic==joint_topic}", file=sys.stderr)

        if topic == joint_topic:
            try:
                msg = decode(topic, payload)
                n = min(6, len(msg.position))
                pos = np.array(list(msg.position[:n]) + [0.0] * (6 - n), dtype=np.float64)
                vel = np.array(list(msg.velocity[:n]) + [0.0] * (6 - n), dtype=np.float64) if len(msg.velocity) else np.zeros(6)
                eff = np.array(list(msg.effort[:n]) + [0.0] * (6 - n), dtype=np.float64) if len(msg.effort) else np.zeros(6)
                joint_rows.append((t, pos, vel, eff))
                msg_count[joint_topic] += 1
            except Exception as e:
                print(f"[Warn] 解码 {joint_topic} 失败: {e}", file=sys.stderr)
        elif topic == gripper_topic:
            try:
                msg = decode(topic, payload)
                s = str(msg.data).strip().lower()
                if s in ('idle', 'grasping', 'grasped'):
                    gripper_rows.append((t, s))
                    msg_count[gripper_topic] += 1
            except Exception as e:
                print(f"[Warn] 解码 {gripper_topic} 失败: {e}", file=sys.stderr)

    print(f"[Bag Info] 读取到 {msg_count[joint_topic]} 条关节消息，{msg_count[gripper_topic]} 条 gripper 消息", file=sys.stderr)

    if not joint_rows:
        raise RuntimeError(f'bag 中无关节数据，无法评分。(尝试了话题: {joint_topic})')
    if not gripper_rows:
        gripper_rows = [(joint_rows[0][0], 'idle')]

    joint_rows.sort(key=lambda x: x[0])
    gripper_rows.sort(key=lambda x: x[0])

    frames: List[Frame] = []
    gi = 0
    current_status = gripper_rows[0][1]
    seen_grasped = False

    for t, pos, vel, eff in joint_rows:
        while gi + 1 < len(gripper_rows) and gripper_rows[gi + 1][0] <= t:
            gi += 1
            current_status = gripper_rows[gi][1]

        if current_status == 'grasped':
            phase = PhaseState.TRANSPORT.value
            seen_grasped = True
        elif current_status == 'grasping':
            phase = PhaseState.GRASPING.value
        else:
            phase = PhaseState.RELEASE.value if seen_grasped else PhaseState.APPROACH.value

        frames.append(Frame(
            timestamp=t,
            positions=pos,
            velocities=vel,
            efforts=eff,
            phase=phase,
            gripper_status=current_status,
            is_grasped_flag=1 if current_status == 'grasped' else 0,
        ))

    return frames, joint_topic, gripper_topic

def main():
    parser = argparse.ArgumentParser(description='机械臂轨迹离线评估器（读取 rosbag2/mcap）')
    parser.add_argument('--bag', type=str, required=True,
                        help='rosbag 目录，或其中 .mcap 文件路径')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='评估结果输出目录 (默认: ./eval_results)')
    parser.add_argument('--joint_topics', type=str, default='/joint_states,/joint_states_sim',
                        help='关节话题候选，逗号分隔（优先使用排在前面的）')
    parser.add_argument('--gripper_topic', type=str, default='/gripper_status',
                        help='gripper 状态话题')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bag_uri = _resolve_bag_uri(args.bag)
    joint_candidates = [x.strip() for x in args.joint_topics.split(',') if x.strip()]

    frames, joint_topic, gripper_topic = _load_frames_from_bag(
        bag_uri=bag_uri,
        joint_topic_candidates=joint_candidates,
        gripper_topic=args.gripper_topic,
    )

    print('=' * 60)
    print('[Evaluator] 离线评估模式')
    print(f'[Evaluator] Bag目录: {bag_uri}')
    print(f'[Evaluator] Joint话题: {joint_topic}')
    print(f'[Evaluator] Gripper话题: {gripper_topic}')
    print(f'[Evaluator] 读取帧数: {len(frames)}')
    print('=' * 60)

    evaluator = TrajectoryEvaluator()
    results = evaluator.evaluate(frames)

    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    evaluator.report(results, session_name=session)

    metrics_path = os.path.join(args.output_dir, f'metrics_{session}.json')
    raw_path = os.path.join(args.output_dir, f'raw_{session}.npz')
    evaluator.save_metrics(results, metrics_path)
    evaluator.save_raw_data(frames, raw_path)


if __name__ == '__main__':
    main()
