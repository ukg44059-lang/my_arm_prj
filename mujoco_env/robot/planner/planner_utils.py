"""
Planner Utilities - Generic path utilities for robot planning.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np


class ConfigurationSpace:
    """Simple joint configuration space helper."""

    def __init__(self, joint_limits: np.ndarray):
        limits = np.asarray(joint_limits, dtype=float)
        if limits.ndim != 2 or limits.shape[1] != 2:
            raise ValueError("joint_limits must be shaped (n_joints, 2)")
        self.joint_limits = limits
        self.n_joints = limits.shape[0]

    def is_valid(self, config: np.ndarray) -> bool:
        q = np.asarray(config, dtype=float)
        if q.shape[0] != self.n_joints:
            return False
        return bool(np.all(q >= self.joint_limits[:, 0]) and np.all(q <= self.joint_limits[:, 1]))

    def clip(self, config: np.ndarray) -> np.ndarray:
        q = np.asarray(config, dtype=float)
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])


def distance_metric(config1: np.ndarray, config2: np.ndarray) -> float:
    """Euclidean distance in joint space."""
    return float(np.linalg.norm(np.asarray(config1, dtype=float) - np.asarray(config2, dtype=float)))


def interpolate_path(config_start: np.ndarray, config_end: np.ndarray, step_size: float = 0.05) -> List[np.ndarray]:
    """Linear interpolation between two configurations with bounded step size."""
    q0 = np.asarray(config_start, dtype=float)
    q1 = np.asarray(config_end, dtype=float)

    if step_size <= 0:
        return [q0.copy(), q1.copy()]

    dist = distance_metric(q0, q1)
    n_steps = max(1, int(np.ceil(dist / step_size)))
    t = np.linspace(0.0, 1.0, n_steps + 1)
    return [(1.0 - a) * q0 + a * q1 for a in t]


def compute_path_length(path: List[np.ndarray]) -> float:
    """Compute cumulative path length."""
    if len(path) < 2:
        return 0.0
    return float(sum(distance_metric(path[i], path[i + 1]) for i in range(len(path) - 1)))


def simplify_path(path: List[np.ndarray], collision_fn: Callable[[np.ndarray, np.ndarray], bool]) -> List[np.ndarray]:
    """Greedy shortcut simplification with segment-validity callback.

    collision_fn(a, b) should return True when segment a->b is collision-free.
    """
    if len(path) <= 2:
        return path

    simplified = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not collision_fn(path[i], path[j]):
            j -= 1
        simplified.append(path[j])
        i = j

    return simplified


def compute_curvature(path: List[np.ndarray]) -> np.ndarray:
    """Approximate discrete curvature magnitude at each waypoint."""
    n = len(path)
    if n < 3:
        return np.zeros(n, dtype=float)

    q = np.asarray(path, dtype=float)
    curvature = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        v1 = q[i] - q[i - 1]
        v2 = q[i + 1] - q[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-12 and n2 > 1e-12:
            cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            curvature[i] = np.arccos(cosang)
    return curvature


def resample_path(path: List[np.ndarray], resolution: float) -> List[np.ndarray]:
    """Resample path by approximate arc-length spacing."""
    if len(path) <= 1 or resolution <= 0:
        return path

    q = [np.asarray(p, dtype=float) for p in path]
    cumulative = [0.0]
    for i in range(1, len(q)):
        cumulative.append(cumulative[-1] + distance_metric(q[i - 1], q[i]))

    total = cumulative[-1]
    if total <= 1e-12:
        return [q[0].copy(), q[-1].copy()] if len(q) > 1 else [q[0].copy()]

    targets = np.arange(0.0, total, resolution)
    if len(targets) == 0 or abs(targets[-1] - total) > 1e-12:
        targets = np.append(targets, total)

    out: List[np.ndarray] = []
    seg = 0
    for s in targets:
        while seg < len(cumulative) - 2 and cumulative[seg + 1] < s:
            seg += 1
        s0, s1 = cumulative[seg], cumulative[seg + 1]
        if s1 - s0 <= 1e-12:
            out.append(q[seg].copy())
            continue
        a = (s - s0) / (s1 - s0)
        out.append((1.0 - a) * q[seg] + a * q[seg + 1])

    return out
