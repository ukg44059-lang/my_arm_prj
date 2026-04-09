"""
Collision Checker - MuJoCo-based collision detection for path planning
碰撞检测器 - 基于MuJoCo的路径规划碰撞检测
"""

import numpy as np
import mujoco
from typing import List, Optional, Callable

try:
    from .planner_utils import interpolate_path
except ImportError:
    from planner_utils import interpolate_path


class MuJoCoCollisionChecker:
    """
    Collision checker using MuJoCo physics engine
    使用MuJoCo物理引擎的碰撞检测器
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 joint_names: Optional[List[str]] = None,
                 check_self_collision: bool = True,
                 exclude_bodies: Optional[List[str]] = None):
        """
        Initialize collision checker

        Args:
            model: MuJoCo model
            data: MuJoCo data (will be copied for collision checking)
            joint_names: Names of joints to control (default: ["joint1", ..., "joint6"])
            check_self_collision: Whether to check self-collision
            exclude_bodies: List of body names to exclude from collision detection
                          (e.g., ['floor', 'table'] to only check robot-obstacle collision)
        """
        self.model = model
        # Create a separate data instance for collision checking
        self.collision_data = mujoco.MjData(model)

        # Joint configuration
        if joint_names is None:
            joint_names = [f"joint{i+1}" for i in range(6)]
        self.joint_names = joint_names
        self.n_joints = len(joint_names)

        # Get joint IDs
        self.joint_ids = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name
            )
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in model")
            self.joint_ids.append(joint_id)

        # Collision settings
        self.check_self_collision = check_self_collision
        self.exclude_bodies = exclude_bodies if exclude_bodies is not None else []

        # Get excluded body IDs
        self.exclude_body_ids = []
        for body_name in self.exclude_bodies:
            body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name
            )
            if body_id >= 0:
                self.exclude_body_ids.append(body_id)

        print(f"✓ MuJoCo Collision Checker initialized")
        print(f"  - Joints: {self.n_joints}")
        print(f"  - Self-collision check: {check_self_collision}")
        print(f"  - Excluded bodies: {len(self.exclude_bodies)}")

    def set_joint_configuration(self, config: np.ndarray):
        """
        Set joint configuration in collision data

        Args:
            config: Joint configuration
        """
        if len(config) != self.n_joints:
            raise ValueError(f"Expected {self.n_joints} joint values, got {len(config)}")

        # Set joint positions
        for i, joint_id in enumerate(self.joint_ids):
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.collision_data.qpos[qpos_addr] = config[i]

        # Update forward kinematics
        mujoco.mj_forward(self.model, self.collision_data)

    def check_collision_at_config(self, config: np.ndarray) -> bool:
        """
        Check if robot is in collision at given configuration

        Args:
            config: Joint configuration to check

        Returns:
            True if collision detected, False otherwise
        """
        # Set configuration
        self.set_joint_configuration(config)

        # Check all contacts
        for i in range(self.collision_data.ncon):
            contact = self.collision_data.contact[i]

            # Get geom IDs
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            # Get body IDs
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]

            # Skip excluded bodies
            if body1_id in self.exclude_body_ids or body2_id in self.exclude_body_ids:
                continue

            # Check penetration distance (negative means penetration)
            if contact.dist < 0:
                return True

        return False

    def check_path_collision(self, path: List[np.ndarray],
                            step_size: float = 0.05) -> bool:
        """
        Check if entire path is collision-free

        Args:
            path: List of configurations
            step_size: Step size for interpolation (smaller = more accurate)

        Returns:
            True if path is collision-free, False otherwise
        """
        if len(path) < 2:
            return not self.check_collision_at_config(path[0])

        # Check each segment
        for i in range(len(path) - 1):
            if not self.check_segment_collision_free(path[i], path[i+1], step_size):
                return False

        return True

    def check_segment_collision_free(self, config1: np.ndarray,
                                    config2: np.ndarray,
                                    step_size: float = 0.05) -> bool:
        """
        Check if path segment between two configurations is collision-free

        Args:
            config1: Start configuration
            config2: End configuration
            step_size: Step size for interpolation

        Returns:
            True if segment is collision-free, False otherwise
        """
        # Interpolate between configurations
        interpolated = interpolate_path(config1, config2, step_size)

        # Check collision at each interpolated point
        for config in interpolated:
            if self.check_collision_at_config(config):
                return False

        return True

    def get_collision_info_at_config(self, config: np.ndarray) -> dict:
        """
        Get detailed collision information at configuration

        Args:
            config: Joint configuration

        Returns:
            Dictionary with collision info:
                - 'in_collision': bool
                - 'num_contacts': int
                - 'contacts': list of contact details
        """
        self.set_joint_configuration(config)

        contacts = []
        for i in range(self.collision_data.ncon):
            contact = self.collision_data.contact[i]

            geom1_id = contact.geom1
            geom2_id = contact.geom2
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]

            # Skip excluded bodies
            if body1_id in self.exclude_body_ids or body2_id in self.exclude_body_ids:
                continue

            if contact.dist < 0:  # Penetration
                geom1_name = self.model.geom(geom1_id).name or f"geom_{geom1_id}"
                geom2_name = self.model.geom(geom2_id).name or f"geom_{geom2_id}"
                body1_name = self.model.body(body1_id).name or f"body_{body1_id}"
                body2_name = self.model.body(body2_id).name or f"body_{body2_id}"

                contacts.append({
                    'geom1': geom1_name,
                    'geom2': geom2_name,
                    'body1': body1_name,
                    'body2': body2_name,
                    'penetration': -contact.dist,
                    'position': contact.pos.copy()
                })

        return {
            'in_collision': len(contacts) > 0,
            'num_contacts': len(contacts),
            'contacts': contacts
        }

    def get_clearance(self, config: np.ndarray) -> float:
        """
        Get minimum clearance (distance to nearest obstacle)

        Args:
            config: Joint configuration

        Returns:
            Minimum distance to obstacles (negative if in collision)
        """
        self.set_joint_configuration(config)

        min_dist = float('inf')

        for i in range(self.collision_data.ncon):
            contact = self.collision_data.contact[i]

            geom1_id = contact.geom1
            geom2_id = contact.geom2
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]

            # Skip excluded bodies
            if body1_id in self.exclude_body_ids or body2_id in self.exclude_body_ids:
                continue

            if contact.dist < min_dist:
                min_dist = contact.dist

        return min_dist if min_dist != float('inf') else 1.0


class CollisionCheckerWrapper:
    """
    Wrapper for custom collision checking functions
    自定义碰撞检测函数的包装器
    """

    def __init__(self, collision_fn: Callable[[np.ndarray], bool]):
        """
        Initialize collision checker wrapper

        Args:
            collision_fn: Function that takes configuration and returns True if collision
        """
        self.collision_fn = collision_fn

    def check_collision_at_config(self, config: np.ndarray) -> bool:
        """Check collision at configuration"""
        return self.collision_fn(config)

    def check_segment_collision_free(self, config1: np.ndarray,
                                    config2: np.ndarray,
                                    step_size: float = 0.05) -> bool:
        """Check if segment is collision-free"""
        interpolated = interpolate_path(config1, config2, step_size)
        for config in interpolated:
            if self.collision_fn(config):
                return False
        return True

    def check_path_collision(self, path: List[np.ndarray],
                            step_size: float = 0.05) -> bool:
        """Check if path is collision-free"""
        for i in range(len(path) - 1):
            if not self.check_segment_collision_free(path[i], path[i+1], step_size):
                return False
        return True


class SafetyMarginCollisionChecker:
    """
    Collision checker with safety margin
    带安全边距的碰撞检测器
    """

    def __init__(self, base_checker: MuJoCoCollisionChecker,
                 safety_margin: float = 0.01):
        """
        Initialize safety margin collision checker

        Args:
            base_checker: Base collision checker
            safety_margin: Minimum clearance required (meters)
        """
        self.base_checker = base_checker
        self.safety_margin = safety_margin

    def check_collision_at_config(self, config: np.ndarray) -> bool:
        """Check collision with safety margin"""
        clearance = self.base_checker.get_clearance(config)
        return clearance < self.safety_margin

    def check_segment_collision_free(self, config1: np.ndarray,
                                    config2: np.ndarray,
                                    step_size: float = 0.05) -> bool:
        """Check segment with safety margin"""
        interpolated = interpolate_path(config1, config2, step_size)
        for config in interpolated:
            if self.check_collision_at_config(config):
                return False
        return True

    def check_path_collision(self, path: List[np.ndarray],
                            step_size: float = 0.05) -> bool:
        """Check path with safety margin"""
        for i in range(len(path) - 1):
            if not self.check_segment_collision_free(path[i], path[i+1], step_size):
                return False
        return True
