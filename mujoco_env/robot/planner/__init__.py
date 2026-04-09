"""
Planning Utilities Library for Robot Arm Control
机械臂规划工具库

Features:
- MuJoCo-based collision detection
- Path smoothing and optimization
- Trajectory parameterization
- Joint interpolation utilities
"""

# Configuration Space and Utilities
from .planner_utils import (
    ConfigurationSpace,
    distance_metric,
    interpolate_path,
    simplify_path,
    compute_curvature,
    resample_path,
    compute_path_length
)

# Collision Checking
from .collision_checker import (
    MuJoCoCollisionChecker,
    CollisionCheckerWrapper,
    SafetyMarginCollisionChecker
)

# Path Smoothing
from .path_smoother import (
    PathSmoother,
    TrajectoryParameterizer,
    compute_path_statistics
)

# Joint Interpolation
from .joint_interpolator import (
    CubicJointInterpolator,
    QuinticJointInterpolator,
    cubic_interpolate,
    quintic_interpolate
)

__all__ = [
    # Configuration Space
    'ConfigurationSpace',

    # Utilities
    'distance_metric',
    'interpolate_path',
    'simplify_path',
    'compute_curvature',
    'resample_path',
    'compute_path_length',

    # Collision Checking
    'MuJoCoCollisionChecker',
    'CollisionCheckerWrapper',
    'SafetyMarginCollisionChecker',

    # Path Processing
    'PathSmoother',
    'TrajectoryParameterizer',
    'compute_path_statistics',

    # Joint Interpolation
    'CubicJointInterpolator',
    'QuinticJointInterpolator',
    'cubic_interpolate',
    'quintic_interpolate',
]

__version__ = '1.0.0'
__author__ = 'Claude Code'
