"""
RRT Path Planning Library for Robot Arm Control
机械臂RRT路径规划库

This library provides RRT-family algorithms for collision-free path planning:
- Basic RRT
- RRT* (asymptotically optimal)
- RRT-Connect (bidirectional)
- Informed RRT* (ellipsoidal sampling)
- Bidirectional RRT

Features:
- MuJoCo-based collision detection
- Path smoothing and optimization
- Trajectory parameterization
- Efficient nearest neighbor search with KD-trees
"""

# Configuration Space and Utilities
from .planner_utils import (
    ConfigurationSpace,
    RRTNode,
    RRTTree,
    distance_metric,
    interpolate_path,
    steer,
    sample_goal_bias,
    informed_sampling,
    compute_rrt_star_radius,
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

# RRT Planners
from .rrt_base import (
    RRTPlanner,
    BidirectionalRRT
)

from .rrt_variants import (
    RRTStarPlanner,
    RRTConnectPlanner,
    InformedRRTStarPlanner
)

# Path Smoothing
from .path_smoother import (
    PathSmoother,
    TrajectoryParameterizer,
    compute_path_statistics
)

__all__ = [
    # Configuration Space
    'ConfigurationSpace',
    'RRTNode',
    'RRTTree',

    # Utilities
    'distance_metric',
    'interpolate_path',
    'steer',
    'sample_goal_bias',
    'informed_sampling',
    'compute_rrt_star_radius',
    'simplify_path',
    'compute_curvature',
    'resample_path',
    'compute_path_length',

    # Collision Checking
    'MuJoCoCollisionChecker',
    'CollisionCheckerWrapper',
    'SafetyMarginCollisionChecker',

    # Planners
    'RRTPlanner',
    'BidirectionalRRT',
    'RRTStarPlanner',
    'RRTConnectPlanner',
    'InformedRRTStarPlanner',

    # Path Processing
    'PathSmoother',
    'TrajectoryParameterizer',
    'compute_path_statistics',
]

__version__ = '1.0.0'
__author__ = 'Claude Code'
