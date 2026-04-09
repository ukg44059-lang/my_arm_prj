"""
Planner Utilities - Helper functions for RRT path planning
路径规划工具 - RRT路径规划的辅助函数
"""

import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Optional


class ConfigurationSpace:
    """
    Configuration space representation for robot joint space
    机器人关节空间的配置空间表示
    """

    def __init__(self, joint_limits: np.ndarray):
        """
        Initialize configuration space

        Args:
            joint_limits: (n_joints, 2) array of [min, max] for each joint
        """
        self.joint_limits = np.array(joint_limits)
        self.n_joints = len(joint_limits)
        self.lower_bounds = joint_limits[:, 0]
        self.upper_bounds = joint_limits[:, 1]

    def sample_random_config(self) -> np.ndarray:
        """
        Sample a random configuration in joint space

        Returns:
            Random joint configuration
        """
        return np.random.uniform(self.lower_bounds, self.upper_bounds)

    def clip_config(self, config: np.ndarray) -> np.ndarray:
        """
        Clip configuration to joint limits

        Args:
            config: Joint configuration

        Returns:
            Clipped configuration
        """
        return np.clip(config, self.lower_bounds, self.upper_bounds)

    def is_config_valid(self, config: np.ndarray) -> bool:
        """
        Check if configuration is within joint limits

        Args:
            config: Joint configuration

        Returns:
            True if valid, False otherwise
        """
        return np.all(config >= self.lower_bounds) and np.all(config <= self.upper_bounds)


class RRTNode:
    """
    Node in RRT tree
    RRT树中的节点
    """

    def __init__(self, config: np.ndarray, parent: Optional['RRTNode'] = None):
        """
        Initialize RRT node

        Args:
            config: Joint configuration
            parent: Parent node in the tree
        """
        self.config = np.array(config)
        self.parent = parent
        self.cost = 0.0  # Cost from start (for RRT*)
        self.children: List['RRTNode'] = []

        if parent is not None:
            self.cost = parent.cost + distance_metric(parent.config, config)
            parent.children.append(self)


class RRTTree:
    """
    RRT tree structure with efficient nearest neighbor search
    RRT树结构，带高效最近邻搜索
    """

    def __init__(self, root: RRTNode):
        """
        Initialize RRT tree

        Args:
            root: Root node of the tree
        """
        self.root = root
        self.nodes: List[RRTNode] = [root]
        self.kdtree: Optional[KDTree] = None
        self._rebuild_kdtree()

    def add_node(self, node: RRTNode):
        """
        Add a node to the tree

        Args:
            node: Node to add
        """
        self.nodes.append(node)
        self._rebuild_kdtree()

    def _rebuild_kdtree(self):
        """Rebuild KD-tree for efficient nearest neighbor search"""
        if len(self.nodes) > 0:
            configs = np.array([node.config for node in self.nodes])
            self.kdtree = KDTree(configs)

    def find_nearest_node(self, config: np.ndarray) -> RRTNode:
        """
        Find nearest node to given configuration

        Args:
            config: Target configuration

        Returns:
            Nearest node in the tree
        """
        if self.kdtree is None or len(self.nodes) == 0:
            return self.root

        _, idx = self.kdtree.query(config)
        return self.nodes[idx]

    def find_near_nodes(self, config: np.ndarray, radius: float) -> List[RRTNode]:
        """
        Find all nodes within radius of configuration (for RRT*)

        Args:
            config: Target configuration
            radius: Search radius

        Returns:
            List of nodes within radius
        """
        if self.kdtree is None or len(self.nodes) == 0:
            return []

        indices = self.kdtree.query_ball_point(config, radius)
        return [self.nodes[i] for i in indices]

    def get_path_to_node(self, node: RRTNode) -> List[np.ndarray]:
        """
        Extract path from root to given node

        Args:
            node: Target node

        Returns:
            List of configurations from root to node
        """
        path = []
        current = node
        while current is not None:
            path.append(current.config)
            current = current.parent
        return list(reversed(path))


def distance_metric(config1: np.ndarray, config2: np.ndarray,
                   weights: Optional[np.ndarray] = None) -> float:
    """
    Compute weighted distance between two configurations

    Args:
        config1: First configuration
        config2: Second configuration
        weights: Optional joint weights (default: all 1.0)

    Returns:
        Weighted Euclidean distance
    """
    if weights is None:
        weights = np.ones_like(config1)

    diff = config1 - config2
    return np.sqrt(np.sum((weights * diff) ** 2))


def interpolate_path(config_start: np.ndarray, config_end: np.ndarray,
                    step_size: float) -> List[np.ndarray]:
    """
    Interpolate path between two configurations

    Args:
        config_start: Start configuration
        config_end: End configuration
        step_size: Maximum step size between configurations

    Returns:
        List of interpolated configurations
    """
    distance = np.linalg.norm(config_end - config_start)

    if distance <= step_size:
        return [config_start, config_end]

    n_steps = int(np.ceil(distance / step_size))
    alphas = np.linspace(0, 1, n_steps + 1)

    path = []
    for alpha in alphas:
        config = config_start + alpha * (config_end - config_start)
        path.append(config)

    return path


def steer(config_from: np.ndarray, config_to: np.ndarray,
         max_step: float) -> np.ndarray:
    """
    Steer from one configuration towards another with maximum step size

    Args:
        config_from: Starting configuration
        config_to: Target configuration
        max_step: Maximum step size

    Returns:
        New configuration in direction of target
    """
    direction = config_to - config_from
    distance = np.linalg.norm(direction)

    if distance <= max_step:
        return config_to.copy()

    # Normalize and scale by max_step
    direction = direction / distance
    return config_from + max_step * direction


def compute_path_length(path: List[np.ndarray]) -> float:
    """
    Compute total length of path

    Args:
        path: List of configurations

    Returns:
        Total path length
    """
    if len(path) < 2:
        return 0.0

    length = 0.0
    for i in range(len(path) - 1):
        length += distance_metric(path[i], path[i + 1])

    return length


def sample_goal_bias(goal_config: np.ndarray, config_space: ConfigurationSpace,
                    goal_bias: float = 0.05) -> np.ndarray:
    """
    Sample configuration with bias towards goal

    Args:
        goal_config: Goal configuration
        config_space: Configuration space
        goal_bias: Probability of sampling goal (0-1)

    Returns:
        Sampled configuration
    """
    if np.random.random() < goal_bias:
        return goal_config.copy()
    else:
        return config_space.sample_random_config()


def informed_sampling(start_config: np.ndarray, goal_config: np.ndarray,
                     c_best: float, config_space: ConfigurationSpace) -> np.ndarray:
    """
    Informed sampling for Informed RRT* (samples in ellipsoid)

    Args:
        start_config: Start configuration
        goal_config: Goal configuration
        c_best: Best path cost found so far
        config_space: Configuration space

    Returns:
        Sampled configuration in ellipsoid
    """
    # Simplified version - sample uniformly and reject if outside ellipsoid
    # For full implementation, need rotation matrix computation

    c_min = distance_metric(start_config, goal_config)

    # If no improvement possible, sample randomly
    if c_best >= float('inf'):
        return config_space.sample_random_config()

    # Sample in ellipsoid (simplified rejection sampling)
    max_attempts = 100
    for _ in range(max_attempts):
        sample = config_space.sample_random_config()

        # Check if sample is in ellipsoid
        d1 = distance_metric(sample, start_config)
        d2 = distance_metric(sample, goal_config)

        if d1 + d2 <= c_best:
            return sample

    # Fallback to random sampling
    return config_space.sample_random_config()


def compute_rrt_star_radius(n_nodes: int, n_dimensions: int,
                           gamma: float = 1.0) -> float:
    """
    Compute search radius for RRT* rewiring

    Args:
        n_nodes: Current number of nodes in tree
        n_dimensions: Dimensionality of configuration space
        gamma: Scaling parameter (typically 1.0)

    Returns:
        Search radius for near nodes
    """
    # Formula: gamma * (log(n) / n)^(1/d)
    if n_nodes <= 1:
        return float('inf')

    return gamma * np.power(np.log(n_nodes) / n_nodes, 1.0 / n_dimensions)


def simplify_path(path: List[np.ndarray], collision_fn) -> List[np.ndarray]:
    """
    Simplify path by removing unnecessary waypoints (shortcutting)

    Args:
        path: Original path
        collision_fn: Function to check if path segment is collision-free
                     Should take (config1, config2) and return bool

    Returns:
        Simplified path
    """
    if len(path) <= 2:
        return path

    simplified = [path[0]]
    i = 0

    while i < len(path) - 1:
        # Try to connect to farthest reachable waypoint
        j = len(path) - 1
        while j > i + 1:
            if collision_fn(path[i], path[j]):
                # Direct connection is collision-free
                simplified.append(path[j])
                i = j
                break
            j -= 1
        else:
            # Can't skip any waypoints, move to next
            i += 1
            simplified.append(path[i])

    return simplified


def compute_curvature(path: List[np.ndarray]) -> List[float]:
    """
    Compute approximate curvature at each point in path

    Args:
        path: List of configurations

    Returns:
        List of curvature values (0 at endpoints)
    """
    if len(path) < 3:
        return [0.0] * len(path)

    curvatures = [0.0]  # First point

    for i in range(1, len(path) - 1):
        # Compute angle between consecutive segments
        v1 = path[i] - path[i-1]
        v2 = path[i+1] - path[i]

        # Normalize
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm > 0 and v2_norm > 0:
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm

            # Angle between vectors
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            # Curvature approximation
            avg_length = (v1_norm + v2_norm) / 2.0
            curvature = angle / avg_length if avg_length > 0 else 0.0
        else:
            curvature = 0.0

        curvatures.append(curvature)

    curvatures.append(0.0)  # Last point

    return curvatures


def resample_path(path: List[np.ndarray], resolution: float) -> List[np.ndarray]:
    """
    Resample path to have uniform spacing

    Args:
        path: Original path
        resolution: Desired spacing between points

    Returns:
        Resampled path
    """
    if len(path) < 2:
        return path

    resampled = [path[0]]

    accumulated_dist = 0.0
    target_dist = resolution

    for i in range(len(path) - 1):
        segment_start = path[i]
        segment_end = path[i + 1]
        segment_length = distance_metric(segment_start, segment_end)

        while accumulated_dist + segment_length >= target_dist:
            # Interpolate point at target distance
            alpha = (target_dist - accumulated_dist) / segment_length
            new_point = segment_start + alpha * (segment_end - segment_start)
            resampled.append(new_point)

            # Update for next target
            accumulated_dist = 0.0
            segment_start = new_point
            segment_length = distance_metric(segment_start, segment_end)
            target_dist = resolution

        accumulated_dist += segment_length

    # Add final point
    resampled.append(path[-1])

    return resampled


def compute_path_statistics(path: List[np.ndarray]) -> dict:
    """
    Compute statistics for a path

    Args:
        path: List of configurations

    Returns:
        Dictionary with path statistics
    """
    if len(path) < 2:
        return {
            'length': 0.0,
            'num_waypoints': len(path),
            'avg_segment_length': 0.0,
            'max_segment_length': 0.0,
            'min_segment_length': 0.0
        }

    segment_lengths = []
    total_length = 0.0

    for i in range(len(path) - 1):
        segment_length = distance_metric(path[i], path[i + 1])
        segment_lengths.append(segment_length)
        total_length += segment_length

    return {
        'length': total_length,
        'num_waypoints': len(path),
        'avg_segment_length': np.mean(segment_lengths),
        'max_segment_length': np.max(segment_lengths),
        'min_segment_length': np.min(segment_lengths)
    }
