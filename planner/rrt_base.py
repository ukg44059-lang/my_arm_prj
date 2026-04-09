"""
RRT Base - Core RRT algorithm implementation
RRT基础 - 核心RRT算法实现
"""

import numpy as np
import time
from typing import Optional, List, Tuple

try:
    from .planner_utils import (
        ConfigurationSpace, RRTNode, RRTTree,
        distance_metric, steer, sample_goal_bias
    )
    from .collision_checker import MuJoCoCollisionChecker
except ImportError:
    from planner_utils import (
        ConfigurationSpace, RRTNode, RRTTree,
        distance_metric, steer, sample_goal_bias
    )
    from collision_checker import MuJoCoCollisionChecker


class RRTPlanner:
    """
    Basic RRT (Rapidly-exploring Random Tree) planner
    基础RRT（快速探索随机树）规划器
    """

    def __init__(self,
                 config_space: ConfigurationSpace,
                 collision_checker: MuJoCoCollisionChecker,
                 max_step_size: float = 0.2,
                 goal_bias: float = 0.05,
                 goal_tolerance: float = 0.1):
        """
        Initialize RRT planner

        Args:
            config_space: Configuration space definition
            collision_checker: Collision checker instance
            max_step_size: Maximum step size for tree extension
            goal_bias: Probability of sampling goal configuration
            goal_tolerance: Distance threshold to consider goal reached
        """
        self.config_space = config_space
        self.collision_checker = collision_checker
        self.max_step_size = max_step_size
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance

        # Planning state
        self.tree: Optional[RRTTree] = None
        self.goal_node: Optional[RRTNode] = None

        # Statistics
        self.planning_time = 0.0
        self.num_iterations = 0
        self.num_nodes = 0

    def plan(self,
             start_config: np.ndarray,
             goal_config: np.ndarray,
             max_iterations: int = 5000,
             max_time: float = 30.0) -> Tuple[Optional[List[np.ndarray]], dict]:
        """
        Plan path from start to goal

        Args:
            start_config: Start configuration
            goal_config: Goal configuration
            max_iterations: Maximum number of iterations
            max_time: Maximum planning time (seconds)

        Returns:
            path: List of configurations from start to goal (None if failed)
            info: Dictionary with planning statistics
        """
        # Reset statistics
        self.num_iterations = 0
        self.goal_node = None
        start_time = time.time()

        # Validate start and goal
        if not self.config_space.is_config_valid(start_config):
            return None, {'success': False, 'error': 'Invalid start configuration'}

        if not self.config_space.is_config_valid(goal_config):
            return None, {'success': False, 'error': 'Invalid goal configuration'}

        if self.collision_checker.check_collision_at_config(start_config):
            return None, {'success': False, 'error': 'Start configuration in collision'}

        if self.collision_checker.check_collision_at_config(goal_config):
            return None, {'success': False, 'error': 'Goal configuration in collision'}

        # Initialize tree with start configuration
        root = RRTNode(start_config)
        self.tree = RRTTree(root)

        # Main planning loop
        for iteration in range(max_iterations):
            self.num_iterations = iteration + 1

            # Check timeout
            if time.time() - start_time > max_time:
                break

            # Sample random configuration (with goal bias)
            sample = sample_goal_bias(goal_config, self.config_space, self.goal_bias)

            # Find nearest node in tree
            nearest_node = self.tree.find_nearest_node(sample)

            # Steer towards sample
            new_config = steer(nearest_node.config, sample, self.max_step_size)

            # Check if new configuration is valid and collision-free
            if not self.config_space.is_config_valid(new_config):
                continue

            # Check collision for the path segment
            if not self.collision_checker.check_segment_collision_free(
                nearest_node.config, new_config, step_size=0.05
            ):
                continue

            # Add new node to tree
            new_node = RRTNode(new_config, parent=nearest_node)
            self.tree.add_node(new_node)

            # Check if goal is reached
            if distance_metric(new_config, goal_config) <= self.goal_tolerance:
                # Try to connect directly to goal
                if self.collision_checker.check_segment_collision_free(
                    new_config, goal_config, step_size=0.05
                ):
                    self.goal_node = RRTNode(goal_config, parent=new_node)
                    break

        # Record planning time
        self.planning_time = time.time() - start_time
        self.num_nodes = len(self.tree.nodes)

        # Extract path
        if self.goal_node is not None:
            path = self.tree.get_path_to_node(self.goal_node)
            success = True
        else:
            path = None
            success = False

        # Compile statistics
        info = {
            'success': success,
            'planning_time': self.planning_time,
            'num_iterations': self.num_iterations,
            'num_nodes': self.num_nodes,
            'path_length': self._compute_path_length(path) if path else 0.0
        }

        return path, info

    def _compute_path_length(self, path: Optional[List[np.ndarray]]) -> float:
        """Compute total path length"""
        if path is None or len(path) < 2:
            return 0.0

        length = 0.0
        for i in range(len(path) - 1):
            length += distance_metric(path[i], path[i + 1])
        return length

    def get_tree_nodes(self) -> List[np.ndarray]:
        """
        Get all nodes in the tree (for visualization)

        Returns:
            List of all node configurations
        """
        if self.tree is None:
            return []
        return [node.config for node in self.tree.nodes]

    def get_tree_edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all edges in the tree (for visualization)

        Returns:
            List of (parent_config, child_config) tuples
        """
        if self.tree is None:
            return []

        edges = []
        for node in self.tree.nodes:
            if node.parent is not None:
                edges.append((node.parent.config, node.config))
        return edges


class BidirectionalRRT(RRTPlanner):
    """
    Bidirectional RRT - Grows trees from both start and goal
    双向RRT - 从起点和终点同时生长树
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree_goal: Optional[RRTTree] = None

    def plan(self,
             start_config: np.ndarray,
             goal_config: np.ndarray,
             max_iterations: int = 5000,
             max_time: float = 30.0) -> Tuple[Optional[List[np.ndarray]], dict]:
        """
        Plan path using bidirectional RRT

        Args:
            start_config: Start configuration
            goal_config: Goal configuration
            max_iterations: Maximum number of iterations
            max_time: Maximum planning time (seconds)

        Returns:
            path: List of configurations from start to goal (None if failed)
            info: Dictionary with planning statistics
        """
        # Reset statistics
        self.num_iterations = 0
        start_time = time.time()

        # Validate configurations
        if not self.config_space.is_config_valid(start_config):
            return None, {'success': False, 'error': 'Invalid start configuration'}

        if not self.config_space.is_config_valid(goal_config):
            return None, {'success': False, 'error': 'Invalid goal configuration'}

        if self.collision_checker.check_collision_at_config(start_config):
            return None, {'success': False, 'error': 'Start configuration in collision'}

        if self.collision_checker.check_collision_at_config(goal_config):
            return None, {'success': False, 'error': 'Goal configuration in collision'}

        # Initialize two trees
        root_start = RRTNode(start_config)
        root_goal = RRTNode(goal_config)
        self.tree = RRTTree(root_start)
        self.tree_goal = RRTTree(root_goal)

        # Keep track of which tree to extend
        tree_a = self.tree
        tree_b = self.tree_goal

        connection_node_a: Optional[RRTNode] = None
        connection_node_b: Optional[RRTNode] = None

        # Main planning loop
        for iteration in range(max_iterations):
            self.num_iterations = iteration + 1

            # Check timeout
            if time.time() - start_time > max_time:
                break

            # Sample random configuration
            sample = self.config_space.sample_random_config()

            # Extend tree A towards sample
            nearest_a = tree_a.find_nearest_node(sample)
            new_config = steer(nearest_a.config, sample, self.max_step_size)

            # Check validity
            if not self.config_space.is_config_valid(new_config):
                continue

            if not self.collision_checker.check_segment_collision_free(
                nearest_a.config, new_config, step_size=0.05
            ):
                continue

            # Add to tree A
            new_node_a = RRTNode(new_config, parent=nearest_a)
            tree_a.add_node(new_node_a)

            # Try to connect to tree B
            nearest_b = tree_b.find_nearest_node(new_config)
            distance_to_b = distance_metric(new_config, nearest_b.config)

            if distance_to_b <= self.max_step_size:
                # Try direct connection
                if self.collision_checker.check_segment_collision_free(
                    new_config, nearest_b.config, step_size=0.05
                ):
                    connection_node_a = new_node_a
                    connection_node_b = nearest_b
                    break

            # Swap trees for next iteration
            tree_a, tree_b = tree_b, tree_a

        # Record planning time
        self.planning_time = time.time() - start_time
        self.num_nodes = len(self.tree.nodes) + len(self.tree_goal.nodes)

        # Extract path
        if connection_node_a is not None and connection_node_b is not None:
            # Determine which path to reverse
            if connection_node_a in self.tree.nodes:
                path_start = self.tree.get_path_to_node(connection_node_a)
                path_goal = self.tree_goal.get_path_to_node(connection_node_b)
                path_goal.reverse()
            else:
                path_start = self.tree_goal.get_path_to_node(connection_node_b)
                path_goal = self.tree.get_path_to_node(connection_node_a)
                path_goal.reverse()
                path_start, path_goal = path_goal, path_start

            path = path_start + path_goal[1:]  # Avoid duplicate connection point
            success = True
        else:
            path = None
            success = False

        # Compile statistics
        info = {
            'success': success,
            'planning_time': self.planning_time,
            'num_iterations': self.num_iterations,
            'num_nodes': self.num_nodes,
            'path_length': self._compute_path_length(path) if path else 0.0
        }

        return path, info
