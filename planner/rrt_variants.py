"""
RRT Variants - Advanced RRT algorithms (RRT*, RRT-Connect, Informed RRT*)
RRT变体 - 高级RRT算法
"""

import numpy as np
import time
from typing import Optional, List, Tuple

try:
    from .planner_utils import (
        ConfigurationSpace, RRTNode, RRTTree,
        distance_metric, steer, sample_goal_bias,
        compute_rrt_star_radius, informed_sampling
    )
    from .collision_checker import MuJoCoCollisionChecker
    from .rrt_base import RRTPlanner
except ImportError:
    from planner_utils import (
        ConfigurationSpace, RRTNode, RRTTree,
        distance_metric, steer, sample_goal_bias,
        compute_rrt_star_radius, informed_sampling
    )
    from collision_checker import MuJoCoCollisionChecker
    from rrt_base import RRTPlanner


class RRTStarPlanner(RRTPlanner):
    """
    RRT* - Asymptotically optimal RRT variant
    RRT* - 渐进最优的RRT变体
    """

    def __init__(self, *args, rewire_factor: float = 1.5, **kwargs):
        """
        Initialize RRT* planner

        Args:
            rewire_factor: Factor for computing rewiring radius
            *args, **kwargs: Arguments for base RRTPlanner
        """
        super().__init__(*args, **kwargs)
        self.rewire_factor = rewire_factor

    def plan(self,
             start_config: np.ndarray,
             goal_config: np.ndarray,
             max_iterations: int = 5000,
             max_time: float = 30.0) -> Tuple[Optional[List[np.ndarray]], dict]:
        """
        Plan path using RRT*

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

        # Validate configurations
        if not self.config_space.is_config_valid(start_config):
            return None, {'success': False, 'error': 'Invalid start configuration'}

        if not self.config_space.is_config_valid(goal_config):
            return None, {'success': False, 'error': 'Invalid goal configuration'}

        if self.collision_checker.check_collision_at_config(start_config):
            return None, {'success': False, 'error': 'Start configuration in collision'}

        if self.collision_checker.check_collision_at_config(goal_config):
            return None, {'success': False, 'error': 'Goal configuration in collision'}

        # Initialize tree
        root = RRTNode(start_config)
        root.cost = 0.0
        self.tree = RRTTree(root)

        # Main planning loop
        for iteration in range(max_iterations):
            self.num_iterations = iteration + 1

            # Check timeout
            if time.time() - start_time > max_time:
                break

            # Sample random configuration
            sample = sample_goal_bias(goal_config, self.config_space, self.goal_bias)

            # Find nearest node
            nearest_node = self.tree.find_nearest_node(sample)

            # Steer towards sample
            new_config = steer(nearest_node.config, sample, self.max_step_size)

            # Check validity
            if not self.config_space.is_config_valid(new_config):
                continue

            # Find near nodes for rewiring
            search_radius = compute_rrt_star_radius(
                len(self.tree.nodes),
                self.config_space.n_joints,
                self.rewire_factor
            )
            near_nodes = self.tree.find_near_nodes(new_config, search_radius)

            if len(near_nodes) == 0:
                near_nodes = [nearest_node]

            # Choose best parent
            best_parent = None
            best_cost = float('inf')

            for near_node in near_nodes:
                # Check collision-free path
                if not self.collision_checker.check_segment_collision_free(
                    near_node.config, new_config, step_size=0.05
                ):
                    continue

                # Compute cost
                cost = near_node.cost + distance_metric(near_node.config, new_config)

                if cost < best_cost:
                    best_cost = cost
                    best_parent = near_node

            if best_parent is None:
                continue

            # Add new node
            new_node = RRTNode(new_config, parent=best_parent)
            new_node.cost = best_cost
            self.tree.add_node(new_node)

            # Rewire tree
            self._rewire(new_node, near_nodes)

            # Check if goal is reached
            if distance_metric(new_config, goal_config) <= self.goal_tolerance:
                if self.collision_checker.check_segment_collision_free(
                    new_config, goal_config, step_size=0.05
                ):
                    goal_cost = new_node.cost + distance_metric(new_config, goal_config)

                    # Update goal node if better path found
                    if self.goal_node is None or goal_cost < self.goal_node.cost:
                        self.goal_node = RRTNode(goal_config, parent=new_node)
                        self.goal_node.cost = goal_cost

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
            'path_length': self._compute_path_length(path) if path else 0.0,
            'path_cost': self.goal_node.cost if self.goal_node else 0.0
        }

        return path, info

    def _rewire(self, new_node: RRTNode, near_nodes: List[RRTNode]):
        """
        Rewire tree to reduce costs

        Args:
            new_node: Newly added node
            near_nodes: Nodes near the new node
        """
        for near_node in near_nodes:
            # Skip if near_node is parent of new_node
            if near_node == new_node.parent:
                continue

            # Compute potential new cost
            new_cost = new_node.cost + distance_metric(new_node.config, near_node.config)

            if new_cost < near_node.cost:
                # Check collision-free path
                if self.collision_checker.check_segment_collision_free(
                    new_node.config, near_node.config, step_size=0.05
                ):
                    # Rewire: change parent of near_node to new_node
                    if near_node.parent is not None:
                        near_node.parent.children.remove(near_node)

                    near_node.parent = new_node
                    new_node.children.append(near_node)

                    # Update cost recursively
                    self._update_costs(near_node)

    def _update_costs(self, node: RRTNode):
        """
        Recursively update costs after rewiring

        Args:
            node: Node to update
        """
        if node.parent is not None:
            node.cost = node.parent.cost + distance_metric(node.parent.config, node.config)

        for child in node.children:
            self._update_costs(child)


class RRTConnectPlanner:
    """
    RRT-Connect - Bidirectional RRT with aggressive connection attempts
    RRT-Connect - 带积极连接尝试的双向RRT
    """

    def __init__(self,
                 config_space: ConfigurationSpace,
                 collision_checker: MuJoCoCollisionChecker,
                 max_step_size: float = 0.2,
                 goal_tolerance: float = 0.1):
        """
        Initialize RRT-Connect planner

        Args:
            config_space: Configuration space definition
            collision_checker: Collision checker instance
            max_step_size: Maximum step size for tree extension
            goal_tolerance: Distance threshold for connection
        """
        self.config_space = config_space
        self.collision_checker = collision_checker
        self.max_step_size = max_step_size
        self.goal_tolerance = goal_tolerance

        # Planning state
        self.tree_start: Optional[RRTTree] = None
        self.tree_goal: Optional[RRTTree] = None

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
        Plan path using RRT-Connect

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

        # Initialize trees
        root_start = RRTNode(start_config)
        root_goal = RRTNode(goal_config)
        self.tree_start = RRTTree(root_start)
        self.tree_goal = RRTTree(root_goal)

        # Track trees
        tree_a = self.tree_start
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
            new_node_a = self._extend(tree_a, sample)

            if new_node_a is not None:
                # Try to connect tree B to new node
                connection_result = self._connect(tree_b, new_node_a.config)

                if connection_result is not None:
                    # Trees connected!
                    connection_node_a = new_node_a
                    connection_node_b = connection_result
                    break

            # Swap trees
            tree_a, tree_b = tree_b, tree_a

        # Record planning time
        self.planning_time = time.time() - start_time
        self.num_nodes = len(self.tree_start.nodes) + len(self.tree_goal.nodes)

        # Extract path
        if connection_node_a is not None and connection_node_b is not None:
            # Determine correct path direction
            if connection_node_a in self.tree_start.nodes:
                path_start = self.tree_start.get_path_to_node(connection_node_a)
                path_goal = self.tree_goal.get_path_to_node(connection_node_b)
                path_goal.reverse()
            else:
                path_start = self.tree_goal.get_path_to_node(connection_node_b)
                path_goal = self.tree_start.get_path_to_node(connection_node_a)
                path_goal.reverse()
                path_start, path_goal = path_goal, path_start

            path = path_start + path_goal[1:]
            success = True
        else:
            path = None
            success = False

        # Compile statistics
        path_length = 0.0
        if path:
            for i in range(len(path) - 1):
                path_length += distance_metric(path[i], path[i + 1])

        info = {
            'success': success,
            'planning_time': self.planning_time,
            'num_iterations': self.num_iterations,
            'num_nodes': self.num_nodes,
            'path_length': path_length
        }

        return path, info

    def _extend(self, tree: RRTTree, target: np.ndarray) -> Optional[RRTNode]:
        """
        Extend tree towards target configuration

        Args:
            tree: Tree to extend
            target: Target configuration

        Returns:
            New node if successful, None otherwise
        """
        nearest_node = tree.find_nearest_node(target)
        new_config = steer(nearest_node.config, target, self.max_step_size)

        # Check validity
        if not self.config_space.is_config_valid(new_config):
            return None

        if not self.collision_checker.check_segment_collision_free(
            nearest_node.config, new_config, step_size=0.05
        ):
            return None

        # Add to tree
        new_node = RRTNode(new_config, parent=nearest_node)
        tree.add_node(new_node)
        return new_node

    def _connect(self, tree: RRTTree, target: np.ndarray) -> Optional[RRTNode]:
        """
        Aggressively try to connect tree to target

        Args:
            tree: Tree to connect
            target: Target configuration

        Returns:
            Connection node if successful, None otherwise
        """
        current_node = tree.find_nearest_node(target)

        max_connect_steps = 50
        for _ in range(max_connect_steps):
            # Check if we've reached target
            distance = distance_metric(current_node.config, target)

            if distance <= self.goal_tolerance:
                return current_node

            # Steer towards target
            new_config = steer(current_node.config, target, self.max_step_size)

            # Check validity
            if not self.config_space.is_config_valid(new_config):
                return None

            if not self.collision_checker.check_segment_collision_free(
                current_node.config, new_config, step_size=0.05
            ):
                return None

            # Add to tree
            new_node = RRTNode(new_config, parent=current_node)
            tree.add_node(new_node)
            current_node = new_node

        return None


class InformedRRTStarPlanner(RRTStarPlanner):
    """
    Informed RRT* - RRT* with informed sampling in ellipsoidal subset
    Informed RRT* - 在椭球子集中进行知情采样的RRT*
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_best = float('inf')  # Best path cost found

    def plan(self,
             start_config: np.ndarray,
             goal_config: np.ndarray,
             max_iterations: int = 5000,
             max_time: float = 30.0) -> Tuple[Optional[List[np.ndarray]], dict]:
        """
        Plan path using Informed RRT*

        Args:
            start_config: Start configuration
            goal_config: Goal configuration
            max_iterations: Maximum number of iterations
            max_time: Maximum planning time (seconds)

        Returns:
            path: List of configurations from start to goal (None if failed)
            info: Dictionary with planning statistics
        """
        # Reset
        self.num_iterations = 0
        self.goal_node = None
        self.c_best = float('inf')
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

        # Initialize tree
        root = RRTNode(start_config)
        root.cost = 0.0
        self.tree = RRTTree(root)

        # Main planning loop
        for iteration in range(max_iterations):
            self.num_iterations = iteration + 1

            # Check timeout
            if time.time() - start_time > max_time:
                break

            # Informed sampling (sample in ellipsoid if solution found)
            if self.goal_node is not None:
                self.c_best = self.goal_node.cost
                sample = informed_sampling(
                    start_config, goal_config, self.c_best, self.config_space
                )
            else:
                sample = sample_goal_bias(goal_config, self.config_space, self.goal_bias)

            # Rest is same as RRT*
            nearest_node = self.tree.find_nearest_node(sample)
            new_config = steer(nearest_node.config, sample, self.max_step_size)

            if not self.config_space.is_config_valid(new_config):
                continue

            # Find near nodes
            search_radius = compute_rrt_star_radius(
                len(self.tree.nodes),
                self.config_space.n_joints,
                self.rewire_factor
            )
            near_nodes = self.tree.find_near_nodes(new_config, search_radius)

            if len(near_nodes) == 0:
                near_nodes = [nearest_node]

            # Choose best parent
            best_parent = None
            best_cost = float('inf')

            for near_node in near_nodes:
                if not self.collision_checker.check_segment_collision_free(
                    near_node.config, new_config, step_size=0.05
                ):
                    continue

                cost = near_node.cost + distance_metric(near_node.config, new_config)

                if cost < best_cost:
                    best_cost = cost
                    best_parent = near_node

            if best_parent is None:
                continue

            # Add new node
            new_node = RRTNode(new_config, parent=best_parent)
            new_node.cost = best_cost
            self.tree.add_node(new_node)

            # Rewire
            self._rewire(new_node, near_nodes)

            # Check goal
            if distance_metric(new_config, goal_config) <= self.goal_tolerance:
                if self.collision_checker.check_segment_collision_free(
                    new_config, goal_config, step_size=0.05
                ):
                    goal_cost = new_node.cost + distance_metric(new_config, goal_config)

                    if self.goal_node is None or goal_cost < self.goal_node.cost:
                        self.goal_node = RRTNode(goal_config, parent=new_node)
                        self.goal_node.cost = goal_cost

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
            'path_length': self._compute_path_length(path) if path else 0.0,
            'path_cost': self.goal_node.cost if self.goal_node else 0.0
        }

        return path, info
