"""
Cartesian Space RRT-Connect Planner
笛卡尔空间 RRT-Connect 规划器

This planner works in Cartesian (task) space and uses IK to convert to joint space.
该规划器在笛卡尔（任务）空间中工作，并使用逆运动学转换到关节空间。
"""

import numpy as np
import time
from typing import Optional, List, Tuple, Callable
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from planner.planner_utils import distance_metric
    from planner.collision_checker import MuJoCoCollisionChecker
    from arm_ik.arm_ik import ArmInverseKinematics
except ImportError:
    from planner_utils import distance_metric
    from collision_checker import MuJoCoCollisionChecker
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'arm_ik'))
    from arm_ik import ArmInverseKinematics


class CartesianNode:
    """Node in Cartesian space tree"""

    def __init__(self, pose: np.ndarray, joint_config: Optional[np.ndarray] = None):
        """
        Args:
            pose: 4x4 homogeneous transformation matrix (SE3)
            joint_config: Joint configuration corresponding to this pose
        """
        self.pose = pose  # 4x4 matrix
        self.joint_config = joint_config  # Joint angles
        self.parent: Optional['CartesianNode'] = None
        self.children: List['CartesianNode'] = []

    def __repr__(self):
        pos = self.pose[:3, 3]
        return f"CartesianNode(pos={pos})"


class CartesianTree:
    """Tree structure for Cartesian space RRT"""

    def __init__(self, root: CartesianNode):
        self.root = root
        self.nodes: List[CartesianNode] = [root]

    def add_node(self, node: CartesianNode):
        """Add node to tree"""
        self.nodes.append(node)
        if node.parent is not None:
            node.parent.children.append(node)

    def find_nearest_node(self, target_pose: np.ndarray) -> CartesianNode:
        """Find nearest node to target pose using position distance"""
        target_pos = target_pose[:3, 3]

        min_dist = float('inf')
        nearest = self.root

        for node in self.nodes:
            node_pos = node.pose[:3, 3]
            dist = np.linalg.norm(target_pos - node_pos)
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest

    def get_path_to_node(self, node: CartesianNode) -> List[CartesianNode]:
        """Extract path from root to node"""
        path = []
        current = node

        while current is not None:
            path.append(current)
            current = current.parent

        path.reverse()
        return path


class CartesianRRTConnect:
    """
    RRT-Connect planner in Cartesian space with IK
    笛卡尔空间 RRT-Connect 规划器（带逆运动学）
    """

    def __init__(self,
                 ik_solver: ArmInverseKinematics,
                 collision_checker: MuJoCoCollisionChecker,
                 max_step_size: float = 0.05,  # meters in Cartesian space
                 goal_tolerance: float = 0.01,  # meters
                 max_ik_attempts: int = 5):
        """
        Initialize Cartesian RRT-Connect planner

        Args:
            ik_solver: Inverse kinematics solver
            collision_checker: Collision checker for joint configurations
            max_step_size: Maximum step size in Cartesian space (meters)
            goal_tolerance: Goal tolerance in Cartesian space (meters)
            max_ik_attempts: Maximum IK solution attempts per sample
        """
        self.ik_solver = ik_solver
        self.collision_checker = collision_checker
        self.max_step_size = max_step_size
        self.goal_tolerance = goal_tolerance
        self.max_ik_attempts = max_ik_attempts

        # Workspace bounds (can be customized)
        self.workspace_min = np.array([-1.0, -1.0, 0.0])  # meters
        self.workspace_max = np.array([1.0, 1.0, 1.5])     # meters

        # Planning state
        self.tree_start: Optional[CartesianTree] = None
        self.tree_goal: Optional[CartesianTree] = None

        # Statistics
        self.planning_time = 0.0
        self.num_iterations = 0
        self.num_nodes = 0
        self.num_ik_failures = 0

    def set_workspace_bounds(self, min_bounds: np.ndarray, max_bounds: np.ndarray):
        """Set workspace bounds for sampling"""
        self.workspace_min = min_bounds
        self.workspace_max = max_bounds

    def sample_cartesian_pose(self) -> np.ndarray:
        """Sample random pose in workspace"""
        # Sample random position
        pos = np.random.uniform(self.workspace_min, self.workspace_max)

        # Sample random orientation (random rotation matrix)
        # Using axis-angle representation
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(0, 2 * np.pi)

        # Rodrigues' formula for rotation matrix
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

        # Construct 4x4 transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = pos

        return pose

    def cartesian_distance(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """Compute distance between two poses (position only for simplicity)"""
        pos1 = pose1[:3, 3]
        pos2 = pose2[:3, 3]
        return np.linalg.norm(pos2 - pos1)

    def steer_cartesian(self, from_pose: np.ndarray, to_pose: np.ndarray) -> np.ndarray:
        """Steer from one pose towards another by max_step_size"""
        from_pos = from_pose[:3, 3]
        to_pos = to_pose[:3, 3]

        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)

        if dist <= self.max_step_size:
            return to_pose.copy()

        # Interpolate position
        alpha = self.max_step_size / dist
        new_pos = from_pos + alpha * direction

        # For orientation, use SLERP (simplified: just copy for now)
        # TODO: Implement proper SLERP for rotation matrices
        new_pose = to_pose.copy()
        new_pose[:3, 3] = new_pos

        return new_pose

    def solve_ik_with_seed(self, target_pose: np.ndarray,
                          seed_config: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Solve IK for target pose with given seed configuration

        Returns:
            success: Whether IK succeeded
            joint_config: Joint configuration (None if failed)
        """
        success, joint_config = self.ik_solver.get_ik(target_pose, seed_config)

        if not success:
            return False, None

        # Check collision
        if self.collision_checker.check_collision_at_config(joint_config):
            return False, None

        return True, joint_config

    def plan(self,
             start_pose: np.ndarray,
             goal_pose: np.ndarray,
             start_joint_config: np.ndarray,
             goal_joint_config: Optional[np.ndarray] = None,
             max_iterations: int = 1000,
             max_time: float = 30.0) -> Tuple[Optional[List[np.ndarray]],
                                               Optional[List[np.ndarray]], dict]:
        """
        Plan path from start to goal pose in Cartesian space

        Args:
            start_pose: Start pose (4x4 matrix)
            goal_pose: Goal pose (4x4 matrix)
            start_joint_config: Joint configuration at start
            goal_joint_config: Joint configuration at goal (optional, will solve IK if None)
            max_iterations: Maximum planning iterations
            max_time: Maximum planning time (seconds)

        Returns:
            cartesian_path: List of poses from start to goal (None if failed)
            joint_path: List of joint configurations (None if failed)
            info: Planning statistics
        """
        # Reset statistics
        self.num_iterations = 0
        self.num_ik_failures = 0
        start_time = time.time()

        # Solve IK for goal if not provided
        if goal_joint_config is None:
            print("Solving IK for goal pose...")
            success, goal_joint_config = self.solve_ik_with_seed(goal_pose, start_joint_config)
            if not success:
                return None, None, {
                    'success': False,
                    'error': 'Failed to solve IK for goal pose'
                }

        # Validate start and goal
        if self.collision_checker.check_collision_at_config(start_joint_config):
            return None, None, {'success': False, 'error': 'Start configuration in collision'}

        if self.collision_checker.check_collision_at_config(goal_joint_config):
            return None, None, {'success': False, 'error': 'Goal configuration in collision'}

        # Initialize trees
        root_start = CartesianNode(start_pose, start_joint_config)
        root_goal = CartesianNode(goal_pose, goal_joint_config)
        self.tree_start = CartesianTree(root_start)
        self.tree_goal = CartesianTree(root_goal)

        # Track trees for swapping
        tree_a = self.tree_start
        tree_b = self.tree_goal

        connection_node_a: Optional[CartesianNode] = None
        connection_node_b: Optional[CartesianNode] = None

        # Main planning loop
        for iteration in range(max_iterations):
            self.num_iterations = iteration + 1

            # Check timeout
            if time.time() - start_time > max_time:
                break

            # Sample random pose
            sample_pose = self.sample_cartesian_pose()

            # Extend tree A towards sample
            new_node_a = self._extend(tree_a, sample_pose)

            if new_node_a is not None:
                # Try to connect tree B to new node
                connection_result = self._connect(tree_b, new_node_a.pose)

                if connection_result is not None:
                    # Trees connected!
                    connection_node_a = new_node_a
                    connection_node_b = connection_result
                    break

            # Swap trees for balanced growth
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

            path_nodes = path_start + path_goal[1:]

            # Extract poses and joint configs
            cartesian_path = [node.pose for node in path_nodes]
            joint_path = [node.joint_config for node in path_nodes]

            success = True
        else:
            cartesian_path = None
            joint_path = None
            success = False

        # Compile statistics
        path_length = 0.0
        if cartesian_path:
            for i in range(len(cartesian_path) - 1):
                path_length += self.cartesian_distance(cartesian_path[i], cartesian_path[i + 1])

        info = {
            'success': success,
            'planning_time': self.planning_time,
            'num_iterations': self.num_iterations,
            'num_nodes': self.num_nodes,
            'num_ik_failures': self.num_ik_failures,
            'path_length': path_length,
            'num_waypoints': len(cartesian_path) if cartesian_path else 0
        }

        return cartesian_path, joint_path, info

    def _extend(self, tree: CartesianTree, target_pose: np.ndarray) -> Optional[CartesianNode]:
        """Extend tree towards target pose"""
        nearest_node = tree.find_nearest_node(target_pose)
        new_pose = self.steer_cartesian(nearest_node.pose, target_pose)

        # Solve IK for new pose
        seed = nearest_node.joint_config
        success, joint_config = self.solve_ik_with_seed(new_pose, seed)

        if not success:
            self.num_ik_failures += 1
            return None

        # Check segment collision-free
        if not self.collision_checker.check_segment_collision_free(
            nearest_node.joint_config, joint_config, step_size=0.05
        ):
            return None

        # Add to tree
        new_node = CartesianNode(new_pose, joint_config)
        new_node.parent = nearest_node
        tree.add_node(new_node)

        return new_node

    def _connect(self, tree: CartesianTree, target_pose: np.ndarray) -> Optional[CartesianNode]:
        """Aggressively try to connect tree to target pose"""
        current_node = tree.find_nearest_node(target_pose)

        max_connect_steps = 50
        for _ in range(max_connect_steps):
            # Check if reached target
            distance = self.cartesian_distance(current_node.pose, target_pose)

            if distance <= self.goal_tolerance:
                return current_node

            # Steer towards target
            new_pose = self.steer_cartesian(current_node.pose, target_pose)

            # Solve IK
            success, joint_config = self.solve_ik_with_seed(new_pose, current_node.joint_config)

            if not success:
                self.num_ik_failures += 1
                return None

            # Check collision
            if not self.collision_checker.check_segment_collision_free(
                current_node.joint_config, joint_config, step_size=0.05
            ):
                return None

            # Add to tree
            new_node = CartesianNode(new_pose, joint_config)
            new_node.parent = current_node
            tree.add_node(new_node)
            current_node = new_node

        return None


def interpolate_joint_path(joint_path: List[np.ndarray],
                          num_points_per_segment: int = 10) -> np.ndarray:
    """
    Interpolate joint path with more waypoints for smooth motion

    Args:
        joint_path: List of joint configurations
        num_points_per_segment: Number of interpolation points between waypoints

    Returns:
        interpolated_path: Numpy array of shape (N, num_joints)
    """
    if len(joint_path) < 2:
        return np.array(joint_path)

    interpolated = []

    for i in range(len(joint_path) - 1):
        start = joint_path[i]
        end = joint_path[i + 1]

        # Linear interpolation
        for j in range(num_points_per_segment):
            alpha = j / num_points_per_segment
            point = (1 - alpha) * start + alpha * end
            interpolated.append(point)

    # Add final point
    interpolated.append(joint_path[-1])

    return np.array(interpolated)
