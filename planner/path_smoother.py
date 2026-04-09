"""
Path Smoother - Trajectory optimization and smoothing algorithms
路径平滑器 - 轨迹优化和平滑算法
"""

import numpy as np
from typing import List, Optional, Callable, Tuple
from scipy.interpolate import CubicSpline, splprep, splev

try:
    from .planner_utils import (
        distance_metric, interpolate_path,
        simplify_path, resample_path
    )
except ImportError:
    from planner_utils import (
        distance_metric, interpolate_path,
        simplify_path, resample_path
    )


class PathSmoother:
    """
    Path smoothing using various techniques
    使用多种技术进行路径平滑
    """

    def __init__(self, collision_checker=None):
        """
        Initialize path smoother

        Args:
            collision_checker: Optional collision checker for validation
        """
        self.collision_checker = collision_checker

    def smooth_shortcut(self,
                       path: List[np.ndarray],
                       max_iterations: int = 100,
                       collision_check: bool = True) -> List[np.ndarray]:
        """
        Smooth path by shortcutting (removing unnecessary waypoints)

        Args:
            path: Original path
            max_iterations: Maximum number of shortcut attempts
            collision_check: Whether to check collision when shortcutting

        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path

        smoothed = path.copy()

        for _ in range(max_iterations):
            if len(smoothed) <= 2:
                break

            # Randomly select two waypoints
            i = np.random.randint(0, len(smoothed) - 2)
            j = np.random.randint(i + 2, len(smoothed))

            # Try to connect directly
            can_shortcut = True

            if collision_check and self.collision_checker is not None:
                can_shortcut = self.collision_checker.check_segment_collision_free(
                    smoothed[i], smoothed[j], step_size=0.05
                )

            if can_shortcut:
                # Remove intermediate waypoints
                smoothed = smoothed[:i+1] + smoothed[j:]

        return smoothed

    def smooth_cubic_spline(self,
                           path: List[np.ndarray],
                           num_samples: int = 100) -> List[np.ndarray]:
        """
        Smooth path using cubic spline interpolation

        Args:
            path: Original path
            num_samples: Number of samples in smoothed path

        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path

        # Convert path to array
        path_array = np.array(path)
        n_waypoints, n_dims = path_array.shape

        # Parameter for waypoints (0 to 1)
        t = np.linspace(0, 1, n_waypoints)

        # Create cubic splines for each dimension
        smoothed_path = []
        t_smooth = np.linspace(0, 1, num_samples)

        for dim in range(n_dims):
            cs = CubicSpline(t, path_array[:, dim])
            smoothed_dim = cs(t_smooth)

            if dim == 0:
                smoothed_path = [[val] for val in smoothed_dim]
            else:
                for i, val in enumerate(smoothed_dim):
                    smoothed_path[i].append(val)

        # Convert to list of arrays
        smoothed_path = [np.array(config) for config in smoothed_path]

        return smoothed_path

    def smooth_b_spline(self,
                       path: List[np.ndarray],
                       num_samples: int = 100,
                       smoothness: float = 0.0) -> List[np.ndarray]:
        """
        Smooth path using B-spline interpolation

        Args:
            path: Original path
            num_samples: Number of samples in smoothed path
            smoothness: Smoothness parameter (0 = interpolation, >0 = approximation)

        Returns:
            Smoothed path
        """
        if len(path) <= 3:
            return path

        # Convert path to array
        path_array = np.array(path).T  # Transpose for splprep

        # Fit B-spline
        try:
            tck, u = splprep(path_array, s=smoothness, k=min(3, len(path) - 1))

            # Evaluate spline
            u_smooth = np.linspace(0, 1, num_samples)
            smoothed = splev(u_smooth, tck)

            # Convert back to list of arrays
            smoothed_path = [np.array(config) for config in zip(*smoothed)]

            return smoothed_path

        except Exception as e:
            print(f"⚠ B-spline smoothing failed: {e}, returning original path")
            return path

    def smooth_bezier(self,
                     path: List[np.ndarray],
                     num_samples: int = 100) -> List[np.ndarray]:
        """
        Smooth path using Bezier curves (piecewise)

        Args:
            path: Original path
            num_samples: Number of samples per segment

        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path

        smoothed = []

        # Process each segment with cubic Bezier curve
        for i in range(len(path) - 1):
            p0 = path[i]
            p3 = path[i + 1]

            # Control points (simple tangent-based approach)
            if i > 0:
                tangent_start = (path[i] - path[i-1]) / 3.0
            else:
                tangent_start = (p3 - p0) / 3.0

            if i < len(path) - 2:
                tangent_end = (path[i+2] - path[i+1]) / 3.0
            else:
                tangent_end = (p3 - p0) / 3.0

            p1 = p0 + tangent_start
            p2 = p3 - tangent_end

            # Sample Bezier curve
            t_vals = np.linspace(0, 1, num_samples if i == len(path) - 2 else num_samples - 1)

            for t in t_vals:
                # Cubic Bezier formula
                point = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
                smoothed.append(point)

        # Add final point
        if len(smoothed) == 0 or not np.allclose(smoothed[-1], path[-1]):
            smoothed.append(path[-1])

        return smoothed

    def smooth_gradient_descent(self,
                               path: List[np.ndarray],
                               max_iterations: int = 100,
                               step_size: float = 0.01,
                               smoothness_weight: float = 1.0,
                               obstacle_weight: float = 10.0) -> List[np.ndarray]:
        """
        Smooth path using gradient descent optimization

        Args:
            path: Original path
            max_iterations: Maximum optimization iterations
            step_size: Gradient descent step size
            smoothness_weight: Weight for smoothness term
            obstacle_weight: Weight for obstacle avoidance

        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path

        # Convert to array (keep start and goal fixed)
        smoothed = np.array(path, dtype=float)
        n_waypoints = len(smoothed)

        # Optimize internal waypoints only
        for iteration in range(max_iterations):
            gradient = np.zeros_like(smoothed)

            # Smoothness gradient (encourage straight lines)
            for i in range(1, n_waypoints - 1):
                # Second derivative approximation
                smoothness_grad = 2 * smoothed[i] - smoothed[i-1] - smoothed[i+1]
                gradient[i] += smoothness_weight * smoothness_grad

            # Obstacle gradient (push away from obstacles if available)
            if self.collision_checker is not None and hasattr(self.collision_checker, 'get_clearance'):
                for i in range(1, n_waypoints - 1):
                    clearance = self.collision_checker.get_clearance(smoothed[i])

                    if clearance < 0.1:  # Close to obstacle
                        # Simple repulsion (would need proper gradient for real implementation)
                        obstacle_grad = np.random.randn(len(smoothed[i])) * 0.1
                        gradient[i] -= obstacle_weight * obstacle_grad

            # Update waypoints (except start and goal)
            smoothed[1:-1] -= step_size * gradient[1:-1]

        return [config for config in smoothed]


class TrajectoryParameterizer:
    """
    Convert geometric path to time-parameterized trajectory
    将几何路径转换为时间参数化轨迹
    """

    def __init__(self, max_velocity: Optional[np.ndarray] = None,
                 max_acceleration: Optional[np.ndarray] = None):
        """
        Initialize trajectory parameterizer

        Args:
            max_velocity: Maximum velocity for each joint (rad/s)
            max_acceleration: Maximum acceleration for each joint (rad/s^2)
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def parameterize_constant_velocity(self,
                                      path: List[np.ndarray],
                                      velocity: float = 0.5) -> Tuple[List[np.ndarray], List[float]]:
        """
        Parameterize path with constant velocity

        Args:
            path: Geometric path
            velocity: Constant velocity (rad/s)

        Returns:
            path, timestamps
        """
        if len(path) <= 1:
            return path, [0.0]

        timestamps = [0.0]
        total_time = 0.0

        for i in range(len(path) - 1):
            distance = distance_metric(path[i], path[i+1])
            dt = distance / velocity
            total_time += dt
            timestamps.append(total_time)

        return path, timestamps

    def parameterize_trapezoidal(self,
                                path: List[np.ndarray],
                                max_vel: float = 1.0,
                                max_acc: float = 2.0) -> Tuple[List[np.ndarray], List[float]]:
        """
        Parameterize path with trapezoidal velocity profile

        Args:
            path: Geometric path
            max_vel: Maximum velocity (rad/s)
            max_acc: Maximum acceleration (rad/s^2)

        Returns:
            path, timestamps
        """
        if len(path) <= 1:
            return path, [0.0]

        timestamps = [0.0]
        total_time = 0.0

        for i in range(len(path) - 1):
            distance = distance_metric(path[i], path[i+1])

            # Compute time using trapezoidal profile
            # Time to reach max velocity
            t_acc = max_vel / max_acc

            # Distance during acceleration and deceleration
            d_acc = 0.5 * max_acc * t_acc ** 2
            d_total_acc = 2 * d_acc

            if distance <= d_total_acc:
                # Triangular profile (doesn't reach max velocity)
                t_segment = 2 * np.sqrt(distance / max_acc)
            else:
                # Trapezoidal profile
                d_constant = distance - d_total_acc
                t_constant = d_constant / max_vel
                t_segment = 2 * t_acc + t_constant

            total_time += t_segment
            timestamps.append(total_time)

        return path, timestamps

    def interpolate_trajectory(self,
                              path: List[np.ndarray],
                              timestamps: List[float],
                              dt: float = 0.01) -> Tuple[List[np.ndarray], List[float]]:
        """
        Interpolate trajectory at regular time intervals

        Args:
            path: Geometric path
            timestamps: Original timestamps
            dt: Time step for interpolation

        Returns:
            interpolated_path, interpolated_timestamps
        """
        if len(path) <= 1:
            return path, timestamps

        # Create cubic spline for each dimension
        path_array = np.array(path)
        n_dims = path_array.shape[1]

        # New timestamps
        t_new = np.arange(timestamps[0], timestamps[-1], dt)
        t_new = np.append(t_new, timestamps[-1])  # Ensure final time included

        # Interpolate
        interpolated = []
        for dim in range(n_dims):
            cs = CubicSpline(timestamps, path_array[:, dim])
            interpolated.append(cs(t_new))

        # Convert to list of configs
        interpolated_path = [
            np.array([interpolated[dim][i] for dim in range(n_dims)])
            for i in range(len(t_new))
        ]

        return interpolated_path, t_new.tolist()


def compute_path_statistics(path: List[np.ndarray]) -> dict:
    """
    Compute statistics about the path

    Args:
        path: Path to analyze

    Returns:
        Dictionary of statistics
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
    for i in range(len(path) - 1):
        length = distance_metric(path[i], path[i+1])
        segment_lengths.append(length)

    total_length = sum(segment_lengths)

    return {
        'length': total_length,
        'num_waypoints': len(path),
        'avg_segment_length': np.mean(segment_lengths),
        'max_segment_length': np.max(segment_lengths),
        'min_segment_length': np.min(segment_lengths),
        'std_segment_length': np.std(segment_lengths)
    }
