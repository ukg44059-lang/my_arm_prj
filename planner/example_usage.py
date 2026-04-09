"""
Example: RRT Path Planning for Robot Arm
示例：机械臂的RRT路径规划

This script demonstrates how to use the RRT planning library with MuJoCo robot arm.
"""

import numpy as np
import mujoco
import os
import sys

# Add planner to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planner_utils import (
    ConfigurationSpace,
    compute_path_statistics
)
from collision_checker import MuJoCoCollisionChecker
from rrt_base import RRTPlanner
from rrt_variants import (
    RRTStarPlanner,
    RRTConnectPlanner,
    InformedRRTStarPlanner
)
from path_smoother import (
    PathSmoother,
    TrajectoryParameterizer
)


def example_basic_rrt():
    """
    Example: Basic RRT planning
    """
    print("\n" + "="*60)
    print("Example 1: Basic RRT Planning")
    print("="*60)

    # Load MuJoCo model
    model_path = os.path.join(
        os.path.dirname(__file__),
        "../../robot_model/exp/env_robot_torque.xml"
    )

    if not os.path.exists(model_path):
        print(f"⚠ Model file not found: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Define configuration space (6-DOF robot arm)
    joint_limits = np.array([
        [-2.96706, 2.96706],  # Joint 1
        [-2.09440, 2.09440],  # Joint 2
        [-2.96706, 2.96706],  # Joint 3
        [-2.09440, 2.09440],  # Joint 4
        [-2.96706, 2.96706],  # Joint 5
        [-2.09440, 2.09440],  # Joint 6
    ])

    config_space = ConfigurationSpace(joint_limits)

    # Initialize collision checker
    collision_checker = MuJoCoCollisionChecker(
        model=model,
        data=data,
        joint_names=[f"joint{i+1}" for i in range(6)],
        check_self_collision=True,
        exclude_bodies=['floor']  # Exclude floor from collision checking
    )

    # Define start and goal configurations
    start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_config = np.array([1.5, 0.5, -0.5, 0.3, 0.2, 0.1])

    print(f"\nStart: {start_config}")
    print(f"Goal:  {goal_config}")

    # Create RRT planner
    planner = RRTPlanner(
        config_space=config_space,
        collision_checker=collision_checker,
        max_step_size=0.3,
        goal_bias=0.1,
        goal_tolerance=0.15
    )

    # Plan path
    print("\n🚀 Planning with RRT...")
    path, info = planner.plan(
        start_config=start_config,
        goal_config=goal_config,
        max_iterations=3000,
        max_time=10.0
    )

    # Print results
    if info['success']:
        print(f"✓ Path found!")
        print(f"  - Planning time: {info['planning_time']:.3f}s")
        print(f"  - Iterations: {info['num_iterations']}")
        print(f"  - Nodes in tree: {info['num_nodes']}")
        print(f"  - Path length: {info['path_length']:.3f} rad")
        print(f"  - Waypoints: {len(path)}")

        # Compute statistics
        stats = compute_path_statistics(path)
        print(f"\n📊 Path Statistics:")
        print(f"  - Total length: {stats['length']:.3f} rad")
        print(f"  - Avg segment: {stats['avg_segment_length']:.3f} rad")
        print(f"  - Max segment: {stats['max_segment_length']:.3f} rad")

    else:
        print(f"✗ Planning failed!")
        print(f"  - Reason: {info.get('error', 'Unknown')}")
        print(f"  - Time spent: {info['planning_time']:.3f}s")

    return path, info


def example_rrt_star():
    """
    Example: RRT* planning (optimal)
    """
    print("\n" + "="*60)
    print("Example 2: RRT* Planning (Optimal)")
    print("="*60)

    # Load model
    model_path = os.path.join(
        os.path.dirname(__file__),
        "../../robot_model/exp/env_robot_torque.xml"
    )

    if not os.path.exists(model_path):
        print(f"⚠ Model file not found: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Configuration space
    joint_limits = np.array([
        [-2.96706, 2.96706],  # Joint 1
        [-2.09440, 2.09440],  # Joint 2
        [-2.96706, 2.96706],  # Joint 3
        [-2.09440, 2.09440],  # Joint 4
        [-2.96706, 2.96706],  # Joint 5
        [-2.09440, 2.09440],  # Joint 6
    ])

    config_space = ConfigurationSpace(joint_limits)

    # Collision checker
    collision_checker = MuJoCoCollisionChecker(
        model=model,
        data=data,
        joint_names=[f"joint{i+1}" for i in range(6)],
        exclude_bodies=['floor']
    )

    # Start and goal
    start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_config = np.array([1.2, 0.8, -0.6, 0.4, 0.3, 0.2])

    print(f"\nStart: {start_config}")
    print(f"Goal:  {goal_config}")

    # Create RRT* planner
    planner = RRTStarPlanner(
        config_space=config_space,
        collision_checker=collision_checker,
        max_step_size=0.25,
        goal_bias=0.1,
        goal_tolerance=0.15,
        rewire_factor=2.0
    )

    # Plan
    print("\n🚀 Planning with RRT*...")
    path, info = planner.plan(
        start_config=start_config,
        goal_config=goal_config,
        max_iterations=3000,
        max_time=15.0
    )

    # Results
    if info['success']:
        print(f"✓ Optimal path found!")
        print(f"  - Planning time: {info['planning_time']:.3f}s")
        print(f"  - Iterations: {info['num_iterations']}")
        print(f"  - Nodes in tree: {info['num_nodes']}")
        print(f"  - Path cost: {info['path_cost']:.3f}")
        print(f"  - Waypoints: {len(path)}")
    else:
        print(f"✗ Planning failed: {info.get('error', 'Unknown')}")

    return path, info


def example_rrt_connect():
    """
    Example: RRT-Connect planning (fast bidirectional)
    """
    print("\n" + "="*60)
    print("Example 3: RRT-Connect Planning (Fast Bidirectional)")
    print("="*60)

    # Load model
    model_path = os.path.join(
        os.path.dirname(__file__),
        "../../robot_model/exp/env_robot_torque.xml"
    )

    if not os.path.exists(model_path):
        print(f"⚠ Model file not found: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Configuration space
    joint_limits = np.array([
        [-2.96706, 2.96706],
        [-2.09440, 2.09440],
        [-2.96706, 2.96706],
        [-2.09440, 2.09440],
        [-2.96706, 2.96706],
        [-2.09440, 2.09440],
    ])

    config_space = ConfigurationSpace(joint_limits)

    # Collision checker
    collision_checker = MuJoCoCollisionChecker(
        model=model,
        data=data,
        joint_names=[f"joint{i+1}" for i in range(6)],
        exclude_bodies=['floor']
    )

    # Start and goal
    start_config = np.array([0.5, -0.3, 0.2, 0.0, 0.0, 0.0])
    goal_config = np.array([-0.5, 0.8, -0.4, 0.2, 0.1, 0.0])

    print(f"\nStart: {start_config}")
    print(f"Goal:  {goal_config}")

    # Create RRT-Connect planner
    planner = RRTConnectPlanner(
        config_space=config_space,
        collision_checker=collision_checker,
        max_step_size=0.3,
        goal_tolerance=0.15
    )

    # Plan
    print("\n🚀 Planning with RRT-Connect...")
    path, info = planner.plan(
        start_config=start_config,
        goal_config=goal_config,
        max_iterations=2000,
        max_time=10.0
    )

    # Results
    if info['success']:
        print(f"✓ Path found quickly!")
        print(f"  - Planning time: {info['planning_time']:.3f}s")
        print(f"  - Iterations: {info['num_iterations']}")
        print(f"  - Total nodes: {info['num_nodes']}")
        print(f"  - Waypoints: {len(path)}")
    else:
        print(f"✗ Planning failed: {info.get('error', 'Unknown')}")

    return path, info


def example_path_smoothing():
    """
    Example: Path smoothing and trajectory parameterization
    """
    print("\n" + "="*60)
    print("Example 4: Path Smoothing & Trajectory Parameterization")
    print("="*60)

    # First plan a path with RRT
    model_path = os.path.join(
        os.path.dirname(__file__),
        "../../robot_model/exp/env_robot_torque.xml"
    )

    if not os.path.exists(model_path):
        print(f"⚠ Model file not found: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    joint_limits = np.array([
        [-2.96706, 2.96706],
        [-2.09440, 2.09440],
        [-2.96706, 2.96706],
        [-2.09440, 2.09440],
        [-2.96706, 2.96706],
        [-2.09440, 2.09440],
    ])

    config_space = ConfigurationSpace(joint_limits)
    collision_checker = MuJoCoCollisionChecker(
        model=model,
        data=data,
        joint_names=[f"joint{i+1}" for i in range(6)],
        exclude_bodies=['floor']
    )

    planner = RRTPlanner(
        config_space=config_space,
        collision_checker=collision_checker,
        max_step_size=0.3,
        goal_bias=0.1
    )

    start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_config = np.array([1.0, 0.5, -0.3, 0.2, 0.1, 0.0])

    print("\n🚀 Planning path...")
    path, info = planner.plan(start_config, goal_config, max_iterations=2000)

    if not info['success']:
        print("✗ Planning failed, cannot demonstrate smoothing")
        return

    print(f"✓ Original path: {len(path)} waypoints")

    # Smooth path
    smoother = PathSmoother(collision_checker=collision_checker)

    print("\n📐 Smoothing path with shortcutting...")
    smoothed_shortcut = smoother.smooth_shortcut(path, max_iterations=50)
    print(f"  - Shortcut smoothed: {len(smoothed_shortcut)} waypoints")

    print("\n📐 Smoothing with cubic spline...")
    smoothed_spline = smoother.smooth_cubic_spline(path, num_samples=50)
    print(f"  - Spline smoothed: {len(smoothed_spline)} waypoints")

    # Trajectory parameterization
    print("\n⏱ Parameterizing trajectory...")
    parameterizer = TrajectoryParameterizer()

    traj_path, timestamps = parameterizer.parameterize_trapezoidal(
        smoothed_shortcut,
        max_vel=1.0,
        max_acc=2.0
    )

    print(f"  - Total duration: {timestamps[-1]:.3f}s")
    print(f"  - Waypoints with timestamps: {len(traj_path)}")

    # Interpolate to high resolution
    interp_path, interp_times = parameterizer.interpolate_trajectory(
        traj_path,
        timestamps,
        dt=0.01
    )

    print(f"  - Interpolated to {len(interp_path)} points at 100Hz")

    return smoothed_shortcut, smoothed_spline, interp_path


def compare_all_planners():
    """
    Example: Compare all RRT variants
    """
    print("\n" + "="*60)
    print("Example 5: Comparing All RRT Variants")
    print("="*60)

    # Setup
    model_path = os.path.join(
        os.path.dirname(__file__),
        "../../robot_model/exp/env_robot_torque.xml"
    )

    if not os.path.exists(model_path):
        print(f"⚠ Model file not found: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    joint_limits = np.array([
        [-2.96706, 2.96706],
        [-2.09440, 2.09440],
        [-2.96706, 2.96706],
        [-2.09440, 2.09440],
        [-2.96706, 2.96706],
        [-2.09440, 2.09440],
    ])

    config_space = ConfigurationSpace(joint_limits)
    collision_checker = MuJoCoCollisionChecker(
        model=model,
        data=data,
        joint_names=[f"joint{i+1}" for i in range(6)],
        exclude_bodies=['floor']
    )

    start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_config = np.array([1.5, 0.6, -0.5, 0.3, 0.2, 0.1])

    # Test all planners
    planners = {
        'RRT': RRTPlanner(config_space, collision_checker, max_step_size=0.3),
        'RRT*': RRTStarPlanner(config_space, collision_checker, max_step_size=0.25),
        'RRT-Connect': RRTConnectPlanner(config_space, collision_checker, max_step_size=0.3),
        'Informed RRT*': InformedRRTStarPlanner(config_space, collision_checker, max_step_size=0.25),
    }

    results = {}

    for name, planner in planners.items():
        print(f"\n🚀 Testing {name}...")
        path, info = planner.plan(start_config, goal_config, max_iterations=2000, max_time=10.0)
        results[name] = info

        if info['success']:
            print(f"  ✓ Success in {info['planning_time']:.3f}s")
            print(f"    - Waypoints: {len(path)}")
            print(f"    - Nodes: {info['num_nodes']}")
            if 'path_cost' in info:
                print(f"    - Cost: {info['path_cost']:.3f}")
        else:
            print(f"  ✗ Failed")

    # Summary
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Planner':<20} {'Success':<10} {'Time(s)':<10} {'Nodes':<10}")
    print("-"*60)

    for name, info in results.items():
        success = "✓" if info['success'] else "✗"
        time_str = f"{info['planning_time']:.3f}"
        nodes_str = str(info['num_nodes'])
        print(f"{name:<20} {success:<10} {time_str:<10} {nodes_str:<10}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RRT PATH PLANNING LIBRARY - EXAMPLES")
    print("="*60)

    try:
        # Run examples
        example_basic_rrt()
        example_rrt_star()
        example_rrt_connect()
        example_path_smoothing()
        compare_all_planners()

        print("\n" + "="*60)
        print("✓ All examples completed!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
