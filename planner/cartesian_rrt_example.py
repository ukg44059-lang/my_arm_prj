"""
Example: Cartesian Space RRT-Connect Planning
示例：笛卡尔空间 RRT-Connect 规划

This demonstrates the complete workflow:
1. RRT-Connect planning in Cartesian space (end-effector space)
2. Inverse kinematics to convert poses to joint configurations
3. Interpolation for smooth motion
4. Execution on robot arm
"""

import numpy as np
import mujoco
import mujoco.viewer
import os
import sys
import time

# Add planner to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cartesian_rrt_connect import CartesianRRTConnect, interpolate_joint_path
from collision_checker import MuJoCoCollisionChecker
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'arm_ik'))
from arm_ik import ArmInverseKinematics


def get_ee_pose(model, data, ee_site_name='tools_link'):
    """Get current end-effector pose from MuJoCo"""
    site_id = model.site(ee_site_name).id
    pos = data.site_xpos[site_id].copy()
    mat = data.site_xmat[site_id].reshape(3, 3).copy()

    pose = np.eye(4)
    pose[:3, :3] = mat
    pose[:3, 3] = pos
    return pose


def set_joint_positions(model, data, joint_names, positions):
    """Set joint positions in MuJoCo"""
    for name, pos in zip(joint_names, positions):
        joint_id = model.joint(name).id
        qpos_addr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_addr] = pos


def visualize_trajectory_with_line(model, data, joint_path, interpolated_path, joint_names, ee_site_name='tools_link'):
    """
    Visualize trajectory - saves trajectory data and prints visualization info

    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_path: Original waypoints from planner
        interpolated_path: Interpolated joint configurations
        joint_names: List of joint names
        ee_site_name: End-effector site name
    """
    print("\n🎨 Visualizing trajectory...")
    print("Computing trajectory points...")

    # Extract end-effector positions along the path
    ee_positions = []
    ee_positions_waypoints = []

    # Get waypoint positions (original path)
    for joint_config in joint_path:
        set_joint_positions(model, data, joint_names, joint_config)
        mujoco.mj_forward(model, data)
        site_id = model.site(ee_site_name).id
        pos = data.site_xpos[site_id].copy()
        ee_positions_waypoints.append(pos)

    # Get interpolated positions
    for joint_config in interpolated_path:
        set_joint_positions(model, data, joint_names, joint_config)
        mujoco.mj_forward(model, data)
        site_id = model.site(ee_site_name).id
        pos = data.site_xpos[site_id].copy()
        ee_positions.append(pos)

    ee_positions = np.array(ee_positions)
    ee_positions_waypoints = np.array(ee_positions_waypoints)

    print(f"✓ Computed {len(ee_positions)} trajectory points")
    print(f"✓ Original waypoints: {len(ee_positions_waypoints)}")

    # Calculate path statistics
    path_length = 0
    for i in range(len(ee_positions) - 1):
        path_length += np.linalg.norm(ee_positions[i+1] - ee_positions[i])

    # Print trajectory information
    print("\n" + "="*60)
    print("📊 TRAJECTORY VISUALIZATION")
    print("="*60)

    print(f"\n🟢 Start Position: [{ee_positions[0][0]:.4f}, {ee_positions[0][1]:.4f}, {ee_positions[0][2]:.4f}] m")
    print(f"🔵 Goal Position:  [{ee_positions[-1][0]:.4f}, {ee_positions[-1][1]:.4f}, {ee_positions[-1][2]:.4f}] m")

    print(f"\n📏 Path Statistics:")
    print(f"  - Total length: {path_length:.4f} m")
    print(f"  - Planning waypoints: {len(ee_positions_waypoints)}")
    print(f"  - Interpolated points: {len(ee_positions)}")
    print(f"  - Average segment: {path_length/(len(ee_positions)-1):.4f} m")

    print(f"\n🔴 Waypoint Positions:")
    for i, pos in enumerate(ee_positions_waypoints):
        print(f"  [{i}] ({pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}) m")

    # Save trajectory to file
    trajectory_file = os.path.join(os.path.dirname(__file__), "planned_trajectory.npz")
    np.savez(trajectory_file,
             waypoints=ee_positions_waypoints,
             interpolated=ee_positions,
             joint_waypoints=np.array(joint_path),
             joint_interpolated=np.array(interpolated_path))

    print(f"\n💾 Trajectory saved to: {trajectory_file}")
    print("="*60)

    # Open MuJoCo viewer with robot showing the path
    print("\n🤖 Opening MuJoCo viewer...")
    print("The robot will animate along the planned trajectory")
    print("Press 'Ctrl+C' in terminal to stop\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("✓ Viewer opened - animating trajectory...")

            # Animate through the trajectory
            for loop in range(3):  # Loop 3 times
                print(f"\n▶ Loop {loop + 1}/3")

                for i, joint_config in enumerate(interpolated_path):
                    if not viewer.is_running():
                        break

                    set_joint_positions(model, data, joint_names, joint_config)
                    mujoco.mj_forward(model, data)
                    viewer.sync()

                    # Print progress every 20 steps
                    if i % 20 == 0:
                        current_pos = get_ee_pose(model, data)[:3, 3]
                        progress = (i / len(interpolated_path)) * 100
                        print(f"  Progress: {progress:5.1f}% | Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")

                    time.sleep(0.03)  # ~33Hz

                if not viewer.is_running():
                    break

                # Pause at goal
                time.sleep(1.0)

                # Return to start
                for joint_config in reversed(interpolated_path):
                    if not viewer.is_running():
                        break
                    set_joint_positions(model, data, joint_names, joint_config)
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(0.015)  # Faster return

                time.sleep(0.5)

            print("\n✓ Animation completed! Keeping viewer open...")
            print("Close the viewer window to exit.")

            # Keep viewer open at start position
            set_joint_positions(model, data, joint_names, interpolated_path[0])
            mujoco.mj_forward(model, data)

            while viewer.is_running():
                viewer.sync()
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n⏹ Visualization stopped by user")


def example_cartesian_planning():
    """
    Complete example: Cartesian RRT-Connect planning with IK and interpolation
    """
    print("\n" + "="*70)
    print("CARTESIAN SPACE RRT-CONNECT PLANNING")
    print("="*70)

    # ========== Setup ==========
    print("\n[1/6] Loading MuJoCo model...")

    model_path = os.path.join(
        os.path.dirname(__file__),
        "../../robot_model/exp/env_robot_torque.xml"
    )

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Joint names
    joint_names = [f"joint{i+1}" for i in range(6)]

    print(f"✓ Model loaded: {model.nq} DOF robot")

    # ========== Initialize IK Solver ==========
    print("\n[2/6] Initializing IK solver...")

    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "../../robot_model/robot/urdf/arm620.urdf"
    )

    if not os.path.exists(urdf_path):
        print(f"❌ URDF file not found: {urdf_path}")
        return

    # Initialize IK solver with 'tools_link' as end-effector
    ik_solver = ArmInverseKinematics(urdf_path, ee_link='tools_link')
    print("✓ IK solver initialized")

    # ========== Initialize Collision Checker ==========
    print("\n[3/6] Initializing collision checker...")

    collision_checker = MuJoCoCollisionChecker(
        model=model,
        data=data,
        joint_names=joint_names,
        check_self_collision=True,
        exclude_bodies=['floor']
    )
    print("✓ Collision checker initialized")

    # ========== Define Start and Goal ==========
    print("\n[4/6] Defining start and goal configurations...")

    # Start configuration
    start_joint_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Set start configuration and get pose
    set_joint_positions(model, data, joint_names, start_joint_config)
    mujoco.mj_forward(model, data)
    start_pose = get_ee_pose(model, data)

    print(f"Start joint config: {start_joint_config}")
    print(f"Start position: {start_pose[:3, 3]}")

    # Goal: Use a known reachable joint configuration
    goal_joint_config = np.array([0.5, 0.3, -0.2, 0.1, 0.0, 0.0])

    # Get goal pose from goal configuration
    set_joint_positions(model, data, joint_names, goal_joint_config)
    mujoco.mj_forward(model, data)
    goal_pose = get_ee_pose(model, data)

    # Reset back to start
    set_joint_positions(model, data, joint_names, start_joint_config)
    mujoco.mj_forward(model, data)

    print(f"Goal joint config: {goal_joint_config}")
    print(f"Goal position: {goal_pose[:3, 3]}")

    # ========== Initialize Cartesian RRT-Connect Planner ==========
    print("\n[5/6] Planning with Cartesian RRT-Connect...")

    planner = CartesianRRTConnect(
        ik_solver=ik_solver,
        collision_checker=collision_checker,
        max_step_size=0.05,  # 5cm steps in Cartesian space
        goal_tolerance=0.01,  # 1cm tolerance
        max_ik_attempts=5
    )

    # Set workspace bounds (adjust based on your robot)
    planner.set_workspace_bounds(
        min_bounds=np.array([0.0, -0.5, 0.0]),
        max_bounds=np.array([0.8, 0.5, 0.8])
    )

    # Plan path
    print("🚀 Planning path in Cartesian space...")
    cartesian_path, joint_path, info = planner.plan(
        start_pose=start_pose,
        goal_pose=goal_pose,
        start_joint_config=start_joint_config,
        goal_joint_config=goal_joint_config,  # Provide known reachable goal
        max_iterations=1000,
        max_time=30.0
    )

    # Print results
    if info['success']:
        print(f"✓ Path found!")
        print(f"  - Planning time: {info['planning_time']:.3f}s")
        print(f"  - Iterations: {info['num_iterations']}")
        print(f"  - Total nodes: {info['num_nodes']}")
        print(f"  - IK failures: {info['num_ik_failures']}")
        print(f"  - Path length: {info['path_length']:.3f}m")
        print(f"  - Waypoints: {info['num_waypoints']}")

        # ========== Interpolate Path ==========
        print("\n[6/6] Interpolating path for smooth motion...")

        interpolated_path = interpolate_joint_path(joint_path, num_points_per_segment=20)
        print(f"✓ Interpolated to {len(interpolated_path)} points")

        # ========== Visualize Trajectory with Red Line ==========
        visualize_trajectory_with_line(model, data, joint_path, interpolated_path, joint_names)

        # ========== Execute Trajectory (COMMENTED OUT) ==========
        # Uncomment below to execute the trajectory on the robot
        """
        print("\n🎬 Opening viewer to execute trajectory...")
        print("Press 'Ctrl+C' in terminal to stop")

        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # Reset to start
                set_joint_positions(model, data, joint_names, start_joint_config)
                mujoco.mj_forward(model, data)
                viewer.sync()

                # Wait a bit at start
                time.sleep(2.0)

                print("\n▶ Executing planned trajectory...")

                # Execute interpolated path
                for i, joint_config in enumerate(interpolated_path):
                    set_joint_positions(model, data, joint_names, joint_config)
                    mujoco.mj_forward(model, data)
                    viewer.sync()

                    # Control speed
                    time.sleep(0.02)  # 50Hz execution

                    # Print progress
                    if i % 50 == 0:
                        current_pos = get_ee_pose(model, data)[:3, 3]
                        print(f"  Step {i}/{len(interpolated_path)}: pos={current_pos}")

                print("✓ Execution completed!")

                # Stay at goal
                print("\nStaying at goal position (close viewer to exit)...")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n⏹ Stopped by user")
        """

    else:
        print(f"✗ Planning failed!")
        print(f"  - Reason: {info.get('error', 'Unknown')}")
        if 'planning_time' in info:
            print(f"  - Time spent: {info['planning_time']:.3f}s")
        if 'num_ik_failures' in info:
            print(f"  - IK failures: {info['num_ik_failures']}")


def example_with_obstacles():
    """
    Example with obstacles in the environment
    """
    print("\n" + "="*70)
    print("CARTESIAN PLANNING WITH OBSTACLES")
    print("="*70)

    print("\n💡 To add obstacles:")
    print("  1. Add obstacle geoms to your MuJoCo XML file")
    print("  2. The collision checker will automatically detect them")
    print("  3. RRT-Connect will plan around obstacles in Cartesian space")
    print("  4. IK ensures feasible joint configurations")


def print_workflow_summary():
    """Print a summary of the complete workflow"""
    print("\n" + "="*70)
    print("📋 WORKFLOW SUMMARY")
    print("="*70)

    workflow = """
Complete Cartesian Space RRT-Connect Workflow:

1️⃣  RRT-Connect in Cartesian Space
   └─→ Search for path in end-effector space (position + orientation)
   └─→ Faster than joint-space planning for task-space goals
   └─→ Bidirectional search connects start and goal trees

2️⃣  Inverse Kinematics (IK)
   └─→ Convert each Cartesian waypoint to joint configuration
   └─→ Uses Pinocchio IK solver with Jacobian-based optimization
   └─→ Collision checking for each IK solution

3️⃣  Path Interpolation
   └─→ Densify waypoints for smooth motion
   └─→ Linear interpolation in joint space
   └─→ Ensures continuous motion without large jumps

4️⃣  Robot Execution
   └─→ Send interpolated joint configurations to robot
   └─→ Control at desired frequency (e.g., 50Hz)
   └─→ Monitor execution and handle errors

Key Advantages:
✓ Natural for task-space goals (e.g., "move to position X")
✓ Handles obstacles in workspace
✓ Faster planning than high-DOF joint space
✓ Smooth execution with interpolation
"""

    print(workflow)


if __name__ == "__main__":
    print_workflow_summary()

    try:
        # Run main example
        example_cartesian_planning()

        # Additional examples
        example_with_obstacles()

        print("\n" + "="*70)
        print("✓ All examples completed!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
