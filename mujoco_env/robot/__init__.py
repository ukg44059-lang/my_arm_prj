"""
Robot Control Package
机器人控制包

This package provides the Robot class for controlling MuJoCo robot simulations.
该包提供用于控制 MuJoCo 机器人仿真的 Robot 类。

Usage:
    from robot import Robot

    # Initialize robot
    robot = Robot(model_path="path/to/model.xml")

    # Set control mode
    robot.set_control_mode("zero_gravity")

    # In simulation loop:
    robot.update_joint_state()
    targets = robot.compute_joint_targets(sim_time)
    robot.apply_joint_control(targets)
    robot.update_obstacles()
    robot.step()

    # Get robot state
    ee_pos = robot.get_ee_position()
    ee_quat = robot.get_ee_orientation()
"""

from .robot import Robot

__all__ = ['Robot']
