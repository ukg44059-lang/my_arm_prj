#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Publish a Cartesian endpoint target + arm enable for simulation_interface.py.

Usage idea:
1) Start simulator/controller: test/sim_interface/simulation_interface.py
2) Run this script to:
   - continuously publish /arm_enable = True
   - publish one or more PoseStamped targets to /cartesian_target
"""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


class CartesianGoalSender(Node):
	def __init__(self, args: argparse.Namespace):
		super().__init__("cartesian_goal_sender")
		self.args = args

		self.pose_pub = self.create_publisher(PoseStamped, "/cartesian_target", 10)
		self.enable_pub = self.create_publisher(Bool, "/arm_enable", 10)

		self._enable_timer = None

	def wait_subscribers(self, timeout_sec: float = 6.0) -> bool:
		t0 = time.time()
		while rclpy.ok() and (time.time() - t0 < timeout_sec):
			pose_n = self.pose_pub.get_subscription_count()
			en_n = self.enable_pub.get_subscription_count()
			self.get_logger().info(
				f"Subscribers -> /cartesian_target:{pose_n}, /arm_enable:{en_n}"
			)
			if pose_n > 0:
				return True
			rclpy.spin_once(self, timeout_sec=0.2)
		return False

	def start_enable_keepalive(self, hz: float = 10.0) -> None:
		period = 1.0 / max(1.0, hz)

		def _tick() -> None:
			msg = Bool()
			msg.data = True
			self.enable_pub.publish(msg)

		self._enable_timer = self.create_timer(period, _tick)
		self.get_logger().info(f"/arm_enable keepalive started ({hz:.1f} Hz)")

	def stop_enable_keepalive(self) -> None:
		if self._enable_timer is not None:
			try:
				self._enable_timer.cancel()
			except Exception:
				pass

	def publish_target_pose(self) -> None:
		msg = PoseStamped()
		msg.header.stamp = self.get_clock().now().to_msg()
		msg.header.frame_id = self.args.frame

		msg.pose.position.x = float(self.args.x)
		msg.pose.position.y = float(self.args.y)
		msg.pose.position.z = float(self.args.z)
		msg.pose.orientation.w = float(self.args.qw)
		msg.pose.orientation.x = float(self.args.qx)
		msg.pose.orientation.y = float(self.args.qy)
		msg.pose.orientation.z = float(self.args.qz)

		for _ in range(max(1, self.args.repeats)):
			self.pose_pub.publish(msg)
			rclpy.spin_once(self, timeout_sec=0.02)
			time.sleep(max(0.01, self.args.repeat_dt))

		self.get_logger().info(
			f"Published /cartesian_target frame={self.args.frame}, "
			f"pos=({self.args.x:.3f}, {self.args.y:.3f}, {self.args.z:.3f}), "
			f"quat=[{self.args.qw:.3f}, {self.args.qx:.3f}, {self.args.qy:.3f}, {self.args.qz:.3f}]"
		)

	def run(self) -> int:
		ok = self.wait_subscribers(timeout_sec=self.args.wait_sub_sec)
		if not ok:
			self.get_logger().error("No subscriber on /cartesian_target. Start test/sim_interface/simulation_interface.py first.")
			return 1

		self.start_enable_keepalive(hz=self.args.enable_hz)
		self.get_logger().info("Arm enable asserted, publishing target...")
		time.sleep(0.2)

		self.publish_target_pose()

		hold_sec = max(0.0, self.args.hold_sec)
		t0 = time.time()
		while rclpy.ok() and (time.time() - t0 < hold_sec):
			rclpy.spin_once(self, timeout_sec=0.05)

		self.stop_enable_keepalive()
		self.get_logger().info("Done.")
		return 0


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Send /arm_enable and one Cartesian target pose")

	# Endpoint target (default uses your recent test point)
	p.add_argument("--x", type=float, default=0.0)
	p.add_argument("--y", type=float, default=-0.0)
	p.add_argument("--z", type=float, default=0.898)

	p.add_argument("--qw", type=float, default=1)
	p.add_argument("--qx", type=float, default=0.0)
	p.add_argument("--qy", type=float, default=0.0)
	p.add_argument("--qz", type=float, default=0.0)

	p.add_argument("--frame", type=str, default="base", help="Pose frame: base/base_link/world")

	p.add_argument("--repeats", type=int, default=5, help="How many times to publish the same target")
	p.add_argument("--repeat-dt", type=float, default=0.05, help="Interval between repeated target publishes")

	p.add_argument("--enable-hz", type=float, default=10.0, help="/arm_enable keepalive frequency")
	p.add_argument("--wait-sub-sec", type=float, default=6.0, help="Timeout waiting /cartesian_target subscriber")
	p.add_argument("--hold-sec", type=float, default=5.0, help="How long to keep /arm_enable asserted after publishing")

	return p


def main() -> None:
	args = build_parser().parse_args()

	rclpy.init()
	node = CartesianGoalSender(args)

	code = 0
	try:
		code = node.run()
	finally:
		node.destroy_node()
		rclpy.shutdown()

	raise SystemExit(code)


if __name__ == "__main__":
	main()

