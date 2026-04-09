"""
ROS2 JointState publisher helper for MuJoCo robot.
ASCII-only; optional ROS2 dependency.
"""
from typing import Iterable, Optional, Sequence
import numpy as np

ROS2_AVAILABLE = False
_import_error = ""
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError as e:
    _import_error = str(e)
    ROS2_AVAILABLE = False


def ros2_available() -> bool:
    return ROS2_AVAILABLE


class JointStatePublisherROS2:
    """Lightweight publisher for /joint_states."""

    def __init__(
        self,
        topic: str = "/joint_states",
        joint_names: Optional[Sequence[str]] = None,
        node_name: str = "mujoco_joint_state_publisher",
    ) -> None:
        if not ROS2_AVAILABLE:
            raise ImportError(
                f"ROS2 not available: {_import_error}\n"
                "Please source your ROS2 environment, e.g.\n"
                "  source /opt/ros/humble/setup.bash\n"
            )

        if not rclpy.ok():
            rclpy.init()

        self.joint_names = list(joint_names) if joint_names is not None else None
        self.node = Node(node_name)
        self.publisher = self.node.create_publisher(JointState, topic, 10)

    def publish(
        self,
        positions: Iterable[float],
        velocities: Optional[Iterable[float]] = None,
        efforts: Optional[Iterable[float]] = None,
        stamp_sec: Optional[float] = None,
    ) -> None:
        pos = np.array(list(positions), dtype=float)
        vel = np.array(list(velocities), dtype=float) if velocities is not None else None
        eff = np.array(list(efforts), dtype=float) if efforts is not None else None

        n = len(pos)
        names = self.joint_names or [f"joint{i+1}" for i in range(n)]
        if len(names) != n:
            names = list(names[:n])

        msg = JointState()
        msg.name = names
        msg.position = pos.tolist()
        msg.velocity = vel.tolist() if vel is not None else []
        msg.effort = eff.tolist() if eff is not None else []

        if stamp_sec is None:
            msg.header.stamp = self.node.get_clock().now().to_msg()
        else:
            msg.header.stamp = rclpy.time.Time(seconds=stamp_sec).to_msg()

        self.publisher.publish(msg)

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def shutdown(self) -> None:
        try:
            self.publisher.destroy()
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
