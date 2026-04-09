"""
ROS2 JointState publisher for MuJoCo robot
将 MuJoCo 关节状态发布到 ROS2 /joint_states 话题

依赖:
  source /opt/ros/humble/setup.bash  # 根据实际 ROS2 版本调整
"""

from typing import Iterable, Optional, Sequence
import numpy as np

# ROS2 imports (optional)
ROS2_AVAILABLE = False
import_error_msg = ""
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError as e:
    import_error_msg = str(e)
    ROS2_AVAILABLE = False


class JointStateROS2Publisher:
    """轻量级 JointState 发布器，可复用在仿真循环中"""

    def __init__(
        self,
        node_name: str = "mujoco_joint_state_publisher",
        topic_name: str = "/joint_states",
        joint_names: Optional[Sequence[str]] = None,
    ) -> None:
        if not ROS2_AVAILABLE:
            err = (
                f"ROS2 not available: {import_error_msg}\n"
                "Please source your ROS2 environment first, e.g.\n"
                "  source /opt/ros/humble/setup.bash\n"
            )
            raise ImportError(err)

        if not rclpy.ok():
            rclpy.init()

        self.joint_names = list(joint_names) if joint_names is not None else None
        self.node = Node(node_name)
        self.publisher = self.node.create_publisher(JointState, topic_name, 10)

    def publish(
        self,
        positions: Iterable[float],
        velocities: Optional[Iterable[float]] = None,
        efforts: Optional[Iterable[float]] = None,
        stamp_sec: Optional[float] = None,
    ) -> None:
        """发布 JointState 消息。长度自动截断/填充到关节数量。"""
        # Convert to numpy for safe slicing
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
        """处理一次 ROS2 回调（可选）。"""
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


def is_ros2_available() -> bool:
    return ROS2_AVAILABLE
