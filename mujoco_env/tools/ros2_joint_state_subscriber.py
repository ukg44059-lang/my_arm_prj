"""
ROS2 JointState subscriber for receiving joint targets.
ASCII-only; optional ROS2 dependency.
"""
from typing import Optional, Callable, List
import numpy as np
import threading
import time

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


class JointStateSubscriberROS2:
    """Lightweight subscriber for /joint_states to receive joint targets."""

    def __init__(
        self,
        topic: str = "/joint_target_states",
        node_name: str = "mujoco_joint_state_subscriber",
        callback: Optional[Callable[[np.ndarray], None]] = None
    ) -> None:
        if not ROS2_AVAILABLE:
            raise ImportError(
                f"ROS2 not available: {_import_error}\n"
                "Please source your ROS2 environment, e.g.\n"
                "  source /opt/ros/humble/setup.bash\n"
            )

        self.callback = callback
        self.latest_joint_positions = None
        self.latest_joint_velocities = None
        self.last_update_time = None
        self.lock = threading.Lock()
        
        # 为每个订阅器创建独立的ROS2上下文
        self.context = rclpy.Context()
        rclpy.init(context=self.context)

        self.node = Node(node_name, context=self.context)
        self.subscription = self.node.create_subscription(
            JointState,
            topic,
            self._joint_state_callback,
            100  # 增加队列深度防止消息丢失
        )
        
        # 创建独立的执行器
        self.executor = rclpy.executors.SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        
        self.running = False
        self.spin_thread = None
        
        print(f"✓ ROS2 Joint State Subscriber initialized")
        print(f"  - Topic: {topic}")
        print(f"  - Node: {node_name}")

    def _joint_state_callback(self, msg):
        """Internal callback for joint state messages."""
        with self.lock:
            if len(msg.position) >= 7:  # Ensure we have at least 7 joints (including gripper)
                self.latest_joint_positions = np.array(msg.position[:7])
                if len(msg.velocity) >= 7:
                    self.latest_joint_velocities = np.array(msg.velocity[:7])
                else:
                    self.latest_joint_velocities = np.zeros(7)
            elif len(msg.position) >= 6:  # Fallback to 6 joints only
                self.latest_joint_positions = np.array(msg.position[:6])
                if len(msg.velocity) >= 6:
                    self.latest_joint_velocities = np.array(msg.velocity[:6])
                else:
                    self.latest_joint_velocities = np.zeros(6)
            else:
                return  # Not enough joints, skip this message
                
            self.last_update_time = time.time()
            
            # Call external callback if provided
            if self.callback:
                self.callback(self.latest_joint_positions.copy())

    def get_latest_joint_positions(self) -> Optional[np.ndarray]:
        """Get the latest received joint positions."""
        with self.lock:
            if self.latest_joint_positions is not None:
                return self.latest_joint_positions.copy()
            return None

    def get_latest_joint_velocities(self) -> Optional[np.ndarray]:
        """Get the latest received joint velocities."""
        with self.lock:
            if self.latest_joint_velocities is not None:
                return self.latest_joint_velocities.copy()
            return None

    def get_time_since_last_update(self) -> Optional[float]:
        """Get seconds since last message received."""
        with self.lock:
            if self.last_update_time is not None:
                return time.time() - self.last_update_time
            return None

    def has_received_data(self) -> bool:
        """Check if any data has been received."""
        with self.lock:
            return self.latest_joint_positions is not None

    def start_spinning(self):
        """Start spinning ROS2 node in separate thread."""
        if self.running:
            print("⚠ ROS2 subscriber already running")
            return
            
        self.running = True
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

    def stop_spinning(self):
        """Stop spinning ROS2 node."""
        if not self.running:
            return
            
        self.running = False
        if self.spin_thread:
            self.spin_thread.join(timeout=1.0)
            self.spin_thread = None

    def _spin_loop(self):
        """Main spin loop for ROS2 node."""
        while self.running:
            try:
                # 使用独立的执行器避免冲突
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                if self.running:  # 只在运行时报错，避免关闭时的无关错误
                    print(f"❌ ROS2 spin error: {e}")
                break

    def shutdown(self):
        """Clean shutdown of subscriber."""
        self.stop_spinning()
        
        try:
            if hasattr(self, 'executor'):
                self.executor.remove_node(self.node)
                self.executor.shutdown()
        except Exception:
            pass
        
        try:
            self.subscription.destroy()
        except Exception:
            pass
            
        try:
            self.node.destroy_node()
        except Exception:
            pass
        
        try:
            if hasattr(self, 'context'):
                rclpy.shutdown(context=self.context)
        except Exception:
            pass
                
        print("✓ ROS2 Joint State Subscriber shutdown complete")

    def set_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback function for new joint state messages."""
        self.callback = callback

    def get_status(self) -> dict:
        """Get subscriber status information."""
        with self.lock:
            return {
                "running": self.running,
                "has_data": self.has_received_data(),
                "time_since_update": self.get_time_since_last_update(),
                "latest_positions": self.latest_joint_positions.tolist() if self.latest_joint_positions is not None else None
            }