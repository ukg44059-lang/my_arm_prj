"""
ROS2 综合传感器数据发布器
发布MuJoCo仿真中的所有传感器数据，包括：
- 关节状态（位置、速度、力矩）
- 末端执行器位姿
- 腕部摄像头RGB/深度图像
- 外部摄像头数据
"""

from typing import Optional, Dict, Any
import numpy as np
import threading
import time
import cv2

ROS2_AVAILABLE = False
_import_error = ""
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from sensor_msgs.msg import JointState, Image, CompressedImage, CameraInfo
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from tf2_ros import TransformBroadcaster
    from cv_bridge import CvBridge
    import builtin_interfaces.msg
    ROS2_AVAILABLE = True
except ImportError as e:
    _import_error = str(e)
    ROS2_AVAILABLE = False


def ros2_available() -> bool:
    return ROS2_AVAILABLE


class MuJoCoSensorPublisher:
    """
    MuJoCo仿真传感器数据综合发布器
    
    发布话题：
    - /joint_states: 关节状态
    - /ee_pose: 末端执行器位姿
    - /camera/ee/rgb/image_raw: 腕部RGB图像
    - /camera/ee/depth/image_raw: 腕部深度图像
    - /camera/external/rgb/image_raw: 外部RGB图像
    - /tf: 坐标变换（可选）
    """

    def __init__(
        self,
        node_name: str = "mujoco_sensor_publisher",
        joint_names: Optional[list] = None,
        publish_tf: bool = False,
        publish_compressed: bool = True,
        image_quality: int = 90
    ):
        if not ROS2_AVAILABLE:
            raise ImportError(
                f"ROS2 not available: {_import_error}\n"
                "Please source your ROS2 environment and install required packages:\n"
                "  sudo apt install ros-humble-sensor-msgs ros-humble-geometry-msgs\n"
                "  sudo apt install ros-humble-cv-bridge ros-humble-tf2-ros\n"
            )

        # Initialize ROS2 if not already done
        if not rclpy.ok():
            rclpy.init()

        self.node = Node(node_name)
        self.cv_bridge = CvBridge()
        
        # Joint configuration
        self.joint_names = joint_names or [f"joint{i+1}" for i in range(6)]
        self.joint_names.append("gripper")  # Add gripper joint
        
        # Publishing flags
        self.publish_tf = publish_tf
        self.publish_compressed = publish_compressed
        self.image_quality = image_quality
        
        # Initialize publishers
        self._init_publishers()
        
        # Transform broadcaster
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self.node)
        
        # Threading control
        self.running = False
        self.spin_thread = None
        self.publish_rate = 30.0  # Hz
        
        # Data buffers
        self.latest_joint_data = None
        self.latest_ee_pose = None
        self.latest_images = {}
        self.data_lock = threading.Lock()
        
        print(f"✓ MuJoCo Sensor Publisher initialized")
        print(f"  - Node: {node_name}")
        print(f"  - Joint names: {self.joint_names}")
        print(f"  - Publish rate: {self.publish_rate} Hz")
        print(f"  - Compressed images: {self.publish_compressed}")
        if self.publish_tf:
            print(f"  - TF broadcasting enabled")

    def _init_publishers(self):
        """初始化所有发布器"""
        
        # QoS配置
        sensor_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        reliable_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        # 关节状态发布器
        self.joint_state_pub = self.node.create_publisher(
            JointState, '/joint_states', reliable_qos
        )
        
        # 末端执行器位姿发布器
        self.ee_pose_pub = self.node.create_publisher(
            PoseStamped, '/ee_pose', reliable_qos
        )
        
        # 图像发布器
        self.image_publishers = {}
        self.compressed_image_publishers = {}
        
        # 腕部摄像头
        self.image_publishers['ee_rgb'] = self.node.create_publisher(
            Image, '/camera/ee/rgb/image_raw', sensor_qos
        )
        self.image_publishers['ee_depth'] = self.node.create_publisher(
            Image, '/camera/ee/depth/image_raw', sensor_qos
        )
        
        # 外部摄像头
        self.image_publishers['external_rgb'] = self.node.create_publisher(
            Image, '/camera/external/rgb/image_raw', sensor_qos
        )
        
        # 压缩图像发布器（可选）
        if self.publish_compressed:
            self.compressed_image_publishers['ee_rgb'] = self.node.create_publisher(
                CompressedImage, '/camera/ee/rgb/image_raw/compressed', sensor_qos
            )
            self.compressed_image_publishers['ee_depth'] = self.node.create_publisher(
                CompressedImage, '/camera/ee/depth/image_raw/compressed', sensor_qos
            )
            self.compressed_image_publishers['external_rgb'] = self.node.create_publisher(
                CompressedImage, '/camera/external/rgb/image_raw/compressed', sensor_qos
            )

    def update_joint_state(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        efforts: Optional[np.ndarray] = None,
        gripper_position: float = 0.0,
        gripper_effort: float = 0.0
    ):
        """
        更新关节状态数据
        
        Args:
            positions: 关节位置 (6,) rad
            velocities: 关节速度 (6,) rad/s，可选
            efforts: 关节力矩 (6,) Nm，可选
            gripper_position: 夹爪位置 [0-1]
            gripper_effort: 夹爪力矩
        """
        with self.data_lock:
            # 组合机械臂关节和夹爪
            full_positions = np.append(positions, gripper_position)
            full_velocities = np.append(velocities if velocities is not None else np.zeros(6), 0.0)
            full_efforts = np.append(efforts if efforts is not None else np.zeros(6), gripper_effort)
            
            self.latest_joint_data = {
                'positions': full_positions,
                'velocities': full_velocities,
                'efforts': full_efforts,
                'timestamp': time.time()
            }

    def update_ee_pose(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        frame_id: str = "base_link"
    ):
        """
        更新末端执行器位姿数据
        
        Args:
            position: 位置 [x, y, z]
            quaternion: 四元数 [w, x, y, z]
            frame_id: 坐标系名称
        """
        with self.data_lock:
            self.latest_ee_pose = {
                'position': position.copy(),
                'quaternion': quaternion.copy(),
                'frame_id': frame_id,
                'timestamp': time.time()
            }

    def update_camera_image(
        self,
        camera_name: str,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        frame_id: Optional[str] = None
    ):
        """
        更新摄像头图像数据
        
        Args:
            camera_name: 摄像头名称 ('ee' 或 'external')
            rgb_image: RGB图像 (H, W, 3) uint8
            depth_image: 深度图像 (H, W) float32，单位：米
            frame_id: 图像坐标系
        """
        with self.data_lock:
            if camera_name not in self.latest_images:
                self.latest_images[camera_name] = {}
            
            if rgb_image is not None:
                self.latest_images[camera_name]['rgb'] = {
                    'data': rgb_image.copy(),
                    'frame_id': frame_id or f"{camera_name}_rgb_optical_frame",
                    'timestamp': time.time()
                }
            
            if depth_image is not None:
                self.latest_images[camera_name]['depth'] = {
                    'data': depth_image.copy(),
                    'frame_id': frame_id or f"{camera_name}_depth_optical_frame", 
                    'timestamp': time.time()
                }

    def _publish_joint_state(self):
        """发布关节状态"""
        with self.data_lock:
            if self.latest_joint_data is None:
                return
            
            joint_data = self.latest_joint_data.copy()
        
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        
        msg.name = self.joint_names
        msg.position = joint_data['positions'].tolist()
        msg.velocity = joint_data['velocities'].tolist()
        msg.effort = joint_data['efforts'].tolist()
        
        self.joint_state_pub.publish(msg)

    def _publish_ee_pose(self):
        """发布末端执行器位姿"""
        with self.data_lock:
            if self.latest_ee_pose is None:
                return
            
            ee_data = self.latest_ee_pose.copy()
        
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = ee_data['frame_id']
        
        msg.pose.position.x = float(ee_data['position'][0])
        msg.pose.position.y = float(ee_data['position'][1])
        msg.pose.position.z = float(ee_data['position'][2])
        
        msg.pose.orientation.w = float(ee_data['quaternion'][0])
        msg.pose.orientation.x = float(ee_data['quaternion'][1])
        msg.pose.orientation.y = float(ee_data['quaternion'][2])
        msg.pose.orientation.z = float(ee_data['quaternion'][3])
        
        self.ee_pose_pub.publish(msg)
        
        # 发布TF变换（可选）
        if self.publish_tf:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = msg.header.stamp
            tf_msg.header.frame_id = ee_data['frame_id']
            tf_msg.child_frame_id = "ee_link"
            
            tf_msg.transform.translation.x = msg.pose.position.x
            tf_msg.transform.translation.y = msg.pose.position.y
            tf_msg.transform.translation.z = msg.pose.position.z
            
            tf_msg.transform.rotation = msg.pose.orientation
            
            self.tf_broadcaster.sendTransform(tf_msg)

    def _publish_images(self):
        """发布所有摄像头图像"""
        with self.data_lock:
            images_data = {}
            for camera_name, camera_data in self.latest_images.items():
                images_data[camera_name] = {}
                for image_type, image_info in camera_data.items():
                    images_data[camera_name][image_type] = image_info.copy()
        
        for camera_name, camera_data in images_data.items():
            for image_type, image_info in camera_data.items():
                self._publish_single_image(camera_name, image_type, image_info)

    def _publish_single_image(self, camera_name: str, image_type: str, image_info: Dict[str, Any]):
        """发布单张图像"""
        try:
            image_data = image_info['data']
            frame_id = image_info['frame_id']
            
            # 创建ROS图像消息
            if image_type == 'rgb':
                # RGB图像
                img_msg = self.cv_bridge.cv2_to_imgmsg(image_data, encoding="rgb8")
                topic_key = f"{camera_name}_rgb"
            elif image_type == 'depth':
                # 深度图像：转换为16位毫米单位
                depth_mm = (image_data * 1000).astype(np.uint16)
                img_msg = self.cv_bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
                topic_key = f"{camera_name}_depth"
            else:
                return
            
            img_msg.header.stamp = self.node.get_clock().now().to_msg()
            img_msg.header.frame_id = frame_id
            
            # 发布原始图像
            if topic_key in self.image_publishers:
                self.image_publishers[topic_key].publish(img_msg)
            
            # 发布压缩图像（可选）
            if self.publish_compressed and topic_key in self.compressed_image_publishers:
                compressed_msg = CompressedImage()
                compressed_msg.header = img_msg.header
                
                if image_type == 'rgb':
                    # 压缩RGB图像为JPEG
                    compressed_msg.format = "jpeg"
                    encode_param = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
                    _, compressed_data = cv2.imencode('.jpg', cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR), encode_param)
                else:
                    # 压缩深度图像为PNG
                    compressed_msg.format = "png"
                    _, compressed_data = cv2.imencode('.png', depth_mm)
                
                compressed_msg.data = compressed_data.tobytes()
                self.compressed_image_publishers[topic_key].publish(compressed_msg)
                
        except Exception as e:
            print(f"❌ Failed to publish {camera_name} {image_type} image: {e}")

    def start_publishing(self):
        """开始发布传感器数据"""
        if self.running:
            print("⚠ Sensor publisher already running")
            return
        
        self.running = True
        self.spin_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.spin_thread.start()
        print(f"✓ Sensor publisher started at {self.publish_rate} Hz")

    def stop_publishing(self):
        """停止发布传感器数据"""
        if not self.running:
            return
        
        self.running = False
        if self.spin_thread:
            self.spin_thread.join(timeout=2.0)
            self.spin_thread = None
        print("✓ Sensor publisher stopped")

    def _publish_loop(self):
        """主发布循环"""
        dt = 1.0 / self.publish_rate
        
        while self.running and rclpy.ok():
            loop_start = time.time()
            
            try:
                # 发布所有传感器数据
                self._publish_joint_state()
                self._publish_ee_pose()
                self._publish_images()
                
                # 处理ROS2回调
                rclpy.spin_once(self.node, timeout_sec=0.001)
                
                # 控制发布频率
                loop_duration = time.time() - loop_start
                sleep_time = max(0, dt - loop_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"❌ Publish loop error: {e}")
                break

    def shutdown(self):
        """清理关闭"""
        self.stop_publishing()
        
        # 清理发布器
        for pub in self.image_publishers.values():
            try:
                pub.destroy()
            except:
                pass
        
        for pub in self.compressed_image_publishers.values():
            try:
                pub.destroy()
            except:
                pass
        
        try:
            self.joint_state_pub.destroy()
            self.ee_pose_pub.destroy()
        except:
            pass
        
        try:
            self.node.destroy_node()
        except:
            pass
        
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except:
                pass
        
        print("✓ MuJoCo Sensor Publisher shutdown complete")

    def get_status(self) -> Dict[str, Any]:
        """获取发布器状态"""
        with self.data_lock:
            return {
                "running": self.running,
                "publish_rate": self.publish_rate,
                "joint_data_available": self.latest_joint_data is not None,
                "ee_pose_available": self.latest_ee_pose is not None,
                "cameras_available": list(self.latest_images.keys()),
                "publish_compressed": self.publish_compressed,
                "publish_tf": self.publish_tf
            }

    def set_publish_rate(self, rate: float):
        """设置发布频率"""
        self.publish_rate = max(1.0, min(100.0, rate))  # 限制在1-100Hz
        print(f"✓ Publish rate set to {self.publish_rate} Hz")