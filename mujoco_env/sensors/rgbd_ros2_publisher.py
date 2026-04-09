"""
ROS2 Image Publisher for MuJoCo RGBD Sensor
将MuJoCo RGBD传感器的图像通过ROS2 topic发布

依赖:
需要source ROS2环境后使用:
  source /opt/ros/humble/setup.bash  # 或其他ROS2版本
  python3 your_script.py
"""

import numpy as np

# Try to import ROS2 packages
ROS2_AVAILABLE = False
import_error_msg = ""

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError as e:
    import_error_msg = str(e)
    ROS2_AVAILABLE = False


class RGBD_ROS2_Publisher:
    """
    ROS2 图像发布器（使用 cv_bridge）

    发布的topics:
    - /{camera_name}/rgb/image_raw: RGB图像
    - /{camera_name}/depth/image_raw: 深度图像
    - /{camera_name}/camera_info: 相机信息
    """

    def __init__(self, camera_name: str = "mujoco_camera", node_name: str = "mujoco_rgbd_publisher", verbose: bool = False):
        """
        初始化ROS2发布器

        Args:
            camera_name: 相机名称，用于topic命名
            node_name: ROS2节点名称
        """
        if not ROS2_AVAILABLE:
            error_msg = f"ROS2 not available: {import_error_msg}\n"
            error_msg += "Please install required packages:\n"
            error_msg += "  pip install rclpy sensor_msgs cv_bridge\n"
            error_msg += "And source ROS2 environment:\n"
            error_msg += "  source /opt/ros/humble/setup.bash"
            raise ImportError(error_msg)

        self.camera_name = camera_name
        self.verbose = verbose

        # 初始化ROS2
        if not rclpy.ok():
            rclpy.init()

        # 创建节点
        self.node = Node(node_name)

        # 创建 CvBridge
        self.bridge = CvBridge()

        # 创建发布器
        self.rgb_pub = self.node.create_publisher(
            Image,
            f'/{camera_name}/rgb/image_raw',
            10
        )

        self.depth_pub = self.node.create_publisher(
            Image,
            f'/{camera_name}/depth/image_raw',
            10
        )

        self.camera_info_pub = self.node.create_publisher(
            CameraInfo,
            f'/{camera_name}/camera_info',
            10
        )

        # 帧计数器
        self.frame_id = 0

        if self.verbose:
            print(f"✓ ROS2 RGBD Publisher initialized (using cv_bridge)")
            print(f"  - Node: {node_name}")
            print(f"  - Camera: {camera_name}")
            print(f"  - Topics:")
            print(f"    • /{camera_name}/rgb/image_raw")
            print(f"    • /{camera_name}/depth/image_raw")
            print(f"    • /{camera_name}/camera_info")

    def publish_rgbd(self, rgb: np.ndarray, depth: np.ndarray,
                     intrinsics: np.ndarray = None, timestamp_sec: float = None):
        """
        发布RGB-D图像

        Args:
            rgb: RGB图像 (H, W, 3), uint8, RGB格式
            depth: 深度图像 (H, W), float32, 单位米
            intrinsics: 3x3相机内参矩阵 或 (fx, fy, cx, cy) 元组
            timestamp_sec: 时间戳（秒），如果为None则使用当前时间
        """
        if not ROS2_AVAILABLE:
            return

        try:
            # 创建时间戳
            stamp = self.node.get_clock().now().to_msg()

            # 发布 RGB 图像
            if rgb is not None:
                rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
                rgb_msg.header.stamp = stamp
                rgb_msg.header.frame_id = f'{self.camera_name}_optical_frame'
                self.rgb_pub.publish(rgb_msg)

            # 发布深度图像
            if depth is not None:
                # 转换深度图为 32FC1 格式
                depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
                depth_msg.header.stamp = stamp
                depth_msg.header.frame_id = f'{self.camera_name}_optical_frame'
                self.depth_pub.publish(depth_msg)

            # 发布相机内参
            if intrinsics is not None:
                camera_info_msg = CameraInfo()
                camera_info_msg.header.stamp = stamp
                camera_info_msg.header.frame_id = f'{self.camera_name}_optical_frame'

                # 设置图像尺寸
                if rgb is not None:
                    camera_info_msg.height = rgb.shape[0]
                    camera_info_msg.width = rgb.shape[1]
                elif depth is not None:
                    camera_info_msg.height = depth.shape[0]
                    camera_info_msg.width = depth.shape[1]

                # 解析内参矩阵
                if isinstance(intrinsics, np.ndarray) and intrinsics.shape == (3, 3):
                    # 3x3 矩阵格式
                    fx = intrinsics[0, 0]
                    fy = intrinsics[1, 1]
                    cx = intrinsics[0, 2]
                    cy = intrinsics[1, 2]
                else:
                    # 假设是 (fx, fy, cx, cy) 元组
                    fx, fy, cx, cy = intrinsics

                # 设置畸变模型
                camera_info_msg.distortion_model = "plumb_bob"
                camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # 无畸变

                # 设置内参矩阵 K
                camera_info_msg.k = [
                    fx, 0.0, cx,
                    0.0, fy, cy,
                    0.0, 0.0, 1.0
                ]

                # 设置投影矩阵 P
                camera_info_msg.p = [
                    fx, 0.0, cx, 0.0,
                    0.0, fy, cy, 0.0,
                    0.0, 0.0, 1.0, 0.0
                ]

                # 矫正矩阵 R (单位矩阵)
                camera_info_msg.r = [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0
                ]

                self.camera_info_pub.publish(camera_info_msg)

            self.frame_id += 1

        except Exception as e:
            print(f"❌ 发布 RGBD 数据失败: {e}")
            import traceback
            traceback.print_exc()

    def spin_once(self, timeout_sec: float = 0.0):
        """
        处理ROS2回调一次（非阻塞）

        Args:
            timeout_sec: 超时时间（秒）
        """
        if not ROS2_AVAILABLE:
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=timeout_sec)
        except Exception as e:
            print(f"❌ spin_once 失败: {e}")

    def shutdown(self):
        """关闭ROS2发布器"""
        if not ROS2_AVAILABLE:
            return
        try:
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            if self.verbose:
                print("✓ ROS2 RGBD Publisher shutdown")
        except Exception as e:
            print(f"⚠ Error during shutdown: {e}")

    def destroy(self):
        """销毁节点（别名方法）"""
        self.shutdown()


class RGBD_Sensor_With_ROS2:
    """
    集成ROS2发布功能的RGBD传感器包装器

    包装MuJoCo_RGBD_Sensor，添加ROS2发布功能
    """

    def __init__(self, rgbd_sensor, camera_name: str = None,
                 enable_ros2: bool = True):
        """
        初始化

        Args:
            rgbd_sensor: MuJoCo_RGBD_Sensor实例
            camera_name: 相机名称（用于ROS2 topic）
            enable_ros2: 是否启用ROS2发布
        """
        self.sensor = rgbd_sensor
        self.enable_ros2 = enable_ros2 and ROS2_AVAILABLE

        if camera_name is None:
            camera_name = rgbd_sensor.cam_name

        # 创建ROS2发布器
        if self.enable_ros2:
            try:
                self.ros2_pub = RGBD_ROS2_Publisher(
                    camera_name=camera_name,
                    node_name=f"{camera_name}_publisher"
                )
            except Exception as e:
                print(f"⚠ Failed to initialize ROS2 publisher: {e}")
                self.enable_ros2 = False
        else:
            self.ros2_pub = None
            if not ROS2_AVAILABLE:
                print("ℹ ROS2 not available, running without ROS2 publishing")

    def render_and_publish(self, publish_to_ros2: bool = True):
        """
        渲染RGBD图像并发布到ROS2

        Args:
            publish_to_ros2: 是否发布到ROS2

        Returns:
            rgb, depth: RGB和深度图像
        """
        # 渲染图像
        rgb, depth = self.sensor.render_rgbd()

        # 发布到ROS2
        if self.enable_ros2 and publish_to_ros2 and self.ros2_pub is not None:
            self.ros2_pub.publish_rgbd(
                rgb=rgb,
                depth=depth,
                intrinsics=self.sensor.intr,
                timestamp_sec=None
            )
            # 处理ROS2回调
            self.ros2_pub.spin_once(timeout_sec=0.0)

        return rgb, depth

    def publish_images(self, rgb: np.ndarray, depth: np.ndarray):
        """
        发布已渲染的图像到ROS2（避免重复渲染）

        Args:
            rgb: 已渲染的RGB图像
            depth: 已渲染的深度图像
        """
        if self.enable_ros2 and self.ros2_pub is not None:
            self.ros2_pub.publish_rgbd(
                rgb=rgb,
                depth=depth,
                intrinsics=self.sensor.intr,
                timestamp_sec=None
            )
            # 处理ROS2回调
            self.ros2_pub.spin_once(timeout_sec=0.0)

    def display(self, rgb: np.ndarray = None, depth: np.ndarray = None):
        """
        显示RGB和深度图像

        Args:
            rgb: RGB图像（如果为None则重新渲染）
            depth: 深度图像（如果为None则重新渲染）
        """
        if rgb is None or depth is None:
            rgb, depth = self.sensor.render_rgbd()

        self.sensor.display_rgb(rgb)
        self.sensor.display_depth(depth)

    def shutdown(self):
        """关闭ROS2发布器"""
        if self.enable_ros2 and self.ros2_pub is not None:
            self.ros2_pub.shutdown()


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("ROS2 RGBD Publisher Example")
    print("="*70)

    if not ROS2_AVAILABLE:
        print("\n✗ ROS2 not available!")
        print("Install with: pip install rclpy sensor_msgs cv_bridge")
    else:
        print("\n✓ ROS2 is available (using cv_bridge)")
        print("\nTo test this publisher:")
        print("1. Run this with a MuJoCo RGBD sensor")
        print("2. In another terminal, run:")
        print("   ros2 topic list")
        print("   ros2 topic echo /mujoco_camera/rgb/image_raw")
        print("   rviz2  # Visualize images")

    print("="*70)
