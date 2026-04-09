#!/usr/bin/env python3
"""
机械臂仿真主程序

功能流程:
1. 自动采样 → IK → 插值
2. 轨迹可视化显示
3. 机器人末端跟随轨迹
4. ROS2 Joint State发布 (实时发布关节状态到 /joint_states 话题)
5. 60Hz图像显示 + 自动视频保存

使用方法:
    # 使用系统Python以支持ROS2功能
    /usr/bin/python3 main/main.py
    
ROS2功能:
    - 自动发布关节状态到 /joint_states 话题 (500Hz)
    - 订阅关节控制命令从 /joint_target 话题
    - 混合控制模式:
      * 默认演示: 自动生成随机轨迹演示 (每5秒一个新目标)
      * ROS2控制: 收到命令时立即响应，优先级最高
    - 查看关节状态:
      source /opt/ros/humble/setup.bash
      ros2 topic echo /joint_states
    - 控制机械臂:
      ros2 topic pub /joint_target sensor_msgs/msg/JointState \
        '{name: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], 
          position: [0.0, 0.5, -1.0, 0.0, -0.5, 0.0]}'

视频功能:
    - 自动保存机器人运行视频 (60FPS)
    - 文件名格式: robot_trajectory_YYYYMMDD_HHMMSS.mp4
    - 按 'q' 键退出并保存视频
"""

import numpy as np
import sys
import os
import time
import mujoco
import threading
import argparse
from queue import Queue, Empty
from typing import Optional, Tuple
from datetime import datetime

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../sensors'))

from rl_cartesian_env_ik import RLCartesianEnvWithIK

# 导入 ROS2 RGBD 发布器（使用标准库 cv_bridge）
try:
    from rgbd_ros2_publisher import RGBD_ROS2_Publisher
    ros2_publisher_available = True
except ImportError:
    RGBD_ROS2_Publisher = None
    ros2_publisher_available = False
    print("⚠ RGBD_ROS2_Publisher not available")

# 导入热成像传感器
try:
    from thermal_sensor import ThermalSensor
    thermal_sensor_available = True
except ImportError:
    ThermalSensor = None
    thermal_sensor_available = False
    print("⚠ ThermalSensor not available")

# 导入共享内存相关
try:
    from multiprocessing import shared_memory
    import pickle
    shared_memory_available = True
except ImportError:
    shared_memory_available = False
    print("⚠ shared_memory not available")


class ROS2PublisherThread:
    """
    独立的 ROS2 发布线程
    通过共享内存获取图像数据并发布到 ROS2
    """

    def __init__(self, camera_configs: list):
        """
        初始化 ROS2 发布线程

        Args:
            camera_configs: 相机配置列表，每个配置包含 camera_name 和 node_name
                           例如: [{'camera_name': 'ee_camera', 'node_name': 'ee_camera_publisher'}]
        """
        self.camera_configs = camera_configs
        self.publishers = {}
        self.is_running = False
        self.thread = None

        # 共享内存字典 {camera_name: {'rgb': shm, 'depth': shm, 'intrinsics': shm, 'ready': bool}}
        self.shared_memory = {}
        self.memory_lock = threading.Lock()

        # 图像数据缓存
        self.image_cache = {}

    def initialize_publishers(self):
        """初始化所有 ROS2 发布器"""
        if not ros2_publisher_available:
            print("❌ ROS2 发布器不可用")
            return False

        try:
            for config in self.camera_configs:
                camera_name = config['camera_name']
                node_name = config['node_name']

                publisher = RGBD_ROS2_Publisher(
                    camera_name=camera_name,
                    node_name=node_name
                )
                self.publishers[camera_name] = publisher

                # 初始化共享内存槽位
                self.shared_memory[camera_name] = {
                    'rgb': None,
                    'depth': None,
                    'intrinsics': None,
                    'ready': False
                }

            print(f"✓ ROS2 发布线程初始化完成 ({len(self.publishers)} 个相机)")
            return True

        except Exception as e:
            print(f"❌ ROS2 发布器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_image_data(self, camera_name: str, rgb, depth, intrinsics, timestamp_sec):
        """
        更新图像数据到共享缓存

        Args:
            camera_name: 相机名称
            rgb: RGB 图像
            depth: 深度图像
            intrinsics: 相机内参
            timestamp_sec: 时间戳
        """
        with self.memory_lock:
            self.image_cache[camera_name] = {
                'rgb': rgb.copy() if rgb is not None else None,
                'depth': depth.copy() if depth is not None else None,
                'intrinsics': intrinsics,
                'timestamp_sec': timestamp_sec,
                'ready': True
            }

    def _publish_loop(self):
        """ROS2 发布循环（在独立线程中运行）"""
        print("📡 ROS2 发布线程已启动")

        frame_count = 0

        while self.is_running:
            try:
                # 遍历所有相机
                for camera_name, publisher in self.publishers.items():
                    # 获取图像数据
                    with self.memory_lock:
                        if camera_name not in self.image_cache:
                            continue

                        cache = self.image_cache[camera_name]
                        if not cache.get('ready', False):
                            continue

                        # 复制数据（避免长时间持有锁）
                        rgb = cache['rgb']
                        depth = cache['depth']
                        intrinsics = cache['intrinsics']
                        timestamp_sec = cache['timestamp_sec']

                        # 标记为已处理
                        cache['ready'] = False

                    # 发布到 ROS2（在锁外执行）
                    if rgb is not None:
                        try:
                            publisher.publish_rgbd(
                                rgb=rgb,
                                depth=depth,
                                intrinsics=intrinsics,
                                timestamp_sec=timestamp_sec
                            )
                            frame_count += 1

                        except Exception as e:
                            if frame_count % 100 == 0:
                                print(f"⚠ {camera_name} 发布失败: {e}")

                # 短暂休眠，避免CPU占用过高
                time.sleep(0.001)  # 1ms

            except Exception as e:
                print(f"❌ ROS2 发布循环错误: {e}")
                import traceback
                traceback.print_exc()

        print("📡 ROS2 发布线程已停止")

    def start(self):
        """启动 ROS2 发布线程"""
        if self.is_running:
            print("⚠ ROS2 发布线程已在运行")
            return False

        if not self.initialize_publishers():
            return False

        self.is_running = True
        self.thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.thread.start()

        print("✓ ROS2 发布线程已启动")
        return True

    def stop(self):
        """停止 ROS2 发布线程"""
        if not self.is_running:
            return

        print("🛑 正在停止 ROS2 发布线程...")
        self.is_running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("⚠ ROS2 发布线程未能在超时时间内结束")

        # 清理发布器
        for publisher in self.publishers.values():
            try:
                publisher.shutdown()
            except:
                pass

        print("✓ ROS2 发布线程已停止")


# ROS2 Joint State Subscriber (optional)
try:
    from ros2_joint_state_subscriber import JointStateSubscriberROS2, ros2_available
except ImportError:
    JointStateSubscriberROS2 = None
    def ros2_available():
        return False

# Teaching Status Subscriber and Trajectory Recorder
try:
    from teaching_status_subscriber import TeachingStatusSubscriber
    from trajectory_data_recorder import TrajectoryDataRecorder
    teaching_recording_available = True
except ImportError:
    TeachingStatusSubscriber = None
    TrajectoryDataRecorder = None
    teaching_recording_available = False

# Cartesian Target / RRT / IK 链路已移除

# ROS2 Rosbag Recorder
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../tools/ros2_recorder'))
    from rosbag_recorder import RosbagRecorder
    rosbag_recorder_available = True
except ImportError:
    RosbagRecorder = None
    rosbag_recorder_available = False
    print("⚠ RosbagRecorder not available")

class ContinuousArmController:
    """
    连续机械臂控制器 - 基于轨迹反馈控制
    
    功能:
    1. 全程保持机械臂控制
    2. 没有新目标时维持当前状态
    3. 有新目标时生成并跟踪轨迹
    4. 实时可视化轨迹执行
    5. 轨迹反馈控制：只有当前关节达到目标关节时才推进到下一个轨迹点
    
    轨迹反馈控制机制:
    - 每个控制循环计算当前关节与目标关节的误差
    - 当误差小于阈值且连续满足若干次时，推进到下一个轨迹点
    - 这确保了机械臂真正到达每个轨迹点后才继续执行
    """
    
    def __init__(self, env: RLCartesianEnvWithIK, enable_ros_control: bool = False, enable_teaching_recording: bool = True, enable_rosbag: bool = False):
        self.env = env
        self._base_body_id = -1
        self._base_body_name = None
        for candidate_name in ['base_link', 'base']:
            candidate_id = mujoco.mj_name2id(self.env.robot.model, mujoco.mjtObj.mjOBJ_BODY, candidate_name)
            if candidate_id >= 0:
                self._base_body_id = candidate_id
                self._base_body_name = candidate_name
                break
        self.current_trajectory = None
        self.trajectory_step = 0
        self.target_queue = Queue()  # 新目标队列
        self.is_running = False
        self.control_thread = None

        # 控制参数
        self.control_frequency = 500  # Hz (高频率控制+Joint State发布)
        self.dt = 1.0 / self.control_frequency

        # 轨迹参数
        self.trajectory_duration = 3.0  # 期望轨迹执行时间（秒）
        self.interpolation_steps = int(self.trajectory_duration * self.control_frequency)  # 基于时间的插值步数
        self.execution_speed = 0.8  # 执行速度倍率

        # 轨迹执行参数（开环控制）
        self.waypoint_counter = 0  # 轨迹点计数器

        # ROS2 spin 优化（用于高频发布）
        self.ros2_spin_counter = 0  # spin计数器

        # ROS控制模式
        self.enable_ros_control = enable_ros_control and ros2_available()
        self.ros_subscriber = None
        self.ros_target_joints = None
        self.control_mode = "ros_direct"
        self.ros_target_lock = threading.Lock()  # 默认ROS2直接控制模式

        # 夹爪抓取状态发布
        # gripper_command > closing_threshold 时认为夹爪在关闭
        # 且 EE 到最近可抓取物体 < grasp_distance_threshold 时认为已抓住
        self.grasp_distance_threshold = 0.05   # m
        self.gripper_closing_threshold = 128.0
        self.grasp_contact_distance_threshold = 0.003  # m，接触距离阈值（<=该值视为有效接触）
        self.grasp_normal_force_threshold = 0.5  # N，接触法向力阈值
        self._gripper_status_pub_node = None
        self._gripper_status_pub = None
        self._grasp_dist_pub = None
        self._graspable_body_ids = []   # 在 start() 中扫描
        self._gripper_body_ids = []     # 在 start() 中扫描
        self._left_finger_body_ids = []
        self._right_finger_body_ids = []
        self._ee_pose_pub_info = None

        # Teaching 和记录功能
        self.enable_teaching_recording = enable_teaching_recording and teaching_recording_available
        self.teaching_subscriber = None
        self.trajectory_recorder = None
        self.current_teaching_status = None
        self.is_teaching_recording = False

        # Rosbag 录制功能
        self.enable_rosbag = enable_rosbag and rosbag_recorder_available
        self.rosbag_recorder = None
        self.rosbag_executor = None  # 保存 executor 引用以便清理
        self.rosbag_publisher = None  # 用于发布 /teaching_status 话题给 rosbag recorder

        # 按键检测示教
        self.keyboard_teaching_enabled = True
        self.last_key_time = 0

        # 相机图像缓存（避免OpenGL冲突）
        self.latest_camera_image = None
        self.camera_image_lock = threading.Lock()
        self.camera_thread = None
        self.camera_thread_running = False

        # 用于主线程传递相机图像给控制线程
        self.camera_image_queue = Queue(maxsize=2)

        # Cartesian Target / RRT / IK 链路已移除

        if self.enable_ros_control:
            try:
                import time
                # 生成唯一节点名避免冲突
                node_name = f"arm_controller_subscriber_{int(time.time() * 1000) % 10000}"
                self.ros_subscriber = JointStateSubscriberROS2(
                    topic="/joint_target",
                    node_name=node_name,
                    callback=self._ros_joint_callback
                )
                print(f"✓ ROS2 Joint State Subscriber initialized (node: {node_name})")
            except Exception as e:
                print(f"⚠ Failed to initialize ROS2 subscriber: {e}")
                self.enable_ros_control = False

        # 初始化夹爪状态发布器（/gripper_status, /grasp_distance）
        if ros2_available():
            try:
                import rclpy
                from rclpy.node import Node
                from std_msgs.msg import String as StringMsg, Float64 as Float64Msg

                _pub_ctx = rclpy.Context()
                rclpy.init(context=_pub_ctx)
                _pub_node_name = f"gripper_status_publisher_{int(time.time() * 1000) % 10000}"
                _pub_node = Node(_pub_node_name, context=_pub_ctx)
                self._gripper_status_pub = _pub_node.create_publisher(StringMsg, '/gripper_status', 10)
                self._grasp_dist_pub = _pub_node.create_publisher(Float64Msg, '/grasp_distance', 10)
                self._gripper_status_pub_node = _pub_node
                print(f"✓ 夹爪状态发布器已启动 (节点: {_pub_node_name})")
                print(f"  - /gripper_status  (std_msgs/String): idle | grasping | grasped")
                print(f"  - /grasp_distance  (std_msgs/Float64): EE到最近物体距离 (m)")
            except Exception as e:
                print(f"⚠ 夹爪状态发布器初始化失败: {e}")

        # 初始化末端位姿发布器（/ee_pose）
        if ros2_available():
            try:
                import rclpy
                from rclpy.node import Node
                from geometry_msgs.msg import PoseStamped

                _ee_pub_ctx = rclpy.Context()
                rclpy.init(context=_ee_pub_ctx)
                _ee_pub_node_name = f"ee_pose_publisher_{int(time.time() * 1000) % 10000}"
                _ee_pub_node = Node(_ee_pub_node_name, context=_ee_pub_ctx)
                _ee_pose_pub = _ee_pub_node.create_publisher(PoseStamped, '/ee_pose', 10)
                self._ee_pose_pub_info = {
                    'context': _ee_pub_ctx,
                    'node': _ee_pub_node,
                    'publisher': _ee_pose_pub,
                }
                print(f"✓ EE pose publisher initialized (node: {_ee_pub_node_name})")
                print(f"  - /ee_pose (geometry_msgs/PoseStamped, base_link frame)")
            except Exception as e:
                print(f"⚠ EE pose publisher init failed: {e}")

        # 初始化 Teaching 功能
        if self.enable_teaching_recording:
            try:
                # 初始化轨迹记录器
                self.trajectory_recorder = TrajectoryDataRecorder(base_data_dir="data")

                # 如果ROS2可用，也启用ROS2订阅
                if ros2_available():
                    teaching_node_name = f"teaching_status_subscriber_{int(time.time() * 1000) % 10000}"
                    self.teaching_subscriber = TeachingStatusSubscriber(
                        topic="/teaching_status",
                        node_name=teaching_node_name,
                        callback=self._teaching_status_callback
                    )
                    print(f"✓ ROS2 Teaching Status Subscriber: {teaching_node_name}")

                print(f"✓ Teaching Recording System initialized")
                print(f"  - 数据存储: data/ 目录")
                print(f"  - 按键控制: 't' 开始/停止录制")
                print(f"  - ROS2控制: /teaching_status 话题 (如果可用)")
            except Exception as e:
                print(f"⚠ Failed to initialize Teaching Recording: {e}")
                self.enable_teaching_recording = False

        # 初始化 Rosbag 录制器
        if self.enable_rosbag:
            try:
                # 创建 RosbagRecorder 实例（在独立线程中运行）
                self.rosbag_recorder = RosbagRecorder(
                    output_dir="/home/tao/Downloads/Arm_Project/rosbag_data",
                    topics=[
                        '/ee_camera/rgb/image_raw',
                        '/external_camera/rgb/image_raw',
                        '/joint_states_sim',
                        '/joint_target',
                        '/gripper_status'
                    ],
                    storage_format="mcap",
                    compression_mode="message",  # 启用压缩
                    compression_format="zstd",
                    topic_rename_map={  # 自动重命名话题
                        '/joint_states_sim': '/joint_states',
                        '/joint_target': '/joint_cmd'
                    }
                )

                # 创建 ROS2 publisher 用于向 rosbag recorder 发送控制命令
                if ros2_available():
                    try:
                        import rclpy
                        from std_msgs.msg import String

                        # 创建一个临时节点用于发布 teaching_status
                        teaching_pub_node_name = f"teaching_status_publisher_{int(time.time() * 1000) % 10000}"

                        # 注意: 这里需要一个简单的publisher，不需要完整的节点
                        # 我们将在 start() 中初始化它
                        self.rosbag_publisher_node_name = teaching_pub_node_name

                    except Exception as e:
                        print(f"⚠ 无法创建 teaching_status publisher: {e}")

                print(f"✓ Rosbag Recording System initialized")
                print(f"  - 输出目录: rosbag_data/")
                print(f"  - 压缩模式: message (zstd)")
                print(f"  - 话题重命名: 启用")
                print(f"  - 按键控制: 'b' 开始/停止 rosbag 录制")

            except Exception as e:
                print(f"⚠ Failed to initialize Rosbag Recorder: {e}")
                import traceback
                traceback.print_exc()
                self.enable_rosbag = False

        print(f"✓ 控制器初始化 - 频率: {self.control_frequency}Hz, 轨迹时长: {self.trajectory_duration:.1f}s, 插值点: {self.interpolation_steps}")
        if self.enable_ros_control:
            print(f"✓ ROS2控制模式可用 - 订阅话题: /joint_target")
        if self.enable_teaching_recording:
            print(f"✓ Teaching记录模式可用 - 订阅话题: /teaching_status")
        
    def add_target(self, target_type: str = "random", target_pos: Optional[np.ndarray] = None):
        """添加新目标到队列"""
        self.target_queue.put((target_type, target_pos))
        print(f"New target added: {target_type}")
        
    def _ros_joint_callback(self, joint_positions: np.ndarray):
        """ROS joint state callback function."""
        try:
            if hasattr(self, '_callback_count'):
                self._callback_count += 1
            else:
                self._callback_count = 1

            # /joint_target 方向映射：将 J3、J4 取反（1-based: J3/J4 -> index 2/3）
            remapped_joints = joint_positions.copy()
            if len(remapped_joints) >= 5:
                remapped_joints[2] = -remapped_joints[2]
                remapped_joints[3] = -remapped_joints[3]
                remapped_joints[5] = -remapped_joints[5]

            
            with self.ros_target_lock:
                if hasattr(self, '_last_ros_joints') and self._last_ros_joints is not None:
                    if np.allclose(remapped_joints, self._last_ros_joints, atol=1e-6):
                        return
                
                self.ros_target_joints = remapped_joints
                self._last_ros_joints = remapped_joints.copy()

                self.control_mode = "ros_direct"

            pass
                
        except Exception as e:
            print(f"❌ ROS2回调处理错误: {e}")

    def _teaching_status_callback(self, status: str):
        """Teaching status callback function."""
        try:
            self.current_teaching_status = status

            if status == "start_teaching":
                print(f"📚 开始Teaching - 启动轨迹记录")
                if self.trajectory_recorder:
                    # 生成基于时间戳的会话名称
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    session_name = f"teaching_{timestamp}"
                    recording_dir = self.trajectory_recorder.start_recording(session_name)
                    print(f"🎬 轨迹记录已开始: {recording_dir}")

            elif status == "end_teaching":
                print(f"📚 结束Teaching - 停止轨迹记录")
                if self.trajectory_recorder:
                    recording_dir = self.trajectory_recorder.stop_recording()
                    if recording_dir:
                        print(f"✅ 轨迹记录已完成: {recording_dir}")

        except Exception as e:
            print(f"Teaching status callback error: {e}")

    def _publish_teaching_status(self, status: str):
        """发布 teaching_status 消息到 RosbagRecorder"""
        try:
            if not ros2_available():
                print("⚠ ROS2 不可用，无法发布 teaching_status")
                return

            # 直接调用 rosbag_recorder 的回调
            if self.rosbag_recorder:
                from std_msgs.msg import String
                msg = String()
                msg.data = status
                self.rosbag_recorder.teaching_status_callback(msg)
                print(f"📤 发送 teaching_status: {status}")

        except Exception as e:
            print(f"❌ 发布 teaching_status 失败: {e}")
            import traceback
            traceback.print_exc()


    def set_control_mode(self, mode: str):
        """
        设置控制模式
        
        Args:
            mode: "trajectory" or "ros_direct"
        """
        if mode == "ros_direct" and not self.enable_ros_control:
            print("⚠ ROS控制模式不可用，请检查ROS2环境")
            return False
            
        self.control_mode = mode
        
        if mode == "ros_direct":
            # 清除当前轨迹，切换到ROS直接控制
            self.current_trajectory = None
            self.trajectory_step = 0
            print(f"✓ 切换到ROS直接控制模式 - 等待 /joint_target 消息（支持7个关节：6个机械臂关节 + 1个gripper关节）")
        elif mode == "trajectory":
            print(f"✓ 切换到轨迹控制模式")
        
        return True

    def get_control_mode(self) -> str:
        """获取当前控制模式"""
        return self.control_mode

    def is_ros_control_available(self) -> bool:
        """检查ROS控制是否可用"""
        return self.enable_ros_control
        
    def set_trajectory_duration(self, duration: float):
        """
        设置轨迹执行时间并重新计算插值步数
        
        Args:
            duration: 期望轨迹执行时间（秒）
        """
        self.trajectory_duration = duration
        self.interpolation_steps = int(duration * self.control_frequency)
        print(f"✓ 轨迹时间设置为 {duration:.1f}秒，插值步数: {self.interpolation_steps}")
        
    def generate_trajectory_to_target(self, target_pos: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        生成到目标位置的轨迹
        
        Args:
            target_pos: 目标位置 [x, y, z]
            
        Returns:
            trajectory: 关节轨迹 (n_steps, 6)
            cartesian_waypoints: 笛卡尔路点 (n_waypoints, 7)
        """
            
        # 获取当前关节位置
        current_joints = self.env.robot.data.qpos[:6].copy()
        current_pos = self.env.robot.get_ee_position()
        current_quat = self.env.robot.get_ee_orientation()
        
        # 无IK模式：使用局部雅可比伪逆迭代逼近目标位置
        target_quat = current_quat
        target_joints = current_joints.copy()
        q_backup = self.env.robot.data.qpos[:6].copy()

        converged = False
        for _ in range(60):
            self.env.robot.data.qpos[:6] = target_joints
            mujoco.mj_forward(self.env.robot.model, self.env.robot.data)
            ee_pos = self.env.robot.get_ee_position()
            pos_error = target_pos - ee_pos
            if np.linalg.norm(pos_error) < 0.01:
                converged = True
                break
            dq = self.env._compute_jacobian_pseudoinverse(pos_error)
            target_joints = target_joints + dq
            if hasattr(self.env, 'joint_limits'):
                target_joints = np.clip(target_joints, self.env.joint_limits[:, 0], self.env.joint_limits[:, 1])

        # 恢复状态
        self.env.robot.data.qpos[:6] = q_backup
        mujoco.mj_forward(self.env.robot.model, self.env.robot.data)

        if not converged:
            print(f"❌ 关节目标逼近失败 - 目标位置: {target_pos}")
            return None, None

        print(f"✓ 关节目标逼近成功 - 目标位置: {target_pos}")
        
        # 创建路点：当前位置 -> 目标位置
        joint_waypoints = np.array([current_joints, target_joints])
        cartesian_waypoints = np.array([
            [*current_pos, *current_quat],
            [*target_pos, *target_quat]
        ])
        
        # 生成插值轨迹
        trajectory = self.env.joint_interpolator.interpolate_waypoints(
            joint_waypoints, self.interpolation_steps
        )
        
        # 可视化轨迹
        if hasattr(self.env, 'trajectory_drawer') and self.env.trajectory_drawer is not None:
            # 提取末端位置用于可视化
            ee_positions = []
            for joints in trajectory:
                self.env.robot.data.qpos[:6] = joints
                mujoco.mj_forward(self.env.robot.model, self.env.robot.data)
                ee_pos = self.env.robot.get_ee_position()
                ee_positions.append(ee_pos)
            
            trajectory_points = np.array(ee_positions)
            self.env.trajectory_drawer.draw_trajectory(trajectory_points)
            
        return trajectory, cartesian_waypoints
        
    def control_loop(self):
        """主控制循环 - 实时轨迹跟踪与可视化"""
        final_actual_joints = None
        loop_count = 0
        while self.is_running:
            loop_start_time = time.time()
            loop_count += 1

            try:
                # 检查是否有新目标 (仅在轨迹控制模式下)
                if self.control_mode == "trajectory":
                    try:
                        target_type, target_pos = self.target_queue.get_nowait()
                        
                        if target_type == "random":
                            # 随机采样新目标 - 快速生成不阻塞
                            result = self.env.generate_trajectory_from_sampling(
                                n_targets=1,
                                interpolation_steps=self.interpolation_steps,
                                max_retries_per_target=20
                            )
                            
                            if result[0] is not None:
                                joint_waypoints, cartesian_waypoints, trajectory = result
                                target_pos = cartesian_waypoints[-1, :3]
                                print(f"New random target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                                print(f"Executing trajectory ({len(trajectory)} steps)")

                                # 可视化轨迹
                                self.env.visualize_trajectory(joint_waypoints, cartesian_waypoints, trajectory)

                                self.current_trajectory = trajectory
                                self.trajectory_step = 0
                                self.waypoint_counter = 0
                            else:
                                print("Random target generation failed")

                        elif target_type == "manual" and target_pos is not None:
                            # 手动指定目标 - 立即生成轨迹
                            trajectory, cartesian_waypoints = self.generate_trajectory_to_target(target_pos)
                            if trajectory is not None:
                                print(f"Manual target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                                self.current_trajectory = trajectory
                                self.trajectory_step = 0
                                self.waypoint_counter = 0
                            else:
                                print("Manual target trajectory generation failed")
                                
                    except Empty:
                        # 没有新目标，继续当前行为
                        pass
                
                # 控制逻辑 - 根据当前模式
                if self.control_mode == "ros_direct":
                    with self.ros_target_lock:
                        target_joints_copy = self.ros_target_joints.copy() if self.ros_target_joints is not None else None

                    if target_joints_copy is not None:
                        self.env.robot.update_joint_state()

                        if len(target_joints_copy) >= 7:
                            self.env.robot.apply_joint_control(target_joints_copy[:6])
                            gripper_scaled = target_joints_copy[6] * 255.0
                            self.env.robot.set_gripper_command(gripper_scaled)
                        else:
                            self.env.robot.apply_joint_control(target_joints_copy)

                    else:
                        self.env.robot.update_joint_state()
                        current_joints = self.env.robot.get_joint_positions()
                        self.env.robot.apply_joint_control(current_joints)
                        
                elif self.control_mode == "trajectory":
                    if self.current_trajectory is not None and self.trajectory_step < len(self.current_trajectory):
                        # 当前目标关节角度
                        target_joints = self.current_trajectory[self.trajectory_step]
                        
                        # 更新机器人状态
                        self.env.robot.update_joint_state()
                        
                        # 应用关节控制
                        self.env.robot.apply_joint_control(target_joints)
                        
                        # 简单按控制周期推进轨迹点
                        self.waypoint_counter += 1
                        if self.waypoint_counter >= 1:  # 每个控制周期推进一步
                            self.trajectory_step += 1
                            self.waypoint_counter = 0
                        
                        # 轨迹执行完成
                        if self.trajectory_step >= len(self.current_trajectory):
                            # 获取关节目标和反馈
                            joint_target = self.current_trajectory[-1]
                            joint_feedback = self.env.robot.get_joint_feedback()
                            joint_error = np.linalg.norm(joint_target - joint_feedback)

                            print(f"Trajectory completed, joint error: {joint_error:.6f} rad")

                            self.env.robot.update_joint_state()
                            final_actual_joints = self.current_trajectory[-1].copy()
                            self.env.robot.apply_joint_control(final_actual_joints)
                            self.current_trajectory = None
                            self.trajectory_step = 0
                            self.waypoint_counter = 0
                    else:
                        # 没有轨迹时保持当前位置 - 应用PID控制
                        self.env.robot.update_joint_state()
                        current_joints = self.env.robot.get_joint_positions()
                        if final_actual_joints is None:
                            final_actual_joints = current_joints
                        self.env.robot.apply_joint_control(final_actual_joints)
                
                # 执行一步仿真（这里会应用PID力矩控制）
                self.env.robot.update_obstacles()
                self.env.robot.step()  # 应用PID计算的力矩

                # 发布夹爪状态（每 50 步 ≈ 10Hz）
                if loop_count % 50 == 0 and self._gripper_status_pub is not None:
                    try:
                        from std_msgs.msg import String as StringMsg, Float64 as Float64Msg

                        status_str, min_dist = self._estimate_grasp_state()

                        sm = StringMsg()
                        sm.data = status_str
                        self._gripper_status_pub.publish(sm)

                        dm = Float64Msg()
                        dm.data = min_dist if min_dist != float('inf') else -1.0
                        self._grasp_dist_pub.publish(dm)
                    except Exception:
                        pass

                # 发布末端位姿（每 10 步 ≈ 50Hz）
                if loop_count % 10 == 0 and self._ee_pose_pub_info is not None:
                    try:
                        from geometry_msgs.msg import PoseStamped

                        ee_pos_world = self.env.robot.get_ee_position()
                        ee_quat_world = self.env.robot.get_ee_orientation()

                        if self._base_body_id >= 0:
                            base_rot_world = self.env.robot.data.xmat[self._base_body_id].reshape(3, 3).copy()
                            base_pos_world = self.env.robot.data.xpos[self._base_body_id].copy()
                            ee_pos = base_rot_world.T @ (ee_pos_world - base_pos_world)

                            from scipy.spatial.transform import Rotation as R
                            base_rot_obj = R.from_matrix(base_rot_world)
                            world_rot_obj = R.from_quat([
                                ee_quat_world[1], ee_quat_world[2], ee_quat_world[3], ee_quat_world[0]
                            ])
                            ee_rot_base = base_rot_obj.inv() * world_rot_obj
                            ee_quat_base_xyzw = ee_rot_base.as_quat()
                            ee_quat = np.array([
                                ee_quat_base_xyzw[3],
                                ee_quat_base_xyzw[0],
                                ee_quat_base_xyzw[1],
                                ee_quat_base_xyzw[2],
                            ])
                        else:
                            ee_pos = ee_pos_world
                            ee_quat = ee_quat_world

                        ee_msg = PoseStamped()
                        ee_msg.header.stamp = self._ee_pose_pub_info['node'].get_clock().now().to_msg()
                        ee_msg.header.frame_id = 'base_link'
                        ee_msg.pose.position.x = float(ee_pos[0])
                        ee_msg.pose.position.y = float(ee_pos[1])
                        ee_msg.pose.position.z = float(ee_pos[2])
                        ee_msg.pose.orientation.w = float(ee_quat[0])
                        ee_msg.pose.orientation.x = float(ee_quat[1])
                        ee_msg.pose.orientation.y = float(ee_quat[2])
                        ee_msg.pose.orientation.z = float(ee_quat[3])
                        self._ee_pose_pub_info['publisher'].publish(ee_msg)
                    except Exception:
                        pass

                # V4: 相机渲染移到显示循环中，避免重复渲染导致 OpenGL 冲突
                # 控制循环只负责机械臂控制，不渲染相机
                
                # 记录Teaching轨迹数据（如果在录制中）
                if (self.enable_teaching_recording and 
                    self.trajectory_recorder and 
                    (self.is_teaching_recording or self.trajectory_recorder.is_recording)):
                    try:
                        # 获取当前状态数据
                        current_joints = self.env.robot.get_joint_positions()
                        current_ee_pos = self.env.robot.get_ee_position()
                        current_ee_quat = self.env.robot.get_ee_orientation()
                        
                        # 从队列获取摄像头图像字典（由主线程提供，避免OpenGL冲突）
                        camera_images = None
                        try:
                            camera_images = self.camera_image_queue.get_nowait()
                        except Empty:
                            pass
                        
                        # 记录一帧数据（传递摄像头图像字典）
                        self.trajectory_recorder.record_frame(
                            joint_positions=current_joints,
                            ee_position=current_ee_pos,
                            ee_orientation=current_ee_quat,
                            camera_images=camera_images,
                            timestamp=time.time()
                        )
                        
                    except Exception as e:
                        if hasattr(self, '_recording_error_count'):
                            self._recording_error_count += 1
                        else:
                            self._recording_error_count = 1
                        
                        if self._recording_error_count % 100 == 1:
                            print(f"❌ Teaching记录错误 #{self._recording_error_count}: {e}")
                
                # 发布ROS2 Joint State (如果启用) - 500Hz高频率发布
                if self.env.robot.enable_joint_state_ros2 and self.env.robot.joint_state_pub is not None:
                    try:
                        # 每10次循环才调用spin_once以提高性能 (500Hz/10 = 50Hz spin频率)
                        skip_spin = (self.ros2_spin_counter % 10) != 0
                        self.env.robot.publish_joint_state_ros2(
                            stamp_sec=time.time(), 
                            skip_spin=skip_spin
                        )
                        self.ros2_spin_counter += 1
                    except Exception as e:
                        print(f"⚠ Joint State发布失败: {e}")
                # 同步可视化
                if self.env.viewer is not None:
                    self.env.viewer.sync()
                
                # 控制循环频率
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, self.dt - loop_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Control loop error: {e}")
                break

        print("Control loop ended")

    def _estimate_grasp_state(self):
        """估计当前抓取状态。

        规则：
            1) 夹爪未闭合 -> idle
            2) 左右手指均与同一可抓取物体接触，且接触法向力超过阈值 -> grasped
            3) 夹爪闭合但不满足2 -> grasping
        """
        gripper_cmd = self.env.robot.gripper_command
        is_grasping = gripper_cmd > self.gripper_closing_threshold

        # 计算 EE 到最近可抓取物体距离
        min_dist = float('inf')
        if self._graspable_body_ids:
            ee_pos = self.env.robot.get_ee_position()
            for bid in self._graspable_body_ids:
                obj_pos = self.env.robot.data.xpos[bid]
                d = float(np.linalg.norm(ee_pos - obj_pos))
                if d < min_dist:
                    min_dist = d

        # 接触检测：左右手指分别接触到“同一个”可抓取物体，且法向接触力达阈值
        has_two_side_contact = False
        if self._left_finger_body_ids and self._right_finger_body_ids and self._graspable_body_ids:
            graspable_set = set(self._graspable_body_ids)
            left_set = set(self._left_finger_body_ids)
            right_set = set(self._right_finger_body_ids)

            left_contact_objs = set()
            right_contact_objs = set()

            for i in range(self.env.robot.data.ncon):
                c = self.env.robot.data.contact[i]
                if c.dist > self.grasp_contact_distance_threshold:
                    continue

                b1 = int(self.env.robot.model.geom_bodyid[c.geom1])
                b2 = int(self.env.robot.model.geom_bodyid[c.geom2])

                # 读取接触力（contact frame）: force_torque[0] 是法向分量
                force_torque = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.env.robot.model, self.env.robot.data, i, force_torque)
                normal_force = abs(float(force_torque[0]))
                if normal_force < self.grasp_normal_force_threshold:
                    continue

                # 左指-物体
                if b1 in left_set and b2 in graspable_set:
                    left_contact_objs.add(b2)
                elif b2 in left_set and b1 in graspable_set:
                    left_contact_objs.add(b1)

                # 右指-物体
                if b1 in right_set and b2 in graspable_set:
                    right_contact_objs.add(b2)
                elif b2 in right_set and b1 in graspable_set:
                    right_contact_objs.add(b1)

            has_two_side_contact = len(left_contact_objs & right_contact_objs) > 0

        has_grasped = is_grasping and has_two_side_contact

        if has_grasped:
            return "grasped", min_dist
        if is_grasping:
            return "grasping", min_dist
        return "idle", min_dist
        
    def start(self):
        """启动连续控制"""
        if self.is_running:
            print("Controller already running")
            return

        self.is_running = True

        # 扫描场景中的可抓取物体（排除机器人、地板、桌面、夹爪自身等）
        gripper_kw = ['gripper', 'finger', 'pad', 'coupler', 'follower', 'driver', 'robotiq']
        exclude_kw = ['world', 'base', 'link', 'floor', 'table', 'bench',
                      'wall', 'obstacle', 'target', 'gripper', 'camera',
                      'light', 'site']
        self._graspable_body_ids = []
        self._gripper_body_ids = []
        self._left_finger_body_ids = []
        self._right_finger_body_ids = []
        try:
            for i in range(self.env.robot.model.nbody):
                name = self.env.robot.model.body(i).name.lower()
                if not name:
                    continue
                if any(kw in name for kw in exclude_kw):
                    continue
                if any(kw in name for kw in gripper_kw):
                    continue

                # 仅把 freejoint 的可动物体作为可抓取候选
                body_jntnum = int(self.env.robot.model.body_jntnum[i])
                if body_jntnum <= 0:
                    continue
                jnt_adr = int(self.env.robot.model.body_jntadr[i])
                if int(self.env.robot.model.jnt_type[jnt_adr]) != int(mujoco.mjtJoint.mjJNT_FREE):
                    continue

                if self.env.robot.model.body_mass[i] > 0.01:
                    self._graspable_body_ids.append(i)

            # 扫描夹爪/手指相关 body
            for i in range(self.env.robot.model.nbody):
                name = self.env.robot.model.body(i).name.lower()
                if not name:
                    continue
                if any(kw in name for kw in gripper_kw):
                    self._gripper_body_ids.append(i)
                    if ('left' in name) and any(k in name for k in ['finger', 'pad', 'follower', 'coupler', 'driver']):
                        self._left_finger_body_ids.append(i)
                    if ('right' in name) and any(k in name for k in ['finger', 'pad', 'follower', 'coupler', 'driver']):
                        self._right_finger_body_ids.append(i)
        except Exception as e:
            print(f"⚠ 可抓取物体扫描失败: {e}")

        # 启动时主动发布一次 gripper 状态，避免话题初始为空
        if self._gripper_status_pub is not None:
            try:
                from std_msgs.msg import String as StringMsg, Float64 as Float64Msg
                sm = StringMsg()
                sm.data = "idle"
                self._gripper_status_pub.publish(sm)

                if self._grasp_dist_pub is not None:
                    dm = Float64Msg()
                    dm.data = -1.0
                    self._grasp_dist_pub.publish(dm)
            except Exception:
                pass


        if self.ros_subscriber is not None:
            self.ros_subscriber.start_spinning()

        # 启动Teaching订阅器
        if self.teaching_subscriber is not None:
            self.teaching_subscriber.start_spinning()

        # 启动 Rosbag 录制器（使用 MultiThreadedExecutor 在独立线程中运行）
        if self.rosbag_recorder is not None:
            try:
                import rclpy
                from rclpy.executors import MultiThreadedExecutor

                def run_rosbag_recorder():
                    """在独立线程中运行 rosbag recorder"""
                    try:
                        # 创建独立的 MultiThreadedExecutor
                        self.rosbag_executor = MultiThreadedExecutor(num_threads=2)
                        self.rosbag_executor.add_node(self.rosbag_recorder)

                        # 使用 executor.spin() 而不是 rclpy.spin()
                        self.rosbag_executor.spin()
                    except Exception as e:
                        print(f"Rosbag recorder thread error: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        # 清理 executor
                        if self.rosbag_executor is not None:
                            self.rosbag_executor.shutdown()
                            self.rosbag_executor = None

                self.rosbag_thread = threading.Thread(target=run_rosbag_recorder, daemon=True)
                self.rosbag_thread.start()

            except Exception as e:
                print(f"Rosbag recorder startup failed: {e}")
                import traceback
                traceback.print_exc()

        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
        
    def stop(self):
        """停止连续控制"""
        print("Stopping controller...")
        self.is_running = False

        # 停止ROS订阅器
        if self.ros_subscriber is not None:
            try:
                self.ros_subscriber.stop_spinning()
                time.sleep(0.1)
            except Exception as e:
                print(f"ROS2 subscriber stop error: {e}")

        # 清理夹爪状态发布器
        if self._gripper_status_pub_node is not None:
            try:
                self._gripper_status_pub_node.destroy_node()
            except Exception:
                pass

        # 清理末端位姿发布器
        if self._ee_pose_pub_info is not None:
            try:
                self._ee_pose_pub_info['node'].destroy_node()
                self._ee_pose_pub_info['context'].shutdown()
            except Exception:
                pass

        # 停止Teaching订阅器
        if self.teaching_subscriber is not None:
            try:
                self.teaching_subscriber.stop_spinning()
                time.sleep(0.1)
            except Exception as e:
                print(f"Teaching subscriber stop error: {e}")

        # 停止 Rosbag 录制器
        if self.rosbag_recorder is not None:
            try:
                if self.rosbag_recorder.is_recording:
                    self._publish_teaching_status("end_teaching")
                    time.sleep(0.5)

                if self.rosbag_executor is not None:
                    self.rosbag_executor.shutdown()

                if hasattr(self, 'rosbag_thread') and self.rosbag_thread is not None:
                    self.rosbag_thread.join(timeout=5.0)
                    if self.rosbag_thread.is_alive():
                        print("Rosbag thread did not exit within timeout")

                self.rosbag_recorder.shutdown()

            except Exception as e:
                print(f"Rosbag recorder stop error: {e}")
                import traceback
                traceback.print_exc()

        # 确保停止任何进行中的录制
        if self.trajectory_recorder and self.trajectory_recorder.is_recording:
            try:
                self.trajectory_recorder.stop_recording()
            except Exception as e:
                print(f"Trajectory recorder stop error: {e}")

        # 停止控制线程
        if self.control_thread is not None:
            self.control_thread.join(timeout=2.0)
            if self.control_thread.is_alive():
                print("Control thread did not exit within timeout")
            self.control_thread = None

        print("Controller stopped")


def test_integrated_trajectory_control(
    model_path: Optional[str] = None,
    enable_visualization: bool = True,
    enable_ros_control: bool = True,
    enable_rosbag: bool = True,
    enable_image_publish: bool = True,
    enable_thermal: bool = True,
    enable_teaching_recording: bool = True,
    loop_hz: float = 15.0,
):
    """Integrated trajectory control and visualization test"""
    print("=" * 70)
    print("Arm Trajectory Control Test")
    print("=" * 70)

    try:
        # 1. 创建环境
        print("\nInitializing environment...")
        env_xml_path = model_path or os.path.join(os.path.dirname(__file__), '../robot_model/exp/env_robot_torque.xml')
        env = RLCartesianEnvWithIK(
            model_path=env_xml_path,
            enable_visualization=enable_visualization,
            maintain_orientation=True,
            sample_mode='full',
            uniform_sampling=True         # 均匀采样
        )

        # 2. 重置环境
        _, info = env.reset()

        # 强制重置仿真，确保所有可移动物体回到初始位置
        env.robot.reset_simulation()
        mujoco.mj_forward(env.robot.model, env.robot.data)
        print("Simulation forcefully reset - all movable objects returned to initial positions")

        current_pos = info['ee_position']
        print(f"Environment reset, EE position: {current_pos}")

        # 检查ROS2 Joint State发布状态
        if env.robot.enable_joint_state_ros2 and env.robot.joint_state_pub is not None:
            print(f"ROS2 Joint State: Enabled ({env.robot.joint_state_topic}, 500Hz)")
        else:
            print("ROS2 Joint State: Disabled")

        # 3. 禁用障碍物运动（默认移除障碍物）
        env.robot.disable_obstacles()
        print("Obstacles: Disabled")

        # 设置为力矩控制模式以启用PID控制
        env.robot.set_control_mode("torque")
        print("Control mode: Torque (PID enabled)")

        # 初始化热成像传感器（与末端相机 rgbd_camera_ee 共用视角）
        thermal_sensor = None
        if thermal_sensor_available and enable_thermal:
            try:
                thermal_sensor = ThermalSensor(
                    model=env.robot.model,
                    data=env.robot.data,
                    cam_name="rgbd_camera_ee",
                    width=640, height=480,
                    enable_thermal_blur=False,  # 禁用热扩散，避免视角变化导致的温度抖动
                    blur_kernel_size=7,
                    blur_sigma=0.4,
                    enable_distance_attenuation=True,
                    attenuation_coefficient=0.05,
                    enable_noise=False,  # 禁用噪声，避免每帧随机波动
                    noise_stddev=0.3,
                    enable_internal_gradient=False,  # 禁用内部温度梯度（液体-玻璃差异计算），避免容器移动时温度变化
                    edge_temperature_ratio=0.7
                )
                thermal_sensor.set_liquid_temperature("beaker1", 85.0, glass_conductivity=0.5)
                thermal_sensor.set_liquid_temperature("beaker2", 60.0, glass_conductivity=0.5)
                thermal_sensor.set_liquid_temperature("beaker3", 30.0, glass_conductivity=0.5)
                thermal_sensor.set_body_temperature("bench", 35.0)
                thermal_sensor.set_body_temperature("test_tube_rack", 45.0)
                thermal_sensor.set_liquid_temperature("erlenmeyer_flask", 70.0, glass_conductivity=0.5)
                thermal_sensor.set_liquid_temperature("graduated_cylinder", 40.0, glass_conductivity=0.5)
                print("✓ Thermal sensor initialized (cam: rgbd_camera_ee, 640x480)")
            except Exception as e:
                print(f"⚠ Thermal sensor initialization failed: {e}")
                thermal_sensor = None

        # 初始化热成像 ROS2 发布器
        thermal_publisher = None
        if thermal_sensor is not None:
            try:
                from thermal_sensor import ThermalImagePublisher
                thermal_publisher = ThermalImagePublisher(
                    topic="/thermal_camera/image",
                    node_name="thermal_image_publisher"
                )
                print("✓ Thermal ROS2 publisher: /thermal_camera/image")
            except Exception as e:
                print(f"⚠ Thermal ROS2 publisher init failed: {e}")
                thermal_publisher = None

        # 4. 创建独立的 ROS2 图像发布线程
        ros2_publisher_thread = None

        if ros2_publisher_available and enable_image_publish:
            try:
                node_suffix = int(time.time() * 1000) % 100000
                camera_configs = [
                    {'camera_name': 'ee_camera', 'node_name': f'ee_camera_publisher_{node_suffix}'},
                    {'camera_name': 'external_camera', 'node_name': f'external_camera_publisher_{node_suffix}'}
                ]

                ros2_publisher_thread = ROS2PublisherThread(camera_configs)
                if ros2_publisher_thread.start():
                    print("ROS2 image publisher: Started")
                else:
                    ros2_publisher_thread = None

            except Exception as e:
                print(f"ROS2 publisher initialization failed: {e}")
                import traceback
                traceback.print_exc()
                ros2_publisher_thread = None
        else:
            print("ROS2 image publisher not available")

        # 5. 创建连续控制器并启用ROS2控制和Teaching记录
        print("\nCreating controller...")
        controller = ContinuousArmController(
            env,
            enable_ros_control=enable_ros_control,
            enable_teaching_recording=enable_teaching_recording,
            enable_rosbag=enable_rosbag
        )

        # 初始化轨迹绘制器
        from vizUtils import TrajectoryDrawer
        env.trajectory_drawer = TrajectoryDrawer(max_segments=1000, min_distance=0.002)

        # 启动连续控制
        print("\n🚀 启动连续机械臂控制...")
        controller.start()

        # 6. RGBD图像发布循环
        try:
            print(f"\nStarting RGBD image publishing loop ({loop_hz:.1f}Hz)")
            print("Press Ctrl+C to exit")

            frame_count = 0

            while True:
                loop_start = time.time()

                # 获取两个相机的RGBD图像
                try:
                    # 末端相机 - 只渲染，不发布
                    ee_rgb, ee_depth = env.robot.get_ee_camera_rgbd(publish_ros2=False)
                    # 外部相机 - 只渲染，不发布
                    ext_rgb, ext_depth = env.robot.get_external_camera_rgbd(publish_ros2=False)

                    # 热成像渲染（与末端相机同视角）
                    if thermal_sensor is not None and ee_rgb is not None:
                        try:
                            temperature_map, _ = thermal_sensor.render_thermal_image()
                            
                            # 与 thermal_sensor_example.py 对齐：
                            # 1) 温度图 -> 灰度
                            # 2) RGB -> 灰度（在 blend_with_rgb 内部完成）
                            # 3) 两者按权重融合后发布 mono8
                            thermal_gray = thermal_sensor.temperature_to_grayscale(temperature_map, 0.0, 100.0)
                            # 对热灰度图应用轻微高斯模糊，柔和物体边界
                            import cv2
                            thermal_gray = cv2.GaussianBlur(thermal_gray, (25, 25), 5)
                            rgb_image = thermal_sensor.render_rgb()
                            blended = thermal_sensor.blend_with_rgb(
                                thermal_gray,
                                rgb_image,
                                thermal_weight=0.85
                            )
                            if thermal_publisher is not None:
                                thermal_publisher.publish(blended, stamp_sec=env.robot.sim_time)
                        except Exception as e:
                            print(f"❌ Thermal render FAILED (frame {frame_count}): {e}")
                            import traceback
                            traceback.print_exc()

                    frame_count += 1

                    # 更新图像数据到 ROS2 发布线程
                    if ros2_publisher_thread is not None:
                        try:
                            # 更新末端相机数据
                            if ee_rgb is not None:
                                ee_intrinsics = env.robot.cameras['ee'].intr
                                ros2_publisher_thread.update_image_data(
                                    camera_name='ee_camera',
                                    rgb=ee_rgb,
                                    depth=ee_depth,
                                    intrinsics=ee_intrinsics,
                                    timestamp_sec=env.robot.sim_time
                                )

                            # 更新外部相机数据
                            if ext_rgb is not None:
                                ext_intrinsics = env.robot.cameras['external'].intr
                                ros2_publisher_thread.update_image_data(
                                    camera_name='external_camera',
                                    rgb=ext_rgb,
                                    depth=ext_depth,
                                    intrinsics=ext_intrinsics,
                                    timestamp_sec=env.robot.sim_time
                                )

                        except Exception as e:
                            if frame_count % 100 == 1:
                                print(f"⚠ 图像数据更新失败: {e}")

                    # 如果正在Teaching录制，将所有摄像头图像传递给控制线程
                    if controller.enable_teaching_recording and controller.is_teaching_recording:
                        if controller.trajectory_recorder and controller.trajectory_recorder.is_recording:
                            try:
                                # 准备所有摄像头图像字典
                                camera_images = {}
                                if ee_rgb is not None:
                                    camera_images['ee_rgb'] = ee_rgb.copy()
                                if ee_depth is not None:
                                    camera_images['ee_depth'] = ee_depth.copy()
                                if ext_rgb is not None:
                                    camera_images['external_rgb'] = ext_rgb.copy()
                                if ext_depth is not None:
                                    camera_images['external_depth'] = ext_depth.copy()

                                # 放入队列
                                if camera_images:
                                    controller.camera_image_queue.put_nowait(camera_images)
                            except:
                                pass  # 队列满时跳过

                except Exception as e:
                    print(f"⚠ 图像获取或发布错误: {e}")

                # 控制循环频率到60Hz
                loop_duration = time.time() - loop_start
                target_duration = 1.0 / max(loop_hz, 1e-3)
                sleep_time = target_duration - loop_duration

                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nUser interrupted")
        except EOFError:
            print("\nNon-interactive environment detected, switching to demo mode")
            controller.add_target("random")
            time.sleep(10)
            print("\nDemo completed")

        # 7. 停止控制器
        print("\nStopping controller...")
        controller.stop()

        # 8. 停止 ROS2 发布线程
        if ros2_publisher_thread is not None:
            print("Stopping ROS2 publisher...")
            ros2_publisher_thread.stop()

        # 9. 关闭环境
        print("Closing environment...")
        env.close()

        print("\nTest completed")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="MuJoCo 机械臂仿真接口")
        parser.add_argument("--model", type=str, default=None, help="MuJoCo 场景 XML 路径")
        parser.add_argument("--headless", action="store_true", help="无可视化模式")
        parser.add_argument("--no-ros-control", action="store_true", help="禁用 ROS 控制输入")
        parser.add_argument("--no-rosbag", action="store_true", help="禁用 rosbag 录制")
        parser.add_argument("--no-image-publish", action="store_true", help="禁用 RGBD ROS2 发布")
        parser.add_argument("--no-thermal", action="store_true", help="禁用热成像发布")
        parser.add_argument("--loop-hz", type=float, default=15.0, help="主循环频率 (Hz)")
        args = parser.parse_args()

        test_integrated_trajectory_control(
            model_path=args.model,
            enable_visualization=not args.headless,
            enable_ros_control=not args.no_ros_control,
            enable_rosbag=not args.no_rosbag,
            enable_image_publish=not args.no_image_publish,
            enable_thermal=not args.no_thermal,
            enable_teaching_recording=True,
            loop_hz=args.loop_hz,
        )
    except KeyboardInterrupt:
        print("\nUser interrupted")
    except Exception as e:
        print(f"\nProgram error: {e}")
        import traceback
        traceback.print_exc()
