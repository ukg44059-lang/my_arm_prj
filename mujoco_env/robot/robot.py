"""
Robot Class - Encapsulates all robot initialization and control logic
机器人类 - 封装所有机器人初始化和控制逻辑
"""

import mujoco
from mujoco.renderer import Renderer
import numpy as np
import os
import sys

# 添加 tools 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from obstacle_controller import ObstacleController

# 添加 sensors 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../sensors'))
from RGBD import MuJoCo_RGBD_Sensor

# Optional ROS2 joint state publisher
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
    from ros2_joint_state_publisher import JointStatePublisherROS2, ros2_available
except ImportError:
    JointStatePublisherROS2 = None  # type: ignore
    def ros2_available():
        return False

# 导入控制器
from controller import TorqueController, GravityCompensationController

# ROS2图像发布已禁用


class Robot:
    """
    Robot control class that encapsulates:
    - Model loading and initialization
    - Joint state management (positions, velocities)
    - Torque/motor control
    - Gripper control
    - End-effector tracking
    - Camera management
    - Obstacle control

    机器人控制类，封装：
    - 模型加载和初始化
    - 关节状态管理（位置、速度）
    - 力矩/电机控制
    - 夹爪控制
    - 末端执行器追踪
    - 相机管理
    - 障碍物控制
    """

    def __init__(self, model_path, enable_ros2=False, enable_cameras=True,
                 enable_joint_state_ros2=False, joint_state_topic="/joint_states"):
        """
        Initialize robot with MuJoCo model

        Args:
            model_path: Path to the MuJoCo XML model file
            enable_ros2: Enable classical ROS2 image publishing
            enable_cameras: Initialize cameras in this process (default: True, set to False when using CameraProcessManager)
            enable_joint_state_ros2: Enable ROS2 joint state publishing
            joint_state_topic: ROS2 joint state topic name
        """
        # 死锁调试：启动心跳监控线程
        import threading
        import time
        self._heartbeat_enabled = True
        self._last_step_time = time.time()
        self._step_count = 0
        self._camera_render_count = 0
        self._last_camera_render_time = time.time()
        self._step_lock = threading.Lock()  # 添加步进锁
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self._heartbeat_thread.start()
        
        # Load MuJoCo model and create data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Initialize forward kinematics
        mujoco.mj_forward(self.model, self.data) # pyright: ignore[reportAttributeAccessIssue]

        print(f"✓ MuJoCo model loaded")
        print(f"  - File: {model_path}")
        print(f"  - Actuators: {self.model.nu}")
        print(f"  - DOFs: {self.model.nv}")

        # Joint state management
        self.joint_positions = np.zeros(6)
        self.joint_velocities = np.zeros(6)
        self.joint_targets = np.zeros(6)
        self.num_joints = 6

        # End-effector reference (tools_link is a site, not a body)
        self.ee_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            'tools_link'
        ) # pyright: ignore[reportAttributeAccessIssue]
        print(f"✓ End-effector site ID: {self.ee_site_id}")

        # Torque controller initialization
        self.torque_controller = None
        self._init_torque_controller(model_path)

        # Control parameters
        self.control_mode = "zero_gravity"  # sine_wave, zero_gravity, user_input
        self.gripper_command = 1.0  # Default: open (1.0 = open, -1.0 = close)

        # ROS2设置
        self.enable_ros2 = enable_ros2
        self.enable_cameras = enable_cameras
        self.enable_joint_state_ros2 = enable_joint_state_ros2 and ros2_available()
        self.joint_state_topic = joint_state_topic
        self.joint_state_pub = None
        self.joint_state_pub_count = 0
        self._joint_state_log_last_time = 0.0

        # Obstacle controller - DISABLED (no obstacles in environment)
        self.obstacle_controller = None
        self.obstacles_enabled = False

        # Check if actuatorfrc sensors are available (tau_sensor1..6)
        try:
            self._tau_sensor_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"tau_sensor{i+1}")
                for i in range(self.num_joints)
            ]
            self._has_tau_sensors = all(sid >= 0 for sid in self._tau_sensor_ids)
        except Exception:
            self._has_tau_sensors = False
        if self._has_tau_sensors:
            print("✓ Torque sensors detected: tau_sensor1..6 will be used for effort data")
        else:
            print("ℹ Torque sensors not found, falling back to qfrc_actuator for effort data")

        # JointState ROS2 publisher (optional)
        if self.enable_joint_state_ros2 and JointStatePublisherROS2 is not None:
            try:
                self.joint_state_pub = JointStatePublisherROS2(
                    topic=self.joint_state_topic,
                    joint_names=[f"joint{i+1}" for i in range(self.num_joints)] + ["gripper"],
                    node_name="mujoco_joint_state_publisher"
                )
                print(f"✓ JointState ROS2 publisher enabled -> {self.joint_state_topic}")
            except Exception as e:
                print(f"⚠ Failed to initialize JointState ROS2 publisher: {e}")
                self.joint_state_pub = None
        elif self.enable_joint_state_ros2:
            print("⚠ JointState ROS2 publisher not available (rclpy not found)")

        # Camera and sensor management (only if enabled)
        self.cameras = {}
        self.renderers = {}  # Store renderers for cleanup
        if enable_cameras:
            self._init_cameras()
        else:
            print("ℹ Camera initialization disabled (using external process manager)")

        # Timing
        self.sim_time = 0.0
        self.dt = self.model.opt.timestep

    def _init_torque_controller(self, model_path):
        """Initialize torque controller with gravity compensation"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            torque_ctrl_model_path = os.path.join(
                base_dir,
                "../robot_model/robot/urdf/arm620_rq.urdf"
            )

            # 为每个关节设置不同的PID参数
            # 关节顺序: [Joint1, Joint2, Joint3, Joint4, Joint5, Joint6]
            # 靠近基座的关节需要更大的增益，末端关节需要更小的增益
            kp_gains = np.array([120.0, 150.0, 120.0, 40.0, 40.0, 40.0])   # Proportional gains
            kd_gains = np.array([0, 0, 0, 0, 0, 0])        # Derivative gains
            ki_gains = np.array([1, 1, 1, 0.2, 0.2, 0.2])    # Integral gains

            self.torque_controller = TorqueController(
                model_path=torque_ctrl_model_path,
                kp=kp_gains,
                kd=kd_gains,
                ki=ki_gains,
                use_gravity_compensation=True  # Enable gravity compensation
            )
            print("✓ Torque Controller initialized with per-joint PID gains")
        except Exception as e:
            print(f"⚠ Failed to initialize torque controller: {e}")
            self.torque_controller = None

    def _init_obstacle_controller(self):
        """Initialize obstacle controller with default parameters"""
        self.obstacle_controller.set_frequency(0, 1.5)
        self.obstacle_controller.set_frequency(1, 1.5)
        self.obstacle_controller.set_amplitude(0, 0.8)
        self.obstacle_controller.set_amplitude(1, 0.8)
        print("✓ Obstacle controller initialized")
        self.obstacle_controller.print_status()

    def _init_cameras(self):
        """Initialize RGBD cameras (end-effector and external) with automatic ROS2 integration"""
        # Initialize ROS2 cameras dict
        self.ros2_cameras = {}

        try:
            ee_rgb_renderer = Renderer(
                self.model,
                height=480,
                width=640
            )
            ee_depth_renderer = Renderer(
                self.model,
                height=480,
                width=640
            )

            # Store renderers for cleanup
            self.renderers['ee_rgb'] = ee_rgb_renderer
            self.renderers['ee_depth'] = ee_depth_renderer

            # Create base RGBD sensor
            ee_sensor = MuJoCo_RGBD_Sensor(
                model=self.model,
                data=self.data,
                rgb_renderer=ee_rgb_renderer,
                depth_renderer=ee_depth_renderer,
                cam_name="rgbd_camera_ee",
                fps=30
            )
            self.cameras['ee'] = ee_sensor

            # Wrap with ROS2 if enabled
            if self.enable_ros2:
                try:
                    sys.path.append(os.path.join(os.path.dirname(__file__), '../sensors'))
                    from rgbd_ros2_publisher import RGBD_Sensor_With_ROS2

                    self.ros2_cameras['ee'] = RGBD_Sensor_With_ROS2(
                        rgbd_sensor=ee_sensor,
                        camera_name="ee_camera",
                        enable_ros2=True
                    )
                except Exception as e:
                    print(f"⚠ Failed to enable ROS2 for EE camera: {e}")

            ext_rgb_renderer = Renderer(
                self.model,
                height=480,
                width=640
            )
            ext_depth_renderer = Renderer(
                self.model,
                height=480,
                width=640
            )

            # Store renderers for cleanup
            self.renderers['ext_rgb'] = ext_rgb_renderer
            self.renderers['ext_depth'] = ext_depth_renderer

            # Create base RGBD sensor
            ext_sensor = MuJoCo_RGBD_Sensor(
                model=self.model,
                data=self.data,
                rgb_renderer=ext_rgb_renderer,
                depth_renderer=ext_depth_renderer,
                cam_name="rgbd_camera_external",
                fps=30
            )
            self.cameras['external'] = ext_sensor

            # Wrap with ROS2 if enabled
            if self.enable_ros2:
                try:
                    from rgbd_ros2_publisher import RGBD_Sensor_With_ROS2

                    self.ros2_cameras['external'] = RGBD_Sensor_With_ROS2(
                        rgbd_sensor=ext_sensor,
                        camera_name="external_camera",
                        enable_ros2=True
                    )
                except Exception as e:
                    print(f"⚠ Failed to enable ROS2 for external camera: {e}")

            print("✓ RGBD cameras initialized")
            print(f"  - EE camera: rgbd_camera_ee")
            print(f"  - External camera: rgbd_camera_external")
            print("  - 经典ROS2图像发布已启用" if self.enable_ros2 else "  - ROS2图像发布已禁用")

        except Exception as e:
            print(f"⚠ Failed to initialize cameras: {e}")

    # ========== Joint Control Methods ==========

    def update_joint_state(self):
        """Read current joint positions and velocities from MuJoCo data"""
        self.joint_positions = np.array([
            self.data.joint(f"joint{i+1}").qpos[0] for i in range(self.num_joints)
        ])
        self.joint_velocities = np.array([
            self.data.joint(f"joint{i+1}").qvel[0] for i in range(self.num_joints)
        ])

    def get_joint_positions(self):
        """Get current joint positions"""
        return self.joint_positions.copy()

    def get_joint_velocities(self):
        """Get current joint velocities"""
        return self.joint_velocities.copy()

    def get_joint_feedback(self):
        """Get joint feedback (actual positions after stepping)"""
        return np.array([
            self.data.joint(f"joint{i+1}").qpos[0] for i in range(self.num_joints)
        ])

    def publish_joint_state_ros2(self, stamp_sec: float | None = None, skip_spin: bool = False):
        """
        Publish current joint state via ROS2 if enabled. Includes 6 arm joints + 1 gripper joint.
        
        Args:
            stamp_sec: Timestamp in seconds
            skip_spin: If True, skip spin_once call for higher frequency publishing
        """
        if self.joint_state_pub is None:
            return
        try:
            # Get arm joint data
            arm_positions = self.get_joint_positions()
            arm_velocities = self.get_joint_velocities()
            if self._has_tau_sensors:
                arm_efforts = np.array([
                    self.data.sensordata[self.model.sensor_adr[sid]]
                    for sid in self._tau_sensor_ids
                ])
            else:
                arm_efforts = self.data.qfrc_actuator[:self.num_joints].copy()
            
            # Add gripper data
            gripper_position = np.array([self.gripper_command])  # gripper command as position
            gripper_velocity = np.array([0.0])  # gripper doesn't have velocity feedback
            gripper_effort = np.array([self.data.qfrc_actuator[6] if len(self.data.qfrc_actuator) > 6 else 0.0])
            
            # Combine arm + gripper data
            positions = np.concatenate([arm_positions, gripper_position])
            velocities = np.concatenate([arm_velocities, gripper_velocity])
            efforts = np.concatenate([arm_efforts, gripper_effort])
            
            self.joint_state_pub.publish(
                positions=positions,
                velocities=velocities,
                efforts=efforts,
                stamp_sec=stamp_sec
            )
            
            # Only spin occasionally for high-frequency publishing
            if not skip_spin:
                self.joint_state_pub.spin_once(timeout_sec=0.0)

            self.joint_state_pub_count += 1
        except Exception as e:
            print(f"⚠ JointState publish failed: {e}")

    def set_control_mode(self, mode):
        """
        Set joint control mode

        Args:
            mode: "sine_wave", "zero_gravity", or "user_input"
        """
        if mode not in ["sine_wave", "zero_gravity", "user_input"]:
            print(f"⚠ Unknown control mode: {mode}")
            return
        self.control_mode = mode
        print(f"✓ Control mode set to: {mode}")

    def compute_joint_targets(self, sim_time=None):
        """
        Compute target joint angles based on current control mode

        Args:
            sim_time: Simulation time (optional, uses self.sim_time if None)

        Returns:
            numpy array of 6 target joint angles
        """
        if sim_time is None:
            sim_time = self.sim_time

        if self.control_mode == "sine_wave":
            # Sine wave motion
            targets = np.array([
                0.3 * np.sin(2 * np.pi * 0.2 * sim_time),
                0.2 * np.sin(2 * np.pi * 0.15 * sim_time),
                0.15 * np.sin(2 * np.pi * 0.25 * sim_time),
                0.0,
                0.0,
                0.0
            ])
        elif self.control_mode == "zero_gravity":
            # Zero gravity mode: maintain current position
            targets = self.joint_positions.copy()
        else:  # user_input or other
            # Default: zero position
            targets = np.zeros(self.num_joints)

        self.joint_targets = targets
        return targets

    def apply_joint_control(self, joint_targets=None):
        """
        Apply joint control using torque controller or direct control

        Args:
            joint_targets: Target joint angles (optional, uses computed targets if None)
        """
        if joint_targets is None:
            joint_targets = self.joint_targets

        if self.torque_controller is not None:
            # PID + gravity compensation
            tau_cmd = self.torque_controller.compute_torque(
                self.joint_positions,
                self.joint_velocities,
                joint_targets,
                dq_target=np.zeros(self.num_joints),
                dt=self.dt
            )
            self.data.ctrl[:6] = tau_cmd
        else:
            # Direct position control fallback
            self.data.ctrl[:6] = joint_targets

    # ========== Gripper Control Methods ==========

    def set_gripper_command(self, command):
        """
        Set gripper command

        Args:
            command: 0~255 range (0=fully closed, 255=fully open)
        """
        self.gripper_command = np.clip(255.0 - command, 0.0, 255.0)
        self.data.ctrl[6] = self.gripper_command

    def open_gripper(self):
        """Open gripper"""
        self.set_gripper_command(255.0)

    def close_gripper(self):
        """Close gripper"""
        self.set_gripper_command(0.0)

    def neutral_gripper(self):
        """Set gripper to neutral position"""
        self.set_gripper_command(127.5)

    # ========== Motion Control Methods ==========

    def get_ee_position(self):
        """Get end-effector position"""
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_ee_orientation(self):
        """Get end-effector orientation (quaternion [w, x, y, z])"""
        # Sites don't have quaternions directly, convert from rotation matrix
        mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return quat.copy()

    def get_ee_pose(self):
        """Get end-effector pose (position and quaternion)"""
        return self.get_ee_position(), self.get_ee_orientation()

    def get_ee_velocity(self):
        """
        Get end-effector linear velocity

        Returns:
            numpy.ndarray: Linear velocity [vx, vy, vz]
        """
        # For sites, compute velocity using Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        # Linear velocity = Jacobian * joint velocities
        vel = jacp @ self.data.qvel
        return vel.copy()

    def get_ee_angular_velocity(self):
        """
        Get end-effector angular velocity

        Returns:
            numpy.ndarray: Angular velocity [wx, wy, wz]
        """
        # For sites, compute angular velocity using Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        # Angular velocity = Jacobian * joint velocities
        ang_vel = jacr @ self.data.qvel
        return ang_vel.copy()

    def get_obstacle_position(self, obstacle_name):
        """
        Get obstacle position

        Args:
            obstacle_name: Obstacle body name (e.g., 'obstacle1', 'obstacle2')

        Returns:
            numpy.ndarray: Position [x, y, z], or None if obstacle not found
        """
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            obstacle_name
        ) # pyright: ignore[reportAttributeAccessIssue]

        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return None

    def get_obstacle_velocity(self, obstacle_name):
        """
        Get obstacle linear velocity

        Args:
            obstacle_name: Obstacle body name (e.g., 'obstacle1', 'obstacle2')

        Returns:
            numpy.ndarray: Linear velocity [vx, vy, vz], or None if obstacle not found
        """
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            obstacle_name
        ) # pyright: ignore[reportAttributeAccessIssue]

        if body_id >= 0:
            return self.data.cvel[body_id][:3].copy()
        return None

    def get_all_obstacles_info(self):
        """
        Get complete information for all obstacles

        Returns:
            dict: Obstacle information {obstacle_name: {'position': [x,y,z], 'velocity': [vx,vy,vz]}}
        """
        obstacles_info = {}

        for obstacle_name in ['obstacle1', 'obstacle2']:
            pos = self.get_obstacle_position(obstacle_name)
            vel = self.get_obstacle_velocity(obstacle_name)

            if pos is not None:
                obstacles_info[obstacle_name] = {
                    'position': pos,
                    'velocity': vel if vel is not None else np.zeros(3)
                }

        return obstacles_info

    def get_target_position(self):
        """
        Get target position

        Returns:
            numpy.ndarray: Position [x, y, z], or None if target not found
        """
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            'target_body'
        ) # pyright: ignore[reportAttributeAccessIssue]

        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return None

    def get_target_orientation(self):
        """
        Get target orientation (quaternion)

        Returns:
            numpy.ndarray: Quaternion [w, x, y, z], or None if target not found
        """
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            'target_body'
        ) # pyright: ignore[reportAttributeAccessIssue]

        if body_id >= 0:
            return self.data.xquat[body_id].copy()
        return None

    def get_target_velocity(self):
        """
        Get target linear velocity

        Returns:
            numpy.ndarray: Linear velocity [vx, vy, vz], or None if target not found
        """
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            'target_body'
        ) # pyright: ignore[reportAttributeAccessIssue]

        if body_id >= 0:
            return self.data.cvel[body_id][:3].copy()
        return None

    def get_target_pose(self):
        """
        Get target pose (position and quaternion)

        Returns:
            tuple: (position, quaternion)
        """
        return self.get_target_position(), self.get_target_orientation()

    def update_obstacles(self):
        """Update obstacle control signals - DISABLED"""
        pass

    def reset_obstacles(self):
        """Reset all obstacle control signals to zero - DISABLED"""
        pass

    def set_obstacle_frequency(self, obstacle_idx, frequency):
        """Set obstacle oscillation frequency - DISABLED"""
        pass

    def set_obstacle_amplitude(self, obstacle_idx, amplitude):
        """Set obstacle motion amplitude - DISABLED"""
        pass

    def enable_obstacles(self):
        """启用所有障碍物运动 - DISABLED"""
        self.obstacles_enabled = True

    def disable_obstacles(self):
        """禁用所有障碍物运动 - DISABLED"""
        self.obstacles_enabled = False

    def toggle_obstacles(self):
        """切换障碍物运动状态 - DISABLED"""
        self.obstacles_enabled = not self.obstacles_enabled
        return self.obstacles_enabled

    def is_obstacles_enabled(self):
        """返回当前障碍物启用状态"""
        return self.obstacles_enabled

    # ========== Collision Detection Methods ==========

    def get_minimum_distance_to_obstacles(self, obstacle_names=None):
        """
        计算机械臂与障碍物之间的最近距离
        Calculate the minimum distance between the robot arm and obstacles

        Args:
            obstacle_names: List of obstacle body names to consider
                          If None, considers all bodies with 'obstacle' in name

        Returns:
            dict: {
                'min_distance': float,  # 最小距离 (米)
                'obstacle_name': str,   # 最近的障碍物名称
                'robot_geom': str,      # 最近的机械臂几何体名称
                'obstacle_geom': str,   # 最近的障碍物几何体名称
                'robot_point': np.ndarray,   # 机械臂上最近点的位置
                'obstacle_point': np.ndarray # 障碍物上最近点的位置
            }
        """
        if obstacle_names is None:
            # 自动查找所有障碍物
            obstacle_names = []
            for i in range(self.model.nbody):
                body_name = self.model.body(i).name
                if 'obstacle' in body_name.lower():
                    obstacle_names.append(body_name)

        if len(obstacle_names) == 0:
            return {
                'min_distance': float('inf'),
                'obstacle_name': None,
                'robot_geom': None,
                'obstacle_geom': None,
                'robot_point': None,
                'obstacle_point': None
            }

        # 获取障碍物的body IDs和geom IDs
        obstacle_info = {}
        for obs_name in obstacle_names:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                obs_name
            ) # pyright: ignore[reportAttributeAccessIssue]

            if body_id >= 0:
                # 找到属于该body的所有geom
                geom_ids = []
                for geom_id in range(self.model.ngeom):
                    if self.model.geom_bodyid[geom_id] == body_id:
                        geom_ids.append(geom_id)

                if len(geom_ids) > 0:
                    obstacle_info[obs_name] = {
                        'body_id': body_id,
                        'geom_ids': geom_ids
                    }

        # 获取机械臂的geom IDs (排除基座、障碍物等)
        robot_geom_ids = []
        exclude_keywords = ['floor', 'table', 'obstacle', 'target', 'gripper', 'base']

        for geom_id in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[geom_id]
            body_name = self.model.body(body_id).name.lower()
            geom_name = self.model.geom(geom_id).name.lower()

            # 检查是否应该排除
            should_exclude = False
            for keyword in exclude_keywords:
                if keyword in body_name or keyword in geom_name:
                    should_exclude = True
                    break

            # 只包含机械臂连杆 (link开头的)
            if not should_exclude and 'link' in body_name:
                robot_geom_ids.append(geom_id)

        # 遍历所有机械臂geom和障碍物geom对，计算最近距离
        min_distance = float('inf')
        min_obstacle_name = None
        min_robot_geom_name = None
        min_obstacle_geom_name = None
        min_robot_point = None
        min_obstacle_point = None

        for robot_geom_id in robot_geom_ids:
            for obs_name, obs_data in obstacle_info.items():
                for obs_geom_id in obs_data['geom_ids']:
                    # 计算两个geom之间的距离
                    distance_info = self._compute_geom_distance(robot_geom_id, obs_geom_id)

                    if distance_info['distance'] < min_distance:
                        min_distance = distance_info['distance']
                        min_obstacle_name = obs_name
                        min_robot_geom_name = self.model.geom(robot_geom_id).name or f"geom_{robot_geom_id}"
                        min_obstacle_geom_name = self.model.geom(obs_geom_id).name or f"geom_{obs_geom_id}"
                        min_robot_point = distance_info['point1']
                        min_obstacle_point = distance_info['point2']

        return {
            'min_distance': min_distance,
            'obstacle_name': min_obstacle_name,
            'robot_geom': min_robot_geom_name,
            'obstacle_geom': min_obstacle_geom_name,
            'robot_point': min_robot_point,
            'obstacle_point': min_obstacle_point
        }

    def _compute_geom_distance(self, geom1_id, geom2_id):
        """
        计算两个几何体之间的最近距离
        Compute the minimum distance between two geometries

        Args:
            geom1_id: 第一个几何体ID
            geom2_id: 第二个几何体ID

        Returns:
            dict: {
                'distance': float,        # 距离
                'point1': np.ndarray,     # geom1上的最近点
                'point2': np.ndarray      # geom2上的最近点
            }
        """
        # 使用MuJoCo的碰撞检测来获取距离信息
        # 首先检查接触列表中是否有这对geom
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if ((contact.geom1 == geom1_id and contact.geom2 == geom2_id) or
                (contact.geom1 == geom2_id and contact.geom2 == geom1_id)):

                # 找到了接触/距离信息
                # contact.dist: 正值表示分离距离，负值表示穿透深度
                distance = contact.dist

                # contact.pos 是接触点位置
                contact_pos = contact.pos.copy()

                # contact.frame 前3个元素是接触法向量
                normal = contact.frame[:3].copy()

                # 计算两个最近点
                if contact.geom1 == geom1_id:
                    point1 = contact_pos - normal * distance / 2
                    point2 = contact_pos + normal * distance / 2
                else:
                    point1 = contact_pos + normal * distance / 2
                    point2 = contact_pos - normal * distance / 2

                return {
                    'distance': abs(distance),
                    'point1': point1,
                    'point2': point2
                }

        # 如果没有在接触列表中找到，使用几何体中心点距离估算
        geom1_pos = self.data.geom_xpos[geom1_id]
        geom2_pos = self.data.geom_xpos[geom2_id]

        center_distance = np.linalg.norm(geom1_pos - geom2_pos)

        # 减去两个几何体的大致半径
        geom1_size = np.max(self.model.geom_size[geom1_id])
        geom2_size = np.max(self.model.geom_size[geom2_id])

        approximate_distance = max(0.0, center_distance - geom1_size - geom2_size)

        # 计算最近点（简化版本：沿连线方向）
        direction = (geom2_pos - geom1_pos) / (center_distance + 1e-8)
        point1 = geom1_pos + direction * geom1_size
        point2 = geom2_pos - direction * geom2_size

        return {
            'distance': approximate_distance,
            'point1': point1,
            'point2': point2
        }

    def check_collision(self, exclude_bodies=None):
        """
        Check if robot is in collision with obstacles or environment

        Args:
            exclude_bodies: List of body names to exclude from collision detection
                          (e.g., ['floor', 'table'] for self-collision only)

        Returns:
            bool: True if collision detected, False otherwise
        """
        if exclude_bodies is None:
            exclude_bodies = []

        # Get body IDs to exclude
        exclude_body_ids = []
        for body_name in exclude_bodies:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name
            ) # pyright: ignore[reportAttributeAccessIssue]
            if body_id >= 0:
                exclude_body_ids.append(body_id)

        # Check all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get geom IDs involved in contact
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            # Get body IDs from geom IDs
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]

            # Skip if either body should be excluded
            if body1_id in exclude_body_ids or body2_id in exclude_body_ids:
                continue

            # Collision detected
            return True

        return False

    def check_obstacle_collision(self, obstacle_names=None):
        """
        Check if robot is colliding with specific obstacles

        Args:
            obstacle_names: List of obstacle body names to check
                          If None, checks all bodies with 'obstacle' in name

        Returns:
            collision_info: Dict with keys:
                - 'collision': bool, True if collision detected
                - 'obstacle_name': str, name of obstacle in collision (or None)
                - 'contact_force': float, magnitude of contact force
        """
        if obstacle_names is None:
            # Find all obstacle bodies automatically
            obstacle_names = []
            for i in range(self.model.nbody):
                body_name = self.model.body(i).name
                if 'obstacle' in body_name.lower():
                    obstacle_names.append(body_name)

        # Get obstacle body IDs
        obstacle_body_ids = {}
        for obs_name in obstacle_names:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                obs_name
            ) # pyright: ignore[reportAttributeAccessIssue]
            if body_id >= 0:
                obstacle_body_ids[body_id] = obs_name

        # Check contacts
        max_force = 0.0
        collision_obstacle = None

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            geom1_id = contact.geom1
            geom2_id = contact.geom2

            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]

            # Check if either body is an obstacle
            if body1_id in obstacle_body_ids or body2_id in obstacle_body_ids:
                # Calculate contact force magnitude from constraint forces
                force = 0.0
                if i < self.data.nefc and i < len(self.data.efc_force):
                    force = abs(self.data.efc_force[i])

                if force > max_force:
                    max_force = force
                    if body1_id in obstacle_body_ids:
                        collision_obstacle = obstacle_body_ids[body1_id]
                    else:
                        collision_obstacle = obstacle_body_ids[body2_id]

        return {
            'collision': collision_obstacle is not None,
            'obstacle_name': collision_obstacle,
            'contact_force': max_force
        }

    def get_contact_info(self):
        """
        Get detailed information about all active contacts

        Returns:
            list of dict: Each dict contains:
                - 'geom1_name': str
                - 'geom2_name': str
                - 'body1_name': str
                - 'body2_name': str
                - 'position': np.array (contact position)
                - 'force': np.array (contact force, computed from contact constraints)
                - 'dist': float (penetration distance)
        """
        contacts = []

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get geom and body information
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]

            # Get names (use ID as fallback if name is empty)
            geom1_name = self.model.geom(geom1_id).name or f"geom_{geom1_id}"
            geom2_name = self.model.geom(geom2_id).name or f"geom_{geom2_id}"
            body1_name = self.model.body(body1_id).name or f"body_{body1_id}"
            body2_name = self.model.body(body2_id).name or f"body_{body2_id}"

            # Compute contact force from constraint forces
            # MuJoCo stores contact forces in data.efc_force
            # We need to find the corresponding constraint for this contact
            contact_force = np.zeros(3)
            if i < self.data.nefc:
                # Get the constraint force magnitude
                force_mag = self.data.efc_force[i] if i < len(self.data.efc_force) else 0.0
                # Approximate force direction using contact normal
                contact_force = contact.frame[:3] * force_mag

            contact_info = {
                'geom1_name': geom1_name,
                'geom2_name': geom2_name,
                'body1_name': body1_name,
                'body2_name': body2_name,
                'position': contact.pos.copy(),
                'force': contact_force,
                'dist': contact.dist
            }

            contacts.append(contact_info)

        return contacts

    # ========== Deadlock Debugging Methods ==========
    
    def _heartbeat_monitor(self):
        """心跳监控线程：每1秒打印一次各线程状态"""
        import time
        import threading
        
        last_step_count = 0
        last_camera_count = 0
        step_stall_count = 0
        camera_stall_count = 0
        
        while self._heartbeat_enabled:
            time.sleep(1.0)
            
            current_time = time.time()
            current_step_count = self._step_count
            current_camera_count = self._camera_render_count
            
            # 检查 step 是否在运行
            if current_step_count == last_step_count:
                step_stall_count += 1
                step_status = f"⚠ STALLED ({step_stall_count}s)"
            else:
                step_stall_count = 0
                step_status = "✓ RUNNING"
            
            # 检查相机渲染是否在运行
            if current_camera_count == last_camera_count:
                camera_stall_count += 1
                camera_status = f"⚠ STALLED ({camera_stall_count}s)"
            else:
                camera_stall_count = 0
                camera_status = "✓ RUNNING"
            
            pass
            
            # 如果停滞超过5秒，打印详细线程信息
            if step_stall_count >= 5 or camera_stall_count >= 5:
                print(f"[DEADLOCK DETECTED]")
                if step_stall_count >= 5:
                    print(f"  ⚠ Step stalled for {step_stall_count}s")
                if camera_stall_count >= 5:
                    print(f"  ⚠ Camera stalled for {camera_stall_count}s")
                print(f"  Active threads: {threading.active_count()}")
                for thread in threading.enumerate():
                    print(f"    - {thread.name} (daemon={thread.daemon}, alive={thread.is_alive()})")
            
            last_step_count = current_step_count
            last_camera_count = current_camera_count

    # ========== Simulation Methods ==========

    def step(self):
        """Execute one simulation step"""
        import time
        
        lock_start = time.time()
        self._last_step_time = time.time()
        self._step_count += 1
        
        # 使用锁保护 mj_step，避免与 ROS2 线程冲突
        acquired = self._step_lock.acquire(timeout=2.0)
        
        if not acquired:
            print(f"⚠ [DEADLOCK] Failed to acquire step lock after 2s at step {self._step_count}")
            return
        
        try:
            step_start = time.time()
            lock_wait = step_start - lock_start
            
            if lock_wait > 0.1:
                print(f"⚠ [SLOW LOCK] Waited {lock_wait*1000:.1f}ms for step lock")
            
            mujoco.mj_step(self.model, self.data) # pyright: ignore[reportAttributeAccessIssue]
            step_duration = time.time() - step_start
            
            if step_duration > 0.1:
                print(f"⚠ [SLOW STEP] mj_step took {step_duration*1000:.1f}ms at t={self.sim_time:.2f}s")
        finally:
            self._step_lock.release()
        
        self.sim_time += self.dt

    def reset_simulation(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data) # pyright: ignore[reportAttributeAccessIssue]
        self.sim_time = 0.0
        self.joint_positions = np.zeros(6)
        self.joint_velocities = np.zeros(6)
        self.joint_targets = np.zeros(6)
        self.reset_obstacles()

    def forward_kinematics(self):
        """Update forward kinematics"""
        mujoco.mj_forward(self.model, self.data) # pyright: ignore[reportAttributeAccessIssue]

    # ========== Camera Methods ==========

    def get_ee_camera_rgbd(self, publish_ros2=True):
        """
        Get RGBD data from end-effector camera
        
        Args:
            publish_ros2: Whether to publish to ROS2 (if enabled)
        Returns:
            rgb, depth: RGB and depth images
        """
        import time
        self._camera_render_count += 1
        self._last_camera_render_time = time.time()
        
        if 'ee' in self.cameras:
            return self.cameras['ee'].render_rgbd()
        elif 'ee' in self.ros2_cameras:
            return self.ros2_cameras['ee'].render_and_publish(publish_to_ros2=publish_ros2)
        return None, None

    def get_external_camera_rgbd(self, publish_ros2=True):
        """
        Get RGBD data from external camera
        
        Args:
            publish_ros2: Whether to publish to ROS2 (if enabled)
        Returns:
            rgb, depth: RGB and depth images
        """
        import time
        self._camera_render_count += 1
        self._last_camera_render_time = time.time()
        
        if 'external' in self.cameras:
            return self.cameras['external'].render_rgbd()
        elif 'external' in self.ros2_cameras:
            return self.ros2_cameras['external'].render_and_publish(publish_to_ros2=publish_ros2)
        return None, None

    def display_ee_camera(self, rgb=None, depth=None):
        """Display end-effector camera images"""
        if 'ee' not in self.cameras:
            return

        if rgb is None or depth is None:
            rgb, depth = self.get_ee_camera_rgbd()

        if rgb is not None:
            self.cameras['ee'].display_rgb(rgb, window_name="EE Camera - RGB")
        if depth is not None:
            self.cameras['ee'].display_depth(depth, window_name="EE Camera - Depth")

    def display_external_camera(self, rgb=None, depth=None):
        """Display external camera images"""
        if 'external' not in self.cameras:
            return

        if rgb is None or depth is None:
            rgb, depth = self.get_external_camera_rgbd()

        if rgb is not None:
            self.cameras['external'].display_rgb(rgb, window_name="External Camera - RGB")
        if depth is not None:
            self.cameras['external'].display_depth(depth, window_name="External Camera - Depth")

    # ========== Status Methods ==========

    def print_status(self):
        """Print current robot status"""
        print("\n" + "="*60)
        print("ROBOT STATUS")
        print("="*60)
        print(f"Simulation time: {self.sim_time:.3f}s")
        print(f"Control mode: {self.control_mode}")
        print(f"\nJoint Positions: {self.joint_positions}")
        print(f"Joint Velocities: {self.joint_velocities}")
        print(f"Joint Targets: {self.joint_targets}")
        print(f"\nEE Position: {self.get_ee_position()}")
        print(f"EE Orientation (quat): {self.get_ee_orientation()}")
        print(f"\nGripper Command: {self.gripper_command:.2f}")
        print("="*60 + "\n")

    def get_status_dict(self):
        """Get robot status as dictionary"""
        return {
            'sim_time': self.sim_time,
            'control_mode': self.control_mode,
            'joint_positions': self.joint_positions.copy(),
            'joint_velocities': self.joint_velocities.copy(),
            'joint_targets': self.joint_targets.copy(),
            'ee_position': self.get_ee_position(),
            'ee_orientation': self.get_ee_orientation(),
            'gripper_command': self.gripper_command,
            'num_joints': self.num_joints
        }

    # ========== Cleanup Methods ==========

    def close_cameras(self):
        """Close all camera windows, renderers, and ROS2 publishers"""
        try:
            # ROS2图像发布器清理
            
            # Close ROS2 publishers first (如果启用了传统发布器)
            if self.enable_ros2 and hasattr(self, 'ros2_cameras'):
                for camera_name, ros2_cam in self.ros2_cameras.items():
                    try:
                        ros2_cam.shutdown()
                    except:
                        pass
                print("✓ ROS2 publishers closed")

            # Close all OpenCV windows
            from RGBD import MuJoCo_RGBD_Sensor
            MuJoCo_RGBD_Sensor.close_all_windows()

            # Close all renderers
            for renderer in self.renderers.values():
                if renderer is not None:
                    try:
                        renderer.close()
                    except:
                        pass

            print("✓ All cameras and renderers closed")
        except Exception as e:
            print(f"⚠ Error closing cameras: {e}")

    def cleanup(self):
        """Cleanup all resources (cameras, renderers, etc.)"""
        print("[Robot] Cleaning up resources...")
        self.close_cameras()
        if self.joint_state_pub is not None:
            try:
                self.joint_state_pub.shutdown()
                print("✓ JointState ROS2 publisher shutdown")
            except Exception as e:
                print(f"⚠ Error shutting down JointState publisher: {e}")
        print("✓ Robot cleanup complete")

    # ============================================================
    # PID Controller Configuration Methods
    # ============================================================

    def load_pid_config(self, config_path: str):
        """
        从配置文件加载PID参数

        Args:
            config_path: 配置文件路径 (.yaml, .yml, 或 .json)

        Example:
            robot.load_pid_config("config/pid_custom.yaml")
        """
        if self.torque_controller is None:
            print("⚠ Torque controller not initialized, cannot load PID config")
            return

        self.torque_controller.load_config_from_file(config_path)

    def save_pid_config(self, file_path: str):
        """
        保存当前PID参数到配置文件

        Args:
            file_path: 保存路径 (.yaml, .yml, 或 .json)

        Example:
            robot.save_pid_config("config/pid_current.yaml")
        """
        if self.torque_controller is None:
            print("⚠ Torque controller not initialized, cannot save PID config")
            return

        joint_names = [f"Joint{i+1}" for i in range(6)]
        self.torque_controller.save_config_to_file(file_path, joint_names)

    def set_joint_pid(self, joint_index: int, kp: float = None, kd: float = None, ki: float = None):
        """
        设置单个关节的PID增益

        Args:
            joint_index: 关节索引 (0-5)
            kp: 比例增益
            kd: 微分增益
            ki: 积分增益

        Example:
            robot.set_joint_pid(0, kp=1000.0, kd=50.0)  # 设置Joint1的增益
        """
        if self.torque_controller is None:
            print("⚠ Torque controller not initialized")
            return

        self.torque_controller.set_joint_gains(joint_index, kp, kd, ki)

    def get_joint_pid(self, joint_index: int) -> dict:
        """
        获取单个关节的PID增益

        Args:
            joint_index: 关节索引 (0-5)

        Returns:
            dict: {'kp': float, 'kd': float, 'ki': float}

        Example:
            gains = robot.get_joint_pid(0)
            print(f"Joint1 Kp: {gains['kp']}")
        """
        if self.torque_controller is None:
            print("⚠ Torque controller not initialized")
            return {'kp': 0.0, 'kd': 0.0, 'ki': 0.0}

        return self.torque_controller.get_joint_gains(joint_index)

    def get_all_pid_gains(self) -> dict:
        """
        获取所有关节的PID增益

        Returns:
            dict: 包含所有关节PID参数的字典

        Example:
            all_gains = robot.get_all_pid_gains()
            for joint in all_gains['joints']:
                print(f"Joint{joint['index']}: Kp={joint['kp']}, Kd={joint['kd']}, Ki={joint['ki']}")
        """
        if self.torque_controller is None:
            print("⚠ Torque controller not initialized")
            return {'joints': []}

        return self.torque_controller.get_all_gains()

    def set_all_pid_gains(self, kp=None, kd=None, ki=None):
        """
        设置所有关节的PID增益（统一或独立）

        Args:
            kp: 比例增益 (可以是标量或数组[6,])
            kd: 微分增益 (可以是标量或数组[6,])
            ki: 积分增益 (可以是标量或数组[6,])

        Example:
            # 统一设置所有关节
            robot.set_all_pid_gains(kp=500.0, kd=20.0, ki=0.0)

            # 独立设置每个关节
            robot.set_all_pid_gains(
                kp=[800, 800, 800, 500, 500, 500],
                kd=[20, 40, 30, 10, 10, 10]
            )
        """
        if self.torque_controller is None:
            print("⚠ Torque controller not initialized")
            return

        self.torque_controller.set_gains(kp, kd, ki)

    # ========== ROS2图像发布已禁用 ==========

