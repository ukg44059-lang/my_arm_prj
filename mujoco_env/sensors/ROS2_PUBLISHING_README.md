# ROS2图像发布功能

将MuJoCo RGBD传感器的图像通过ROS2 topic发布，用于机器人视觉应用。

## 功能特性

✅ **自动发布RGB-D图像** - 实时发布RGB和深度图像到ROS2 topics
✅ **相机信息发布** - 发布相机内参和外参信息
✅ **多相机支持** - 支持末端执行器相机和外部相机
✅ **可选启用** - 可以选择启用或禁用ROS2发布
✅ **标准ROS2消息** - 使用sensor_msgs/Image和sensor_msgs/CameraInfo

## 安装依赖

```bash
# 安装ROS2 Python包
pip install rclpy sensor_msgs cv_bridge

# 或者使用conda
conda install -c conda-forge rclpy sensor_msgs cv_bridge
```

## 快速开始

### 1. 启用ROS2的基本使用

```python
from robot import Robot

# 创建Robot实例，启用ROS2发布
robot = Robot(model_path, enable_ros2=True)

# 获取图像并自动发布到ROS2
rgb, depth = robot.get_ee_camera_rgbd(publish_ros2=True)
rgb_ext, depth_ext = robot.get_external_camera_rgbd(publish_ros2=True)
```

### 2. 不使用ROS2

```python
# 不启用ROS2
robot = Robot(model_path, enable_ros2=False)

# 只获取图像，不发布
rgb, depth = robot.get_ee_camera_rgbd()
```

### 3. 在仿真循环中发布图像

```python
from robot import Robot
import numpy as np

# 初始化机器人（启用ROS2）
robot = Robot(model_path, enable_ros2=True)
robot.set_control_mode("zero_gravity")

# 主循环
while True:
    # 更新机器人状态
    robot.update_joint_state()
    targets = robot.compute_joint_targets(robot.sim_time)
    robot.apply_joint_control(targets)
    robot.step()

    # 获取并发布图像（每10帧发布一次以提高性能）
    if frame_count % 10 == 0:
        rgb_ee, depth_ee = robot.get_ee_camera_rgbd(publish_ros2=True)
        rgb_ext, depth_ext = robot.get_external_camera_rgbd(publish_ros2=True)

    frame_count += 1

# 清理
robot.cleanup()
```

## 发布的ROS2 Topics

### 末端执行器相机

| Topic | 消息类型 | 描述 |
|-------|---------|------|
| `/ee_camera/rgb/image_raw` | sensor_msgs/Image | RGB图像 (640x480, rgb8) |
| `/ee_camera/depth/image_raw` | sensor_msgs/Image | 深度图像 (640x480, 32FC1) |
| `/ee_camera/rgb/camera_info` | sensor_msgs/CameraInfo | RGB相机内参 |
| `/ee_camera/depth/camera_info` | sensor_msgs/CameraInfo | 深度相机内参 |

### 外部相机

| Topic | 消息类型 | 描述 |
|-------|---------|------|
| `/external_camera/rgb/image_raw` | sensor_msgs/Image | RGB图像 (640x480, rgb8) |
| `/external_camera/depth/image_raw` | sensor_msgs/Image | 深度图像 (640x480, 32FC1) |
| `/external_camera/rgb/camera_info` | sensor_msgs/CameraInfo | RGB相机内参 |
| `/external_camera/depth/camera_info` | sensor_msgs/CameraInfo | 深度相机内参 |

## 查看发布的图像

### 方法1: 使用ROS2命令行工具

```bash
# 列出所有topics
ros2 topic list

# 查看图像topic的频率
ros2 topic hz /ee_camera/rgb/image_raw

# 查看topic的详细信息
ros2 topic info /ee_camera/rgb/image_raw

# 显示图像消息
ros2 topic echo /ee_camera/rgb/image_raw --no-arr
```

### 方法2: 使用RViz2可视化

```bash
# 启动RViz2
rviz2

# 在RViz2中:
# 1. 点击 "Add" -> "By topic"
# 2. 选择 /ee_camera/rgb/image_raw -> Image
# 3. 选择 /ee_camera/depth/image_raw -> Image
```

### 方法3: 使用rqt_image_view

```bash
# 安装rqt_image_view
sudo apt install ros-<distro>-rqt-image-view

# 运行
rqt_image_view
```

## 测试脚本

运行测试脚本来验证ROS2发布功能：

```bash
# 测试ROS2发布
cd mujoco_env/robot
python test_ros2_publishing.py

# 测试不使用ROS2
python test_ros2_publishing.py --no-ros2
```

## API参考

### Robot类

#### 构造函数

```python
Robot(model_path, enable_ros2=False)
```

**参数:**
- `model_path` (str): MuJoCo XML模型路径
- `enable_ros2` (bool): 是否启用ROS2发布，默认False

#### 方法

##### `get_ee_camera_rgbd(publish_ros2=True)`

获取末端执行器相机的RGB-D图像

**参数:**
- `publish_ros2` (bool): 是否发布到ROS2，默认True

**返回:**
- `rgb` (np.ndarray): RGB图像 (H, W, 3), uint8
- `depth` (np.ndarray): 深度图像 (H, W), float32, 单位米

##### `get_external_camera_rgbd(publish_ros2=True)`

获取外部相机的RGB-D图像

**参数:**
- `publish_ros2` (bool): 是否发布到ROS2，默认True

**返回:**
- `rgb` (np.ndarray): RGB图像 (H, W, 3), uint8
- `depth` (np.ndarray): 深度图像 (H, W), float32, 单位米

## 性能优化

### 1. 控制发布频率

不需要每帧都发布图像，可以降低发布频率：

```python
# 每10帧发布一次
if frame_count % 10 == 0:
    rgb, depth = robot.get_ee_camera_rgbd(publish_ros2=True)
```

### 2. 选择性发布

只发布需要的相机：

```python
# 只发布EE相机
rgb_ee, depth_ee = robot.get_ee_camera_rgbd(publish_ros2=True)

# 不发布外部相机
rgb_ext, depth_ext = robot.get_external_camera_rgbd(publish_ros2=False)
```

### 3. 禁用ROS2

如果不需要ROS2，完全禁用以提高性能：

```python
robot = Robot(model_path, enable_ros2=False)
```

## 故障排除

### 问题1: ImportError: No module named 'rclpy'

**解决方案:** 安装ROS2 Python包

```bash
pip install rclpy sensor_msgs cv_bridge
```

### 问题2: 相机图像不显示在RViz2

**检查清单:**
1. 确认topics已发布: `ros2 topic list`
2. 检查topic频率: `ros2 topic hz /ee_camera/rgb/image_raw`
3. 确认RViz2订阅了正确的topic
4. 检查固定帧(Fixed Frame)设置

### 问题3: 图像发布速度慢

**解决方案:**
1. 降低发布频率（每N帧发布一次）
2. 减小图像分辨率（在sensor初始化时修改）
3. 只发布需要的相机

## 集成示例

### 与robot_mujoco_trajectory.py集成

```python
# 在robot_mujoco_trajectory.py中启用ROS2
robot = Robot(MODEL_PATH, enable_ros2=True)

# 在主循环中
while viewer.is_running():
    # ... 机器人控制代码 ...

    # 每30帧发布一次图像
    if step_count % 30 == 0:
        rgb_ee, depth_ee = robot.get_ee_camera_rgbd(publish_ros2=True)
```

### 与强化学习环境集成

```python
from envs import RobotArmEnv

class RobotArmEnvWithROS2(RobotArmEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 启用ROS2
        self.robot.enable_ros2 = True
        self.robot._init_ros2_publishers()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # 发布相机图像
        rgb, depth = self.robot.get_ee_camera_rgbd(publish_ros2=True)

        return obs, reward, terminated, truncated, info
```

## 相关文件

- `rgbd_ros2_publisher.py`: ROS2发布器实现
- `robot.py`: Robot类（集成ROS2支持）
- `test_ros2_publishing.py`: 测试脚本
- `RGBD.py`: RGBD传感器基础类

## 更新日志

### v1.0 (2025-12-23)
- ✅ 初始实现
- ✅ 支持RGB和深度图像发布
- ✅ 支持相机信息发布
- ✅ 集成到Robot类
- ✅ 添加测试脚本和文档
