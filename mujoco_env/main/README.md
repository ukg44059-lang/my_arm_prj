# main 使用说明

## 结论

`main/` 目录当前有两个主要文件：

- [main.py](/home/tower/Documents/Arm_Project/mujoco_env/main/main.py)：机械臂仿真主程序，负责控制、传感器发布、默认 rosbag 录制、默认 teaching 数据保存
- [trajectory_ros2_evaluator.py](/home/tower/Documents/Arm_Project/mujoco_env/main/trajectory_ros2_evaluator.py)：离线评估脚本，读取 rosbag/mcap 后输出评分结果

---

## 1. 启动主程序

在项目根目录执行：

```bash
source /opt/ros/humble/setup.bash
cd ./Arm_Project/mujoco_env
/usr/bin/python3 main/main.py
```

说明：

- 默认订阅 `/teaching_status`
- 默认保存 teaching 数据到 `data/`
- 默认启用 rosbag 录制器，但只有收到 `/teaching_status` 指令才会真正开始录制

---

## 2. 主程序订阅的话题

### `/joint_target`

- 类型：`sensor_msgs/msg/JointState`
- 作用：发送机械臂目标关节角

发送示例：

```bash
ros2 topic pub /joint_target sensor_msgs/msg/JointState \
'{name: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], position: [0.0, 0.5, -1.0, 0.0, -0.5, 0.0]}'
```

如果要同时控制夹爪，可以带第 7 个值：

```bash
ros2 topic pub /joint_target sensor_msgs/msg/JointState \
'{name: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"], position: [0.0, 0.5, -1.0, 0.0, -0.5, 0.0, 1.0]}'
```

说明：

- 前 6 个值是机械臂关节目标
- 第 7 个值是 gripper 指令
- 程序内部会对部分关节做方向映射

### `/teaching_status`

- 类型：`std_msgs/msg/String`
- 作用：启动/停止数据记录

开始记录：

```bash
ros2 topic pub -1 /teaching_status std_msgs/msg/String '{data: "start_teaching"}'
```

结束记录：

```bash
ros2 topic pub -1 /teaching_status std_msgs/msg/String '{data: "end_teaching"}'
```

说明：

- 会同时触发本地轨迹数据保存到 `data/`
- 默认 rosbag 录制器也会监听这个话题，并开始/停止录制

---

## 3. 主程序发布的话题

### 机器人状态

- `/joint_states`：机械臂关节状态
- `/gripper_status`：夹爪状态，取值 `idle | grasping | grasped`
- `/grasp_distance`：末端到最近可抓取物体的距离，单位米
- `/ee_pose`：末端位姿，`geometry_msgs/msg/PoseStamped`

查看方法：

```bash
ros2 topic echo /joint_states
ros2 topic echo /gripper_status
ros2 topic echo /grasp_distance
ros2 topic echo /ee_pose
```

### 图像与传感器

- `/ee_camera/rgb/image_raw`
- `/ee_camera/depth/image_raw`
- `/ee_camera/camera_info`
- `/external_camera/rgb/image_raw`
- `/external_camera/depth/image_raw`
- `/external_camera/camera_info`
- `/thermal_camera/image`

查看方法：

```bash
ros2 topic hz /ee_camera/rgb/image_raw
ros2 topic hz /external_camera/rgb/image_raw
ros2 topic hz /thermal_camera/image
ros2 topic echo /ee_camera/camera_info
```

说明：

- 图像话题不建议直接 `echo`
- 建议使用 `ros2 topic hz` 看频率，或用 `rqt_image_view` 看图像

---

## 4. 如何开始记录数据

### 4.1 只录 rosbag

启动主程序：

```bash
source /opt/ros/humble/setup.bash
cd /home/tower/Documents/Arm_Project/mujoco_env
/usr/bin/python3 main/main.py
```

开始录制：

```bash
ros2 topic pub -1 /teaching_status std_msgs/msg/String '{data: "start_teaching"}'
```

停止录制：

```bash
ros2 topic pub -1 /teaching_status std_msgs/msg/String '{data: "end_teaching"}'
```

默认 rosbag 输出目录：

```bash
/home/tower/Documents/Arm_Project/rosbag_data
```

录制内容默认包括：

- `/ee_camera/rgb/image_raw`
- `/external_camera/rgb/image_raw`
- `/joint_states_sim`，录制后会重命名成 `/joint_states`
- `/joint_target`，录制后会重命名成 `/joint_cmd`
- `/gripper_status`

这时会有两类输出：

- rosbag/mcap：保存在 `rosbag_data/`
- teaching 轨迹数据：保存在 `data/`

---

## 5. 评估脚本如何使用

评估脚本入口：

- [trajectory_ros2_evaluator.py](/home/tower/Documents/Arm_Project/mujoco_env/main/trajectory_ros2_evaluator.py)

这个脚本当前主入口是离线模式，必须传 `--bag`。

### 5.1 基本用法

```bash
source /opt/ros/humble/setup.bash
cd /home/tower/Documents/Arm_Project/mujoco_env
/usr/bin/python3 main/trajectory_ros2_evaluator.py \
  --bag /home/tower/Documents/Arm_Project/rosbag_data/teaching_YYYYMMDD_HHMMSS \
  --output_dir eval_results
```

也可以直接把 `.mcap` 文件路径传给 `--bag`，但要求同目录有 `metadata.yaml`。

### 5.2 评估脚本会读取的话题

- 关节话题：默认候选 `/joint_states,/joint_states_sim`
- 夹爪话题：默认 `/gripper_status`

如果你的 bag 里话题名不同，可以手动指定：

```bash
/usr/bin/python3 main/trajectory_ros2_evaluator.py \
  --bag /path/to/bag \
  --joint_topics /joint_states,/joint_states_sim,/joint_cmd \
  --gripper_topic /gripper_status \
  --output_dir eval_results
```

### 5.3 输出文件

评估完成后会生成：

- `metrics_*.json`：评分结果
- `raw_*.npz`：原始帧数据

---

## 6. 推荐使用流程

### 查看系统话题

```bash
ros2 topic list
```

### 启动主程序

```bash
source /opt/ros/humble/setup.bash
cd /home/tower/Documents/Arm_Project/mujoco_env
/usr/bin/python3 main/main.py
```

### 查看关键状态

```bash
ros2 topic echo /joint_states
ros2 topic echo /gripper_status
ros2 topic echo /ee_pose
```

### 控制机械臂

```bash
ros2 topic pub /joint_target sensor_msgs/msg/JointState \
'{name: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], position: [0.0, 0.3, -0.8, 0.0, -0.6, 0.0]}'
```

### 开始记录

```bash
ros2 topic pub -1 /teaching_status std_msgs/msg/String '{data: "start_teaching"}'
```

### 结束记录

```bash
ros2 topic pub -1 /teaching_status std_msgs/msg/String '{data: "end_teaching"}'
```

### 离线评估

```bash
/usr/bin/python3 main/trajectory_ros2_evaluator.py \
  --bag /home/tower/Documents/Arm_Project/rosbag_data/teaching_YYYYMMDD_HHMMSS \
  --output_dir eval_results
```
