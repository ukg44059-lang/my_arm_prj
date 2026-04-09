#!/bin/bash
# 快速入门指南 - ROS2 Rosbag 自动录制器

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════╗
║                 ROS2 Rosbag 自动录制器 - 快速入门                         ║
╚══════════════════════════════════════════════════════════════════════════╝

📦 1. 安装依赖
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sudo apt update
sudo apt install ros-humble-rosbag2-storage-mcap

🚀 2. 启动录制器
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cd mujoco_env/tools/ros2_recorder
./start_rosbag_recorder.sh

📝 3. 控制录制
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 开始录制
ros2 topic pub -1 /teaching_status std_msgs/msg/String "data: start_teaching"

# 停止录制
ros2 topic pub -1 /teaching_status std_msgs/msg/String "data: end_teaching"

📊 4. 查看录制数据
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 查看信息
ros2 bag info rosbag_data/teaching_YYYYMMDD_HHMMSS

# 播放数据
ros2 bag play rosbag_data/teaching_YYYYMMDD_HHMMSS

🔧 5. 与机械臂程序集成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
终端1: 启动录制器
  cd mujoco_env/tools/ros2_recorder
  ./start_rosbag_recorder.sh

终端2: 运行机械臂程序
  /usr/bin/python3 mujoco_env/test/test_simple_trajectory.py

终端3 (可选): 手动控制录制
  # 开始
  ros2 topic pub -1 /teaching_status std_msgs/msg/String "data: start_teaching"
  # 停止
  ros2 topic pub -1 /teaching_status std_msgs/msg/String "data: end_teaching"

💡 提示: 在机械臂程序中按 'T' 键也会触发 /teaching_status 话题！

📂 录制数据位置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
rosbag_data/teaching_YYYYMMDD_HHMMSS/

🧪 测试录制器
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
./test_recorder.sh

❓ 获取帮助
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
查看完整文档: cat README.md

╔══════════════════════════════════════════════════════════════════════════╗
║                            准备就绪！                                     ║
╚══════════════════════════════════════════════════════════════════════════╝
EOF
