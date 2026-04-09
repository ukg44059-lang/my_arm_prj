# ROS2 Rosbag 自动录制工具 - 安装完成

## ✅ 已创建的文件

```
mujoco_env/tools/ros2_recorder/
├── __init__.py                    # Python包初始化
├── rosbag_recorder.py             # 核心录制器（主程序）
├── start_rosbag_recorder.sh       # Bash启动脚本（推荐使用）
├── start_recorder.py              # Python启动脚本
├── test_recorder.sh               # 自动测试脚本
├── setup.py                       # Python包安装配置
├── README.md                      # 完整文档
└── QUICKSTART.sh                  # 快速入门指南
```

## 🚀 快速开始

### 1. 安装依赖（仅需一次）

```bash
sudo apt update
sudo apt install ros-humble-rosbag2-storage-mcap
```

### 2. 启动录制器

```bash
cd /home/tower/Documents/Arm_Project/mujoco_env/tools/ros2_recorder
./start_rosbag_recorder.sh
```

### 3. 控制录制

**方法A: 通过ROS2话题**
```bash
# 开始录制
ros2 topic pub -1 /teaching_status std_msgs/msg/String "data: start_teaching"

# 停止录制
ros2 topic pub -1 /teaching_status std_msgs/msg/String "data: end_teaching"
```

**方法B: 在机械臂程序中按 'T' 键**
- `test_simple_trajectory.py` 已经集成了Teaching功能
- 按 'T' 键会自动发布 `/teaching_status` 消息
- 录制器会自动响应

## 🔧 工作原理

```
┌─────────────────────────────┐
│  用户操作                    │
│  - 按 'T' 键                │
│  - 发布ROS2消息              │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  /teaching_status 话题       │
│  std_msgs/msg/String        │
│  data: "start_teaching"     │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  rosbag_recorder.py         │
│  监听话题，自动启动/停止     │
│  ros2 bag record -a         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  rosbag_data/               │
│  teaching_YYYYMMDD_HHMMSS/  │
│  └── *.mcap                 │
└─────────────────────────────┘
```

## 📊 录制的数据包含

- ✅ 关节状态 (`/joint_states`, `/joint_states_sim`)
- ✅ RGBD相机图像 (如果启用了ROS2图像发布)
- ✅ 末端执行器状态
- ✅ 所有其他ROS2话题

## 🎯 与现有程序集成

### 完整工作流程

**终端1 - 启动录制器:**
```bash
cd /home/tower/Documents/Arm_Project/mujoco_env/tools/ros2_recorder
./start_rosbag_recorder.sh
```

**终端2 - 运行机械臂程序:**
```bash
cd /home/tower/Documents/Arm_Project
/usr/bin/python3 mujoco_env/test/test_simple_trajectory.py
```

**在机械臂程序中:**
1. 按 `T` 键开始Teaching录制
2. 执行机械臂操作
3. 再次按 `T` 键停止录制

**终端1 - 录制器会自动:**
- 检测到 `start_teaching` 并开始录制
- 检测到 `end_teaching` 并停止录制
- 显示录制的文件位置和统计信息

## 📁 查看录制数据

### 查看bag信息
```bash
ros2 bag info rosbag_data/teaching_20260129_143052
```

### 播放录制数据
```bash
ros2 bag play rosbag_data/teaching_20260129_143052
```

### 导出特定话题
```bash
ros2 bag play rosbag_data/teaching_20260129_143052 --topics /joint_states
```

## ⚙️ 高级配置

### 只录制特定话题

编辑 `rosbag_recorder.py` 的 `main()` 函数:

```python
recorder = RosbagRecorder(
    output_dir="rosbag_data",
    topics=[
        '/joint_states',
        '/joint_states_sim',
        '/ee_camera/image_raw',
        '/ee_camera/depth/image_raw',
        '/external_camera/image_raw',
        '/external_camera/depth/image_raw'
    ],
    storage_format="mcap"
)
```

### 修改输出目录

```python
recorder = RosbagRecorder(
    output_dir="/path/to/custom/output",
    topics=None,
    storage_format="mcap"
)
```

## 🧪 测试录制器

运行自动化测试:

```bash
cd /home/tower/Documents/Arm_Project/mujoco_env/tools/ros2_recorder
./test_recorder.sh
```

这会自动:
1. 启动录制器
2. 发送 start_teaching
3. 发布测试消息
4. 发送 end_teaching
5. 验证录制结果

## 🐛 故障排除

### 问题1: 找不到 rosbag2 命令
```bash
sudo apt install ros-humble-rosbag2
```

### 问题2: 录制器没有响应
检查话题是否存在:
```bash
ros2 topic list | grep teaching_status
ros2 topic echo /teaching_status
```

### 问题3: 权限错误
确保脚本有执行权限:
```bash
chmod +x start_rosbag_recorder.sh
chmod +x test_recorder.sh
```

### 问题4: ROS2环境未加载
确保已经source ROS2:
```bash
source /opt/ros/humble/setup.bash
```

## 📚 相关文档

- 完整文档: `README.md`
- 快速入门: `./QUICKSTART.sh`
- 测试脚本: `./test_recorder.sh`

## 💡 提示

1. **数据存储**: MCAP格式已经是压缩格式，无需额外压缩
2. **性能**: 录制所有话题时，磁盘写入速度很重要
3. **清理**: 定期清理旧的录制数据以释放空间
4. **备份**: 重要的录制数据建议及时备份

## 🎉 完成！

工具已经准备就绪。现在你可以:
1. ✅ 启动录制器监听 `/teaching_status` 话题
2. ✅ 通过ROS2消息或按键控制录制
3. ✅ 使用MCAP格式高效存储所有ROS2数据
4. ✅ 回放和分析录制的数据

---

**需要帮助?** 查看 `README.md` 或运行 `./QUICKSTART.sh`
