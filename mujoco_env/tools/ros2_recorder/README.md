# ROS2 Rosbag 自动录制工具

这是一个独立的ROS2工具包，用于监听`/teaching_status`话题并自动启动/停止rosbag2录制。

## 功能特性

- ✅ 监听 `/teaching_status` 话题
- ✅ 收到 `start_teaching` 自动开始录制
- ✅ 收到 `end_teaching` 自动停止录制
- ✅ 使用 MCAP 格式存储（高性能）
- ✅ 可配置录制所有话题或指定话题
- ✅ 自动生成带时间戳的录制文件夹
- ✅ 录制完成后显示详细信息
- ✅ 使用系统Python和ROS2环境
- ✨ **NEW: 录制时启用压缩**（防止图像数据占用过多空间）
- ✨ **NEW: 录制后自动重命名话题**（无需手动后处理）

## 安装依赖

```bash
# 安装 ROS2 Humble (如果未安装)
sudo apt update
sudo apt install ros-humble-desktop

# 安装 rosbag2 MCAP 存储支持
sudo apt install ros-humble-rosbag2-storage-mcap

# 安装 mcap Python 包（用于话题重命名）
pip install mcap

# 安装 pyyaml（用于生成存储配置文件）
pip install pyyaml

# 验证安装
dpkg -l | grep rosbag2-storage-mcap
```

## 使用方法

### 方法1: 使用启动脚本（推荐）

```bash
# 进入录制器目录
cd mujoco_env/tools/ros2_recorder

# 运行启动脚本
./start_rosbag_recorder.sh
```

### 方法2: 直接运行Python脚本

```bash
# 确保已经source ROS2环境
source /opt/ros/humble/setup.bash

# 运行录制器
/usr/bin/python3 mujoco_env/tools/ros2_recorder/rosbag_recorder.py
```

## 控制录制

录制器启动后，会监听 `/teaching_status` 话题。你可以通过发布消息来控制录制：

### 开始录制

```bash
ros2 topic pub -1 /teaching_status std_msgs/msg/String "data: start_teaching"
```

### 停止录制

```bash
ros2 topic pub -1 /teaching_status std_msgs/msg/String "data: end_teaching"
```

## 录制数据位置

录制的数据默认保存在 `rosbag_data/` 目录下，文件夹命名格式为：

```
rosbag_data/
  └── teaching_YYYYMMDD_HHMMSS/
      ├── metadata.yaml
      └── teaching_YYYYMMDD_HHMMSS_0.mcap
```

例如：
```
rosbag_data/
  └── teaching_20260129_143052/
      ├── metadata.yaml
      └── teaching_20260129_143052_0.mcap
```

## 查看录制数据

### 查看bag信息

```bash
ros2 bag info rosbag_data/teaching_20260129_143052
```

### 播放bag文件

```bash
ros2 bag play rosbag_data/teaching_20260129_143052
```

### 查看话题列表

```bash
ros2 bag info rosbag_data/teaching_20260129_143052 | grep "Topic information"
```

## 配置选项

你可以修改 `rosbag_recorder.py` 中的 `main()` 函数来自定义配置：

```python
recorder = RosbagRecorder(
    output_dir="rosbag_data",           # 输出目录
    topics=None,                         # None=录制所有话题，或指定列表
    storage_format="mcap",               # 存储格式: mcap 或 sqlite3
    compression_mode="message",          # 压缩模式: 'none', 'file', 'message'
    compression_format="zstd",           # 压缩格式: 'zstd', 'fake_comp'
    topic_rename_map={}                  # 话题重命名映射（可选）
)
```

### 示例 1：只录制特定话题

```python
recorder = RosbagRecorder(
    output_dir="rosbag_data",
    topics=[
        '/joint_states',
        '/camera/image_raw',
        '/camera/depth/image_raw'
    ],
    storage_format="mcap"
)
```

### 示例 2：启用压缩（推荐用于包含图像的录制）

```python
recorder = RosbagRecorder(
    output_dir="rosbag_data",
    topics=['/ee_camera/rgb/image_raw', '/external_camera/rgb/image_raw', '/joint_states'],
    storage_format="mcap",
    compression_mode="message",          # 消息级压缩
    compression_format="zstd"            # Zstandard 压缩算法
)
```

**压缩效果**：图像话题可减少 90%+ 存储空间！

| 场景 | 无压缩 | 启用压缩 | 节省 |
|------|-------|---------|------|
| 2个640x480图像+关节状态 (17秒) | 459 MB | 27 MB | 94% |

### 示例 3：录制并自动重命名话题

```python
recorder = RosbagRecorder(
    output_dir="rosbag_data",
    topics=['/joint_states_sim', '/joint_target_R', '/ee_camera/rgb/image_raw'],
    storage_format="mcap",
    compression_mode="message",
    compression_format="zstd",
    topic_rename_map={
        '/joint_states_sim': '/joint_states',  # 录制 /joint_states_sim，保存为 /joint_states
        '/joint_target_R': '/joint_cmd'        # 录制 /joint_target_R，保存为 /joint_cmd
    }
)
```

**重命名工作流程**：
1. 录制时使用原始话题名（保持与现有系统兼容）
2. 停止录制后，自动重命名 MCAP 文件中的话题
3. 无需手动后处理

### 示例 4：为 LeRobot 准备数据

```python
recorder = RosbagRecorder(
    output_dir="./lerobot_data",
    topics=[
        '/ee_camera/rgb/image_raw',
        '/external_camera/rgb/image_raw',
        '/joint_states_sim',
        '/joint_target_R'
    ],
    storage_format="mcap",
    compression_mode="message",
    compression_format="zstd",
    topic_rename_map={
        '/ee_camera/rgb/image_raw': '/observation.images.top',
        '/external_camera/rgb/image_raw': '/observation.images.wrist',
        '/joint_states_sim': '/observation.state',
        '/joint_target_R': '/action'
    }
)
```

## 与 test_simple_trajectory.py 集成

在你的机械臂测试程序运行时：

1. **启动录制器**（在单独的终端）：
   ```bash
   cd mujoco_env/tools/ros2_recorder
   ./start_rosbag_recorder.sh
   ```

2. **运行机械臂程序**：
   ```bash
   /usr/bin/python3 mujoco_env/test/test_simple_trajectory.py
   ```

3. **开始Teaching录制**：
   - 在程序中按 `T` 键，或
   - 发布ROS2消息：
     ```bash
     ros2 topic pub -1 /teaching_status std_msgs/msg/String "data: start_teaching"
     ```

4. **停止Teaching录制**：
   - 再次按 `T` 键，或
   - 发布ROS2消息：
     ```bash
     ros2 topic pub -1 /teaching_status std_msgs/msg/String "data: end_teaching"
     ```

## 工作流程示意

```
┌─────────────────────┐
│  test_simple_       │
│  trajectory.py      │
│                     │
│  按 'T' 键开始Teaching │
└──────────┬──────────┘
           │
           │ 发布 /teaching_status
           │ "start_teaching"
           ▼
┌─────────────────────┐
│  rosbag_recorder.py │
│                     │
│  监听话题自动启动     │
│  ros2 bag record    │
└──────────┬──────────┘
           │
           │ 录制所有ROS2话题
           │ (关节状态、图像等)
           ▼
┌─────────────────────┐
│  rosbag_data/       │
│  teaching_xxx.mcap  │
└─────────────────────┘
```

## 常见问题

### Q: 录制器没有响应？
A: 确保：
1. ROS2环境已正确source
2. `/teaching_status` 话题存在：`ros2 topic list | grep teaching_status`
3. 消息格式正确：使用 `std_msgs/msg/String`

### Q: 找不到rosbag2命令？
A: 安装rosbag2工具：
```bash
sudo apt install ros-humble-rosbag2
```

### Q: 录制的文件很大？
A: MCAP格式已经是压缩格式。如果需要进一步压缩，可以：
1. 只录制必要的话题（修改`topics`参数）
2. 降低图像话题的发布频率

### Q: 如何导出数据到其他格式？
A: 可以使用ros2 bag工具链或第三方工具：
```bash
# 导出为CSV
ros2 bag convert -i rosbag_data/teaching_xxx -o output.csv

# 使用mcap工具查看
pip install mcap
mcap info rosbag_data/teaching_xxx/teaching_xxx_0.mcap
```

## 技术细节

- **语言**: Python 3
- **ROS2版本**: Humble
- **存储格式**: MCAP (默认) 或 SQLite3
- **依赖包**:
  - `rclpy`
  - `std_msgs`
  - `rosbag2`
  - `rosbag2-storage-mcap`

## 贡献

欢迎提交Issue和Pull Request！

## 验证压缩是否生效

录制完成后，可以通过以下方法验证压缩是否正常工作：

### 方法1：检查文件大小

```bash
# 查看录制文件大小
du -h rosbag_data/teaching_YYYYMMDD_HHMMSS/

# 对比：如果包含两个640x480图像话题，录制约20秒
# - 无压缩: ~450-500 MB
# - 有压缩: ~25-30 MB (节省 94%)
```

### 方法2：使用 mcap 工具查看压缩信息

```bash
# 安装 mcap CLI 工具
pip install mcap-cli

# 查看 MCAP 文件详细信息（包括压缩信息）
mcap info rosbag_data/teaching_YYYYMMDD_HHMMSS/*.mcap

# 输出中会显示:
# Compression: zstd
# Uncompressed size: XXX MB
# Compressed size: YYY MB
```

### 方法3：检查生成的配置文件

录制启动时，会在 `rosbag_data/` 目录下生成临时配置文件 `storage_config_YYYYMMDD_HHMMSS.yaml`。

查看内容确认压缩配置正确：

```bash
cat rosbag_data/storage_config_*.yaml
```

应该看到：

```yaml
output_options:
  compression: Zstd
  compression_level: 3
  force_compression: true
```

### 排查压缩不生效的问题

如果录制的文件仍然很大（接近500MB），可能的原因：

1. **缺少 pyyaml 依赖**：
   ```bash
   pip install pyyaml
   ```

2. **MCAP 存储插件版本过低**：
   ```bash
   # 检查版本
   apt list --installed | grep rosbag2-storage-mcap

   # 更新到最新版本
   sudo apt update
   sudo apt upgrade ros-humble-rosbag2-storage-mcap
   ```

3. **配置文件未正确加载**：检查录制日志中是否有类似以下的输出：
   ```
   📝 创建存储配置文件: /path/to/storage_config_xxx.yaml
   ```

4. **手动测试压缩**：
   ```bash
   # 手动录制测试，使用自定义配置文件
   cat > test_config.yaml << EOF
   output_options:
     compression: Zstd
     compression_level: 3
     force_compression: true
   EOF

   ros2 bag record -a -s mcap --storage-config-file test_config.yaml -o test_bag
   ```

## 许可证

MIT License
