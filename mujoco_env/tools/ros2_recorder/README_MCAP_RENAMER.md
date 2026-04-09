# MCAP 话题重命名工具

这个工具可以修改 MCAP (ROS2 bag) 文件中的话题名称。

## 安装依赖

```bash
pip install mcap
```

## 使用方法

### 基本用法

重命名单个话题：

```bash
python mcap_topic_renamer.py input.mcap output.mcap \
    --rename /old_topic:/new_topic
```

### 重命名多个话题

```bash
python mcap_topic_renamer.py input.mcap output.mcap \
    --rename /camera/image:/camera/rgb/image \
    --rename /joint_states:/robot/joint_states \
    --rename /ee_camera/rgb/image_raw:/observation.images.ee_cam
```

### 处理 rosbag 目录

ROS2 bag 通常是一个目录，里面包含 MCAP 文件和 metadata.yaml。你需要处理其中的 .mcap 文件：

```bash
# 查看 bag 目录内容
ls rosbag_data/teaching_20240130/

# 重命名 MCAP 文件中的话题
python mcap_topic_renamer.py \
    rosbag_data/teaching_20240130/teaching_20240130_0.mcap \
    rosbag_data/teaching_20240130_renamed/teaching_20240130_renamed_0.mcap \
    --rename /ee_camera/rgb/image_raw:/observation.images.ee_cam \
    --rename /external_camera/rgb/image_raw:/observation.images.ext_cam \
    --rename /joint_states_sim:/observation.state \
    --rename /joint_target_R:/action
```

### 预览模式（不实际修改）

使用 `--dry-run` 查看将要执行的操作：

```bash
python mcap_topic_renamer.py input.mcap output.mcap \
    --rename /old_topic:/new_topic \
    --dry-run
```

### 详细输出模式

使用 `-v` 或 `--verbose` 查看详细处理信息：

```bash
python mcap_topic_renamer.py input.mcap output.mcap \
    --rename /old_topic:/new_topic \
    --verbose
```

## 常见使用场景

### 场景 1: 为 LeRobot 准备数据

LeRobot 期望特定的话题命名格式，例如：

```bash
python mcap_topic_renamer.py input.mcap output.mcap \
    --rename /ee_camera/rgb/image_raw:/observation.images.top \
    --rename /external_camera/rgb/image_raw:/observation.images.wrist \
    --rename /joint_states:/observation.state \
    --rename /joint_target:/action
```

### 场景 2: 批量处理多个文件

创建一个批处理脚本：

```bash
#!/bin/bash

INPUT_DIR="rosbag_data"
OUTPUT_DIR="rosbag_data_renamed"

mkdir -p "$OUTPUT_DIR"

for bag_dir in "$INPUT_DIR"/teaching_*; do
    bag_name=$(basename "$bag_dir")
    mcap_file="$bag_dir/${bag_name}_0.mcap"

    if [ -f "$mcap_file" ]; then
        echo "处理: $mcap_file"
        python mcap_topic_renamer.py \
            "$mcap_file" \
            "$OUTPUT_DIR/${bag_name}_renamed.mcap" \
            --rename /ee_camera/rgb/image_raw:/observation.images.ee_cam \
            --rename /external_camera/rgb/image_raw:/observation.images.ext_cam
    fi
done
```

### 场景 3: 验证结果

处理完成后，使用 `ros2 bag info` 验证：

```bash
ros2 bag info output.mcap
```

## 技术细节

- 保留所有消息数据和时间戳
- 保留所有 Schema 和 Channel 元数据
- 保留消息顺序和序列号
- 输出文件格式与输入文件完全兼容

## 注意事项

1. **输出文件大小**: 输出文件大小应该与输入文件基本相同
2. **话题名称**: 确保新话题名称符合 ROS2 命名规范（通常以 `/` 开头）
3. **备份**: 建议在处理前备份原始文件
4. **内存使用**: 脚本会逐条处理消息，内存占用很小，适合处理大文件

## 故障排除

### 错误: 找不到 mcap 模块

```bash
pip install mcap
```

### 错误: 文件不存在

检查输入文件路径是否正确，rosbag 目录下通常有多个文件。

### 话题没有被重命名

使用 `ros2 bag info` 查看原始文件中的实际话题名称，确保指定的旧话题名称完全匹配。
