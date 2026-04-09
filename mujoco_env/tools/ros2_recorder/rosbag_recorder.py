#!/usr/bin/env python3
"""
ROS2 Rosbag 自动录制器

功能:
1. 监听 /teaching_status 话题
2. 收到 'start_teaching' 时开始录制
3. 收到 'end_teaching' 时停止录制
4. 使用 rosbag2 (MCAP格式) 录制所有话题

使用方法:
    # 使用系统Python (必须)
    /usr/bin/python3 -m mujoco_env.tools.ros2_recorder.rosbag_recorder

    # 或者直接运行
    /usr/bin/python3 mujoco_env/tools/ros2_recorder/rosbag_recorder.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import os
import signal
from datetime import datetime
from pathlib import Path
import threading


class RosbagRecorder(Node):
    """
    ROS2 Rosbag 自动录制器节点

    监听 /teaching_status 话题，根据消息自动启动/停止 rosbag2 录制
    """

    def __init__(self,
                 output_dir: str = "rosbag_data",
                 topics: list = None,
                 storage_format: str = "mcap",
                 compression_mode: str = "message",
                 compression_format: str = "zstd",
                 topic_rename_map: dict = None):
        """
        初始化录制器

        Args:
            output_dir: 录制数据输出目录
            topics: 要录制的话题列表，None表示录制所有话题
            storage_format: 存储格式 (mcap, sqlite3)
            compression_mode: 压缩模式 ('none', 'file', 'message')
            compression_format: 压缩格式 ('zstd', 'fake_comp')
            topic_rename_map: 话题重命名映射 {原话题: 新话题}，录制后自动重命名
        """
        super().__init__('rosbag_recorder')

        # 配置参数
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.topics = topics  # None = 录制所有话题
        self.storage_format = storage_format
        self.compression_mode = compression_mode
        self.compression_format = compression_format
        self.topic_rename_map = topic_rename_map or {}

        # 录制状态
        self.is_recording = False
        self.rosbag_process = None
        self.current_bag_dir = None

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 订阅 teaching_status 话题
        self.subscription = self.create_subscription(
            String,
            '/teaching_status',
            self.teaching_status_callback,
            10
        )

        self.get_logger().info("=" * 70)
        self.get_logger().info("🎬 ROS2 Rosbag 自动录制器已启动")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"📁 输出目录: {self.output_dir}")
        self.get_logger().info(f"💾 存储格式: {self.storage_format}")
        if self.storage_format == "mcap":
            self.get_logger().info(f"🗜️  压缩模式: MCAP 内置 (自动压缩)")
        else:
            self.get_logger().info(f"🗜️  压缩模式: {self.compression_mode} ({self.compression_format})")

        if self.topics:
            self.get_logger().info(f"📡 录制话题: {', '.join(self.topics)}")
        else:
            self.get_logger().info("📡 录制话题: 所有话题 (-a)")

        if self.topic_rename_map:
            self.get_logger().info("✏️  话题重命名:")
            for old_topic, new_topic in self.topic_rename_map.items():
                self.get_logger().info(f"   {old_topic} → {new_topic}")

        self.get_logger().info("🎧 正在监听 /teaching_status 话题...")
        self.get_logger().info("   - 发送 'start_teaching' 开始录制")
        self.get_logger().info("   - 发送 'end_teaching' 停止录制")
        self.get_logger().info("=" * 70)

    def teaching_status_callback(self, msg: String):
        """
        /teaching_status 话题回调函数

        Args:
            msg: String消息，data字段包含命令
        """
        command = msg.data.strip().lower()

        if command == "start_teaching":
            if not self.is_recording:
                self.start_recording()
            else:
                self.get_logger().warn("⚠ 录制已在进行中，忽略重复的 start_teaching 命令")

        elif command == "end_teaching":
            if self.is_recording:
                self.stop_recording()
            else:
                self.get_logger().warn("⚠ 当前没有进行录制，忽略 end_teaching 命令")

        else:
            self.get_logger().debug(f"📨 收到未识别的命令: '{command}'")

    def start_recording(self):
        """启动 rosbag2 录制"""
        try:
            # 生成录制文件夹名称（基于时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bag_name = f"teaching_{timestamp}"
            self.current_bag_dir = self.output_dir / bag_name

            # 如果是 MCAP 格式且需要压缩，创建存储配置文件
            storage_config_file = None
            if self.storage_format == "mcap" and self.compression_mode != "none":
                import yaml
                # MCAP 存储插件的配置格式
                storage_config = {
                    'output_options': {
                        'compression': 'Zstd',
                        'compression_level': 22,  # 压缩级别 (1-22, 3是默认值)
                        'force_compression': True
                    }
                }

                # 创建临时配置文件
                storage_config_file = self.output_dir / f"storage_config_{timestamp}.yaml"
                with open(storage_config_file, 'w') as f:
                    yaml.dump(storage_config, f)

                self.get_logger().info(f"📝 创建存储配置文件: {storage_config_file}")

            # 构建 rosbag2 命令
            cmd = [
                "ros2", "bag", "record",
                "-s", self.storage_format,  # 存储格式
                "-o", str(self.current_bag_dir)  # 输出目录
            ]

            # 添加存储配置文件（用于 MCAP 压缩）
            if storage_config_file:
                cmd.extend(["--storage-config-file", str(storage_config_file)])

            # 添加压缩选项（仅对非MCAP格式）
            # 注意: MCAP 存储插件内部处理压缩，不支持 --compression-mode 参数
            if self.compression_mode and self.compression_mode != "none" and self.storage_format != "mcap":
                cmd.extend(["--compression-mode", self.compression_mode])
                cmd.extend(["--compression-format", self.compression_format])

            # 添加话题
            if self.topics:
                cmd.extend(self.topics)
            else:
                cmd.append("-a")  # 录制所有话题

            self.get_logger().info(f"🎬 开始录制: {bag_name}")
            self.get_logger().info(f"📂 保存位置: {self.current_bag_dir}")
            self.get_logger().info(f"💾 存储格式: {self.storage_format}")
            if self.storage_format == "mcap" and self.compression_mode != "none":
                self.get_logger().info(f"🗜️  压缩: Zstd")
            elif self.compression_mode and self.compression_mode != "none":
                self.get_logger().info(f"🗜️  压缩: {self.compression_mode} ({self.compression_format})")
            else:
                self.get_logger().info(f"🗜️  压缩: 无")

            if self.topics:
                self.get_logger().info(f"📡 录制话题: {', '.join(self.topics)}")
            else:
                self.get_logger().info("📡 录制所有话题")

            # 启动 rosbag2 进程
            self.rosbag_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # 创建新进程组，便于清理
            )

            self.is_recording = True

            # 启动输出监控线程
            threading.Thread(
                target=self._monitor_rosbag_output,
                daemon=True
            ).start()

            self.get_logger().info("✅ 录制已启动")

        except Exception as e:
            self.get_logger().error(f"❌ 启动录制失败: {e}")
            import traceback
            traceback.print_exc()

    def stop_recording(self):
        """停止 rosbag2 录制"""
        try:
            if self.rosbag_process is None:
                self.get_logger().warn("⚠ 没有正在运行的录制进程")
                return

            self.get_logger().info("🛑 正在停止录制...")

            # 发送 SIGINT (Ctrl+C) 给进程组
            try:
                os.killpg(os.getpgid(self.rosbag_process.pid), signal.SIGINT)
            except ProcessLookupError:
                self.get_logger().warn("⚠ 录制进程已经结束")

            # 等待进程结束
            try:
                self.rosbag_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.get_logger().warn("⚠ 进程未在超时时间内结束，强制终止")
                os.killpg(os.getpgid(self.rosbag_process.pid), signal.SIGKILL)
                self.rosbag_process.wait()

            self.get_logger().info(f"✅ 录制已停止")
            self.get_logger().info(f"📂 数据已保存到: {self.current_bag_dir}")

            # 显示录制信息
            self._show_bag_info()

            # 如果需要重命名话题，执行重命名
            if self.topic_rename_map:
                self._rename_topics_in_bag()

            self.is_recording = False
            self.rosbag_process = None
            self.current_bag_dir = None

        except Exception as e:
            self.get_logger().error(f"❌ 停止录制失败: {e}")
            import traceback
            traceback.print_exc()

    def _monitor_rosbag_output(self):
        """监控 rosbag2 进程的输出（在独立线程中运行）"""
        if self.rosbag_process is None:
            return

        try:
            # 读取标准错误输出（rosbag2的主要输出在stderr）
            for line in iter(self.rosbag_process.stderr.readline, b''):
                if not self.is_recording:
                    break

                line_str = line.decode('utf-8').strip()
                if line_str:
                    # 只显示重要信息
                    if any(keyword in line_str.lower() for keyword in
                           ['error', 'warn', 'recording', 'topics']):
                        self.get_logger().info(f"  📝 {line_str}")

        except Exception as e:
            self.get_logger().debug(f"输出监控线程错误: {e}")

    def _show_bag_info(self):
        """显示录制的bag文件信息"""
        if self.current_bag_dir is None or not self.current_bag_dir.exists():
            return

        try:
            # 运行 ros2 bag info
            result = subprocess.run(
                ["ros2", "bag", "info", str(self.current_bag_dir)],
                capture_output=True,
                text=True,
                timeout=5.0
            )

            if result.returncode == 0:
                self.get_logger().info("\n" + "=" * 70)
                self.get_logger().info("📊 录制信息:")
                self.get_logger().info("=" * 70)
                for line in result.stdout.strip().split('\n'):
                    self.get_logger().info(f"  {line}")
                self.get_logger().info("=" * 70)

        except Exception as e:
            self.get_logger().debug(f"无法获取bag信息: {e}")

    def _update_metadata_topics(self):
        """更新 metadata.yaml 中的话题名以匹配 mcap 中重命名后的话题"""
        if self.current_bag_dir is None or not self.current_bag_dir.exists():
            return

        metadata_file = self.current_bag_dir / "metadata.yaml"
        if not metadata_file.exists():
            self.get_logger().warn("⚠️  未找到 metadata.yaml，跳过更新")
            return

        try:
            import yaml
            
            # 读取 metadata.yaml
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)

            # 更新话题名
            updated_count = 0
            if 'rosbag2_bagfile_information' in metadata and 'topics_with_message_count' in metadata['rosbag2_bagfile_information']:
                for topic_info in metadata['rosbag2_bagfile_information']['topics_with_message_count']:
                    if 'topic_metadata' in topic_info and 'name' in topic_info['topic_metadata']:
                        old_name = topic_info['topic_metadata']['name']
                        if old_name in self.topic_rename_map:
                            new_name = self.topic_rename_map[old_name]
                            topic_info['topic_metadata']['name'] = new_name
                            updated_count += 1
                            self.get_logger().info(f"   metadata.yaml: {old_name} → {new_name}")

            if updated_count > 0:
                # 写回文件
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                self.get_logger().info(f"✅ metadata.yaml 已更新（{updated_count} 个话题）")
            else:
                self.get_logger().info("ℹ️  metadata.yaml 无需更新")

        except Exception as e:
            self.get_logger().warn(f"⚠️  更新 metadata.yaml 失败: {e}")

    def _rename_topics_in_bag(self):
        """通过调用 mcap_topic_renamer.py 重命名录制的bag文件中的话题"""
        if self.current_bag_dir is None or not self.current_bag_dir.exists():
            return

        try:
            self.get_logger().info("✏️  正在重命名话题...")

            # 查找 mcap 文件
            mcap_files = list(self.current_bag_dir.glob("*.mcap"))
            if not mcap_files:
                self.get_logger().warn("⚠ 未找到 MCAP 文件，跳过重命名")
                return

            mcap_file = mcap_files[0]  # 取第一个 mcap 文件
            renamed_file = self.current_bag_dir / f"{mcap_file.stem}_renamed.mcap"

            # 获取 mcap_topic_renamer.py 的路径
            renamer_script = Path(__file__).parent / "mcap_topic_renamer.py"

            if not renamer_script.exists():
                self.get_logger().error(f"❌ 找不到重命名脚本: {renamer_script}")
                return

            # 构建命令
            cmd = [
                "python3",
                str(renamer_script),
                str(mcap_file),
                str(renamed_file)
            ]

            # 添加重命名规则
            for old_topic, new_topic in self.topic_rename_map.items():
                cmd.extend(["--rename", f"{old_topic}:{new_topic}"])
                self.get_logger().info(f"   {old_topic} → {new_topic}")

            self.get_logger().info(f"🔧 调用重命名工具: mcap_topic_renamer.py")

            # 启动重命名进程
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            if result.returncode == 0:
                self.get_logger().info("✅ 话题重命名完成")

                # 删除原文件
                try:
                    mcap_file.unlink()
                    self.get_logger().info(f"🗑️  已删除原文件: {mcap_file.name}")
                except Exception as e:
                    self.get_logger().warn(f"⚠️  删除原文件失败: {e}")

                # 将重命名后的文件改为原文件名
                try:
                    renamed_file.rename(mcap_file)
                    self.get_logger().info(f"✅ 已将重命名文件替换为原文件名")
                except Exception as e:
                    self.get_logger().warn(f"⚠️  重命名文件失败: {e}")

                # 更新 metadata.yaml 中的话题名
                self._update_metadata_topics()

                # 显示重命名工具的部分输出（统计信息）
                if result.stdout:
                    # 只显示统计信息部分
                    lines = result.stdout.split('\n')
                    in_stats = False
                    for line in lines:
                        if '=' * 70 in line or '统计信息' in line or '已重命名的话题' in line:
                            in_stats = True
                        if in_stats and line.strip():
                            self.get_logger().info(f"  {line}")
            else:
                self.get_logger().error(f"❌ 话题重命名失败，返回码: {result.returncode}")
                if result.stderr:
                    self.get_logger().error(f"错误信息:\n{result.stderr}")

        except subprocess.TimeoutExpired:
            self.get_logger().error("❌ 话题重命名超时（超过5分钟）")
        except Exception as e:
            self.get_logger().error(f"❌ 话题重命名失败: {e}")
            import traceback
            traceback.print_exc()

    def shutdown(self):
        """清理并关闭"""
        if self.is_recording:
            self.get_logger().info("🧹 检测到关闭请求，停止正在进行的录制...")
            self.stop_recording()

        self.get_logger().info("👋 ROS2 Rosbag 录制器已关闭")


def main(args=None):
    """主函数"""
    rclpy.init(args=args)

    # 创建录制器节点
    # 可以自定义参数
    recorder = RosbagRecorder(
        output_dir="/home/tower/Documents/Arm_Project/rosbag_data",  # 输出目录
        topics=['/ee_camera/rgb/image_raw', '/external_camera/rgb/image_raw', '/joint_states_sim', '/joint_target_R'],  # None = 录制所有话题，或指定列表如 ['/joint_states', '/camera/image']
        storage_format="mcap",  # 使用MCAP格式
        compression_mode="message",  # 启用消息级压缩 ('none', 'file', 'message')
        compression_format="zstd",  # 使用 Zstandard 压缩
        topic_rename_map={  # 录制后自动重命名话题
            '/joint_states_sim': '/joint_states',
            '/joint_target_R': '/joint_command'
        }
    )

    try:
        # 运行节点
        rclpy.spin(recorder)

    except KeyboardInterrupt:
        print("\n⏹ 用户中断 (Ctrl+C)")

    finally:
        # 清理
        recorder.shutdown()
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
