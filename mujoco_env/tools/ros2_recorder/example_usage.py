#!/usr/bin/env python3
"""
ROS2 Rosbag 录制器使用示例

展示如何在Python代码中使用录制器
"""

import rclpy
from std_msgs.msg import String
import time


def example_publish_teaching_command():
    """
    示例: 如何通过Python代码发布teaching命令
    """
    rclpy.init()
    node = rclpy.create_node('teaching_command_publisher')
    publisher = node.create_publisher(String, '/teaching_status', 10)

    # 等待订阅者连接
    time.sleep(1.0)

    print("📤 发送 start_teaching 命令...")
    msg = String()
    msg.data = "start_teaching"
    publisher.publish(msg)

    # 模拟一些操作
    print("⏳ 模拟机械臂操作 (5秒)...")
    time.sleep(5.0)

    print("📤 发送 end_teaching 命令...")
    msg.data = "end_teaching"
    publisher.publish(msg)

    # 清理
    node.destroy_node()
    rclpy.shutdown()

    print("✅ 完成")


if __name__ == '__main__':
    try:
        example_publish_teaching_command()
    except KeyboardInterrupt:
        print("\n⏹ 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
