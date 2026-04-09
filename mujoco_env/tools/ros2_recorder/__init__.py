"""
ROS2 自动录制工具包

监听 /teaching_status 话题，自动启动/停止 rosbag2 录制
"""

from .rosbag_recorder import RosbagRecorder

__all__ = ['RosbagRecorder']
