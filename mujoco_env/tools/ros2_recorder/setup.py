"""
ROS2 Rosbag 自动录制工具包

安装方法:
    pip install -e .

或者直接使用（推荐）:
    /usr/bin/python3 rosbag_recorder.py
"""

from setuptools import setup, find_packages

setup(
    name='ros2-rosbag-recorder',
    version='1.0.0',
    description='ROS2 automatic rosbag recorder triggered by /teaching_status topic',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        # 注意: rclpy 和 rosbag2 需要从系统ROS2安装获取
        # 不能通过pip安装
    ],
    entry_points={
        'console_scripts': [
            'rosbag-recorder=rosbag_recorder:main',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
    ],
)
