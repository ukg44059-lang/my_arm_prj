"""
Real-time Joint Angle Plotter
类似MATLAB的实时关节角度可视化工具
显示命令角度和反馈角度的实时曲线
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import threading


class JointAnglePlotter:
    """实时绘制关节命令角度和反馈角度"""

    def __init__(self, n_joints=6, window_size=4000, update_interval=50, time_window_width=10.0):
        """
        初始化关节角度绘图器

        Args:
            n_joints: 关节数量
            window_size: 显示的时间窗口大小（采样点数）
            update_interval: 更新间隔（毫秒）
            time_window_width: 时间窗口宽度（秒），固定显示最近N秒的数据
        """
        self.n_joints = n_joints
        self.window_size = window_size
        self.update_interval = update_interval
        self.time_window_width = time_window_width  # 固定的时间窗口宽度（秒）

        # 数据存储（使用deque实现滑动窗口）
        self.time_data = deque(maxlen=window_size)
        self.command_data = [deque(maxlen=window_size) for _ in range(n_joints)]
        self.feedback_data = [deque(maxlen=window_size) for _ in range(n_joints)]

        # 线程安全锁
        self.lock = threading.Lock()

        # 创建图形
        self.fig, self.axes = plt.subplots(3, 2, figsize=(12, 8))
        self.fig.suptitle('Joint Angles: Command vs Feedback', fontsize=14, fontweight='bold')
        self.axes = self.axes.flatten()

        # 初始化每个子图
        self.command_lines = []
        self.feedback_lines = []

        for i in range(n_joints):
            ax = self.axes[i]
            ax.set_title(f'Joint {i+1}', fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Angle (rad)', fontsize=8)
            ax.grid(True, alpha=0.3)

            # 创建两条线：命令（红色）和反馈（蓝色）
            cmd_line, = ax.plot([], [], 'r-', linewidth=1.5, label='Command', alpha=0.8)
            fb_line, = ax.plot([], [], 'b-', linewidth=1.5, label='Feedback', alpha=0.8)

            self.command_lines.append(cmd_line)
            self.feedback_lines.append(fb_line)

            ax.legend(loc='upper right', fontsize=7)
            ax.set_xlim(0, 10)  # 初始时间范围
            ax.set_ylim(-1, 1)  # 初始角度范围

        plt.tight_layout()

        # 启用交互模式
        plt.ion()
        plt.show(block=False)

        # 更新计数器
        self.update_counter = 0

    def add_data(self, time, command_angles, feedback_angles):
        """
        添加新的数据点

        Args:
            time: 当前仿真时间
            command_angles: 命令角度数组 (shape: [n_joints])
            feedback_angles: 反馈角度数组 (shape: [n_joints])
        """
        with self.lock:
            self.time_data.append(time)

            for i in range(self.n_joints):
                self.command_data[i].append(command_angles[i])
                self.feedback_data[i].append(feedback_angles[i])

    def add_data_batch(self, batch_list):
        """
        批量添加数据点（提高性能）

        Args:
            batch_list: 数据列表，每个元素是 (time, command_angles, feedback_angles)
        """
        with self.lock:
            for time, command_angles, feedback_angles in batch_list:
                self.time_data.append(time)

                for i in range(self.n_joints):
                    self.command_data[i].append(command_angles[i])
                    self.feedback_data[i].append(feedback_angles[i])

    def update_plot(self, force=False):
        """
        更新图形显示

        Args:
            force: 强制更新，忽略更新计数器（用于批量处理后的更新）
        """
        self.update_counter += 1

        # 如果不是强制更新，则检查更新频率
        # 降低更新频率，避免过度消耗资源（每1帧更新一次，实时更新）
        if not force and self.update_counter % 1 != 0:
            return

        with self.lock:
            if len(self.time_data) == 0:
                return

            # 转换为numpy数组
            time_array = np.array(self.time_data)

            # 更新每个关节的图形
            for i in range(self.n_joints):
                cmd_array = np.array(self.command_data[i])
                fb_array = np.array(self.feedback_data[i])

                # 更新线条数据
                self.command_lines[i].set_data(time_array, cmd_array)
                self.feedback_lines[i].set_data(time_array, fb_array)

                # 自动调整坐标轴范围
                ax = self.axes[i]

                # X轴：固定宽度的滚动时间窗口
                if len(time_array) > 0:
                    current_time = time_array[-1]
                    # 固定窗口宽度，整体向右滚动（不留右边距）
                    t_min = max(0, current_time - self.time_window_width)
                    t_max = current_time
                    ax.set_xlim(t_min, t_max)

                # Y轴：根据数据自动调整
                if len(cmd_array) > 0 and len(fb_array) > 0:
                    all_data = np.concatenate([cmd_array, fb_array])
                    y_min = np.min(all_data) - 0.1
                    y_max = np.max(all_data) + 0.1
                    ax.set_ylim(y_min, y_max)

        # 刷新图形（使用blit模式提高性能）
        try:
            # 只刷新需要更新的部分，而不是整个图形
            self.fig.canvas.flush_events()
        except:
            # 如果窗口被关闭，忽略错误
            pass

    def is_open(self):
        """检查窗口是否仍然打开"""
        return plt.fignum_exists(self.fig.number) # pyright: ignore[reportAttributeAccessIssue]

    def close(self):
        """关闭图形窗口"""
        plt.close(self.fig)


if __name__ == "__main__":
    """测试代码"""
    import time

    plotter = JointAnglePlotter(n_joints=6)

    print("Testing JointAnglePlotter...")
    print("Close the plot window to exit.")

    t = 0
    dt = 0.01

    try:
        while plotter.is_open():
            # 生成测试数据（正弦波）
            command_angles = np.array([
                0.3 * np.sin(2 * np.pi * 0.2 * t),
                0.2 * np.sin(2 * np.pi * 0.15 * t),
                0.15 * np.sin(2 * np.pi * 0.25 * t),
                0.25 * np.sin(2 * np.pi * 0.18 * t),
                0.2 * np.sin(2 * np.pi * 0.22 * t),
                0.1 * np.sin(2 * np.pi * 0.12 * t)
            ])

            # 反馈角度有轻微延迟和噪声
            feedback_angles = command_angles * 0.95 + np.random.normal(0, 0.01, 6)

            # 添加数据
            plotter.add_data(t, command_angles, feedback_angles)

            # 更新图形
            plotter.update_plot()

            t += dt
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nTest interrupted")

    plotter.close()
    print("Test completed")
