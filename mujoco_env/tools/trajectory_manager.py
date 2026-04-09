"""
Trajectory Manager for Recording and Saving End-Effector Trajectories
独立的轨迹管理类，处理轨迹记录、存储和导出功能
"""

import os
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional


class TrajectoryManager:
    """
    管理机械臂末端执行器轨迹的记录、存储和导出

    Features:
    - 实时记录轨迹数据 (时间 + 位置 + 姿态四元数)
    - 暂停/恢复记录
    - 导出为txt文件（带格式说明）
    - 数据统计（点数、时间范围等）
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化轨迹管理器

        Args:
            output_dir: 轨迹文件输出目录，默认为当前脚本所在目录
        """
        self.output_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
        self.trajectory_data: List[List[float]] = []
        self.is_recording = False
        self.recording_start_time = None

    def start_recording(self) -> None:
        """开始记录轨迹"""
        self.is_recording = True
        self.trajectory_data = []  # 清空之前的数据
        self.recording_start_time = None
        print("▶ Trajectory recording started")

    def stop_recording(self) -> None:
        """停止记录轨迹"""
        self.is_recording = False
        print(f"⏹ Trajectory recording stopped ({len(self.trajectory_data)} data points)")

    def record_point(self, time: float, ee_pos: np.ndarray, ee_quat: np.ndarray) -> None:
        """
        记录单个轨迹点

        Args:
            time: 时间戳（秒）
            ee_pos: 末端执行器位置 [x, y, z]
            ee_quat: 末端执行器四元数 [w, x, y, z]
        """
        if not self.is_recording:
            return

        if self.recording_start_time is None:
            self.recording_start_time = time

        # 记录相对时间和绝对位置
        relative_time = time - self.recording_start_time
        data_point = [
            relative_time,                          # 相对时间
            ee_pos[0], ee_pos[1], ee_pos[2],       # 位置 (x, y, z)
            ee_quat[0], ee_quat[1],                # 四元数 (w, x)
            ee_quat[2], ee_quat[3]                 # 四元数 (y, z)
        ]
        self.trajectory_data.append(data_point)

    def save_to_file(self, filename: Optional[str] = None) -> Optional[str]:
        """
        将轨迹数据保存到文件

        Args:
            filename: 输出文件名，默认使用时间戳

        Returns:
            保存的文件路径，或None（如果失败）
        """
        if not self.trajectory_data:
            print("⚠ Trajectory data is empty, cannot save")
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.txt"

        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w') as f:
                # 写入文件头
                f.write("# End-effector Trajectory Data\n")
                f.write("# Format: time(s) x(m) y(m) z(m) qw qx qy qz\n")
                f.write("# Description:\n")
                f.write("#   time(s): Relative time since recording started\n")
                f.write("#   x,y,z: End-effector position in meters\n")
                f.write("#   qw,qx,qy,qz: Quaternion representing end-effector orientation\n")
                f.write("#" + "-" * 70 + "\n")

                # 写入数据
                for data_point in self.trajectory_data:
                    f.write(f"{data_point[0]:.6f} ")
                    f.write(f"{data_point[1]:.6f} {data_point[2]:.6f} {data_point[3]:.6f} ")
                    f.write(f"{data_point[4]:.6f} {data_point[5]:.6f} {data_point[6]:.6f} {data_point[7]:.6f}\n")

            print(f"✓ Trajectory saved to: {filepath}")
            self._print_statistics()
            return filepath

        except Exception as e:
            print(f"✗ Failed to save trajectory: {e}")
            return None

    def clear_data(self) -> None:
        """清空轨迹数据"""
        self.trajectory_data = []
        self.is_recording = False
        self.recording_start_time = None
        print("🗑 Trajectory data cleared")

    def get_data_count(self) -> int:
        """获取当前记录的数据点数"""
        return len(self.trajectory_data)

    def get_time_range(self) -> Tuple[float, float]:
        """
        获取轨迹的时间范围

        Returns:
            (start_time, end_time) 元组，如果没有数据返回 (0, 0)
        """
        if not self.trajectory_data:
            return 0, 0
        return self.trajectory_data[0][0], self.trajectory_data[-1][0]

    def get_position_bounds(self) -> dict:
        """
        获取轨迹数据的位置边界

        Returns:
            字典包含 'x_range', 'y_range', 'z_range'
        """
        if not self.trajectory_data:
            return {"x_range": (0, 0), "y_range": (0, 0), "z_range": (0, 0)}

        positions = np.array([point[1:4] for point in self.trajectory_data])

        return {
            "x_range": (positions[:, 0].min(), positions[:, 0].max()),
            "y_range": (positions[:, 1].min(), positions[:, 1].max()),
            "z_range": (positions[:, 2].min(), positions[:, 2].max())
        }

    def _print_statistics(self) -> None:
        """打印轨迹统计信息"""
        if not self.trajectory_data:
            return

        print(f"  - Data points: {len(self.trajectory_data)}")
        start_time, end_time = self.get_time_range()
        print(f"  - Time range: {start_time:.2f}s - {end_time:.2f}s ({end_time - start_time:.2f}s duration)")

        bounds = self.get_position_bounds()
        print(f"  - Position ranges:")
        print(f"    • X: {bounds['x_range'][0]:.4f}m to {bounds['x_range'][1]:.4f}m")
        print(f"    • Y: {bounds['y_range'][0]:.4f}m to {bounds['y_range'][1]:.4f}m")
        print(f"    • Z: {bounds['z_range'][0]:.4f}m to {bounds['z_range'][1]:.4f}m")
