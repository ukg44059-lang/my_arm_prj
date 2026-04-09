"""
Plot Process Manager for Joint Angle Visualization
独立的绘图进程管理类，处理关节角度的实时绘制
"""

import os
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
from ctypes import c_bool
from typing import Optional
from joint_plotter import JointAnglePlotter


class PlotProcessManager:
    """
    管理独立的绘图进程，用于实时绘制关节角度数据

    Features:
    - 在独立进程中运行绘图，不阻塞主仿真循环
    - 批量处理数据，提高绘图效率
    - 支持自定义窗口宽度和数据缓冲大小
    - 自动清理资源
    """

    def __init__(self, n_joints: int = 6, window_size: int = 4000, time_window_width: float = 10.0):
        """
        初始化绘图进程管理器

        Args:
            n_joints: 关节数量（默认6）
            window_size: 数据缓冲区大小（默认4000个采样点）
            time_window_width: 显示窗口宽度（秒，默认10.0）
        """
        self.n_joints = n_joints
        self.window_size = window_size
        self.time_window_width = time_window_width

        self.data_queue: Optional[Queue] = None
        self.running_flag: Optional[Value] = None
        self.process: Optional[Process] = None

    def start(self) -> None:
        """启动绘图进程"""
        if self.process is not None and self.process.is_alive():
            print("⚠ Plot process already running")
            return

        # 创建进程间通信的队列和共享标志
        self.data_queue = Queue(maxsize=1000)
        self.running_flag = Value(c_bool, True)

        # 启动绘图进程
        self.process = Process(
            target=_plot_process_worker,
            args=(
                self.data_queue,
                self.running_flag,
                self.n_joints,
                self.window_size,
                self.time_window_width
            ),
            daemon=False
        )
        self.process.start()
        print(f"✓ Plot process started with PID: {self.process.pid}")
        print(f"  - Joints: {self.n_joints}")
        print(f"  - Time window: {self.time_window_width}s")
        print(f"  - Data buffer: {self.window_size} points")

    def send_data(self, sim_time: float, joint_targets, joint_feedback) -> bool:
        """
        发送关节数据到绘图进程

        Args:
            sim_time: 仿真时间
            joint_targets: 目标关节角度数组
            joint_feedback: 反馈关节角度数组

        Returns:
            是否成功发送（队列已满时返回False）
        """
        if self.data_queue is None or not self.is_running():
            return False

        try:
            # 使用非阻塞方式发送，如果队列满了就跳过
            self.data_queue.put_nowait((sim_time, joint_targets.copy(), joint_feedback))
            return True
        except:
            # 队列已满，跳过这一帧的数据
            return False

    def is_running(self) -> bool:
        """检查绘图进程是否正在运行"""
        if self.process is None:
            return False
        return self.process.is_alive()

    def stop(self) -> None:
        """停止绘图进程"""
        if self.process is None or not self.process.is_alive():
            return

        # 发送终止信号
        if self.running_flag is not None:
            self.running_flag.value = False

        if self.data_queue is not None:
            try:
                self.data_queue.put(None)
            except:
                pass

        # 等待进程结束
        self.process.join(timeout=5.0)

        if self.process.is_alive():
            print("⚠ Plot process did not terminate gracefully, terminating...")
            self.process.terminate()
            self.process.join(timeout=2.0)

        print("✓ Plot process stopped")

    def __del__(self):
        """析构函数：确保进程被正确清理"""
        self.stop()


def _plot_process_worker(data_queue: Queue, running_flag: Value, n_joints: int, window_size: int, time_window_width: float):
    """
    绘图进程的工作函数（在独立进程中运行）

    Args:
        data_queue: 接收数据的队列
        running_flag: 运行标志
        n_joints: 关节数量
        window_size: 数据缓冲区大小
        time_window_width: 时间窗口宽度
    """
    # 确保matplotlib使用正确的后端
    import matplotlib
    matplotlib.use('TkAgg')  # 使用TkAgg后端，支持在独立进程中创建窗口

    # 创建绘图器
    plotter = JointAnglePlotter(
        n_joints=n_joints,
        window_size=window_size,
        time_window_width=time_window_width
    )

    print(f"[Plot Process] Started with PID: {os.getpid()}")
    print(f"[Plot Process] Matplotlib backend: {matplotlib.get_backend()}")
    print(f"[Plot Process] Using BATCH processing mode for better performance")
    print(f"[Plot Process] Time window: {time_window_width}s (fixed width, scrolling)")
    print(f"[Plot Process] Data buffer: {window_size} points")

    try:
        while running_flag.value and plotter.is_open():
            # ====== 批量获取队列中的所有数据 ======
            batch_data = []

            try:
                # 第一次尝试阻塞获取，确保至少有一个数据
                data = data_queue.get(timeout=0.05)

                if data is None:  # None 作为终止信号
                    break

                batch_data.append(data)

                # 然后非阻塞地获取队列中的所有剩余数据
                while not data_queue.empty():
                    try:
                        data = data_queue.get_nowait()
                        if data is None:
                            running_flag.value = False
                            break
                        batch_data.append(data)
                    except:
                        break

                # 批量添加所有数据（使用批量方法提高性能）
                if batch_data:
                    plotter.add_data_batch(batch_data)

                    # 只在批量添加完所有数据后强制更新一次绘图
                    plotter.update_plot(force=True)

                    # 打印批量处理统计（可选，用于调试）
                    if len(batch_data) > 20:
                        print(f"[Plot Process] Batch processed {len(batch_data)} data points")

            except:
                # 队列为空，刷新画布以保持窗口响应
                try:
                    plotter.fig.canvas.flush_events()
                except:
                    pass
                continue

    except KeyboardInterrupt:
        print("[Plot Process] Interrupted by user")
    except Exception as e:
        print(f"[Plot Process] Error: {e}")
    finally:
        plotter.close()
        print("[Plot Process] Closed")
