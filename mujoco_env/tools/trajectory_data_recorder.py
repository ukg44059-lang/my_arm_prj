"""
轨迹数据记录器
记录机器人关节轨迹、末端位置、图像等数据
"""
import os
import time
import json
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Optional, Any
import threading

class TrajectoryDataRecorder:
    """轨迹数据记录器类"""
    
    def __init__(self, base_data_dir: str = "data"):
        self.base_data_dir = base_data_dir
        self.current_recording_dir = None
        self.current_session_name = None
        
        # 数据存储
        self.joint_data = []
        self.ee_position_data = []
        self.ee_orientation_data = []
        self.timestamps = []
        
        # 多摄像头视频录制
        self.video_writers = {}  # {camera_name: video_writer}
        self.video_filenames = {}  # {camera_name: filename}
        self.camera_frame_counts = {}  # {camera_name: frame_count}
        
        # 记录状态
        self.is_recording = False
        self.recording_lock = threading.Lock()
        
        # 确保数据目录存在
        os.makedirs(base_data_dir, exist_ok=True)
        print(f"✓ 轨迹数据记录器初始化 - 数据目录: {base_data_dir}")
    
    def start_recording(self, session_name: Optional[str] = None) -> str:
        """
        开始记录轨迹数据
        
        Args:
            session_name: 会话名称，如果为None则自动生成
            
        Returns:
            str: 记录会话目录路径
        """
        with self.recording_lock:
            if self.is_recording:
                print("⚠ 已在记录中，请先停止当前记录")
                return self.current_recording_dir
            
            # 生成会话名称
            if session_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_name = f"teaching_{timestamp}"
            
            self.current_session_name = session_name
            self.current_recording_dir = os.path.join(self.base_data_dir, session_name)
            
            # 创建会话目录结构
            os.makedirs(self.current_recording_dir, exist_ok=True)
            os.makedirs(os.path.join(self.current_recording_dir, "images"), exist_ok=True)
            
            # 为每个摄像头创建子目录
            camera_names = ['ee_rgb', 'ee_depth', 'external_rgb', 'external_depth']
            for cam_name in camera_names:
                cam_dir = os.path.join(self.current_recording_dir, cam_name)
                os.makedirs(cam_dir, exist_ok=True)
            
            # 重置数据
            self.joint_data = []
            self.ee_position_data = []
            self.ee_orientation_data = []
            self.timestamps = []
            
            # 重置视频录制器
            self.video_writers = {}
            self.video_filenames = {}
            self.camera_frame_counts = {}
            
            self.is_recording = True
            
            print(f"🎬 开始Teaching录制")
            print(f"   会话名称: {session_name}")
            print(f"   存储目录: {self.current_recording_dir}")
            print(f"   数据结构:")
            print(f"     data/")
            print(f"     └── {session_name}/")
            print(f"         ├── ee_rgb/              # 末端RGB视频")
            print(f"         ├── ee_depth/            # 末端深度视频")
            print(f"         ├── external_rgb/        # 外部RGB视频")
            print(f"         ├── external_depth/      # 外部深度视频")
            print(f"         ├── trajectory_data.json # 轨迹数据")
            print(f"         ├── trajectory_data.npz  # NumPy格式")
            print(f"         └── images/              # 关键帧图像")
            
            return self.current_recording_dir
    
    def stop_recording(self) -> Optional[str]:
        """
        停止记录并保存数据
        
        Returns:
            str: 保存的数据目录路径，如果没有在记录则返回None
        """
        with self.recording_lock:
            if not self.is_recording:
                print("⚠ 当前未在记录中")
                return None
            
            self.is_recording = False
            
            print(f"🛑 停止记录轨迹数据...")
            
            # 关闭所有视频写入器
            for cam_name, writer in self.video_writers.items():
                if writer is not None:
                    writer.release()
                    frame_count = self.camera_frame_counts.get(cam_name, 0)
                    print(f"✓ {cam_name}视频已保存: {self.video_filenames[cam_name]} ({frame_count}帧)")
            
            self.video_writers = {}
            self.video_filenames = {}
            self.camera_frame_counts = {}
            
            # 保存轨迹数据
            self._save_trajectory_data()
            
            recording_dir = self.current_recording_dir
            
            # 清理状态
            self.current_recording_dir = None
            self.current_session_name = None
            
            print(f"✅ 轨迹数据记录完成")
            print(f"   数据总数: {len(self.joint_data)} 帧")
            print(f"   存储位置: {recording_dir}")
            
            return recording_dir
    
    def record_frame(
        self,
        joint_positions: np.ndarray,
        ee_position: np.ndarray,
        ee_orientation: np.ndarray,
        camera_images: Optional[Dict[str, np.ndarray]] = None,
        timestamp: Optional[float] = None
    ):
        """
        记录一帧数据
        
        Args:
            joint_positions: 关节位置
            ee_position: 末端位置
            ee_orientation: 末端姿态
            camera_images: 摄像头图像字典 {camera_name: image_array}
                          支持的camera_name: 'ee_rgb', 'ee_depth', 'external_rgb', 'external_depth'
            timestamp: 时间戳
        """
        if not self.is_recording:
            return
        
        if timestamp is None:
            timestamp = time.time()
        
        # 存储轨迹数据（这些数据总是记录）
        self.joint_data.append(joint_positions.copy())
        self.ee_position_data.append(ee_position.copy())
        self.ee_orientation_data.append(ee_orientation.copy())
        self.timestamps.append(timestamp)
        
        # 处理多摄像头图像
        if camera_images is not None and isinstance(camera_images, dict):
            for cam_name, image in camera_images.items():
                if image is None:
                    continue
                    
                try:
                    # 确保该摄像头的视频写入器已初始化
                    if cam_name not in self.video_writers:
                        self._init_video_writer(cam_name, image.shape)
                    
                    # 写入视频帧
                    writer = self.video_writers.get(cam_name)
                    if writer is not None and writer.isOpened():
                        # 处理不同类型的图像
                        if 'depth' in cam_name:
                            # 深度图：归一化到0-255并转为伪彩色
                            depth_normalized = self._normalize_depth_for_video(image)
                            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                            writer.write(depth_colored)
                        else:
                            # RGB图：转换为BGR格式
                            if len(image.shape) == 3 and image.shape[2] == 3:
                                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                writer.write(bgr_image)
                        
                        # 更新帧计数
                        if cam_name not in self.camera_frame_counts:
                            self.camera_frame_counts[cam_name] = 0
                        self.camera_frame_counts[cam_name] += 1
                        
                except Exception as e:
                    if not hasattr(self, f'_video_error_count_{cam_name}'):
                        setattr(self, f'_video_error_count_{cam_name}', 0)
                    error_count = getattr(self, f'_video_error_count_{cam_name}')
                    setattr(self, f'_video_error_count_{cam_name}', error_count + 1)
                    if error_count % 20 == 0:
                        print(f"❌ {cam_name}视频录制错误 #{error_count}: {e}")
        
        # 定期报告记录状态
        if len(self.joint_data) % 500 == 0:
            frame_count = len(self.joint_data)
            video_info = ', '.join([f"{name}:{count}" for name, count in self.camera_frame_counts.items()])
            print(f"📊 已记录 {frame_count} 帧轨迹数据, 视频帧: {video_info}")
    
    def _normalize_depth_for_video(self, depth_img):
        """归一化深度图用于视频保存"""
        if depth_img is None:
            return None
        depth_clean = depth_img.copy()
        depth_clean[np.isinf(depth_clean)] = 0
        if depth_clean.max() > depth_clean.min():
            depth_normalized = ((depth_clean - depth_clean.min()) / 
                              (depth_clean.max() - depth_clean.min()) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_clean, dtype=np.uint8)
        return depth_normalized
    
    def _init_video_writer(self, camera_name: str, image_shape):
        """初始化指定摄像头的视频写入器"""
        try:
            height, width = image_shape[:2]
            
            # 为该摄像头创建视频文件
            video_filename = os.path.join(
                self.current_recording_dir, 
                camera_name,
                f"{camera_name}.mp4"
            )
            
            print(f"📹 初始化{camera_name}视频录制器: {width}x{height}")
            
            # 使用MJPG编码器
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(
                video_filename,
                fourcc,
                20.0,  # 20 FPS
                (width, height)
            )
            
            if writer.isOpened():
                self.video_writers[camera_name] = writer
                self.video_filenames[camera_name] = video_filename
                self.camera_frame_counts[camera_name] = 0
                print(f"✅ {camera_name}视频写入器初始化成功")
                return True
            else:
                print(f"❌ {camera_name}视频写入器初始化失败")
                writer.release()
                return False
                
        except Exception as e:
            print(f"❌ {camera_name}视频写入器初始化错误: {e}")
            return False
    
    def _save_trajectory_data(self):
        """保存轨迹数据到JSON文件"""
        try:
            # 准备数据
            trajectory_data = {
                "session_name": self.current_session_name,
                "recording_start_time": self.timestamps[0] if self.timestamps else 0,
                "recording_end_time": self.timestamps[-1] if self.timestamps else 0,
                "total_frames": len(self.joint_data),
                "camera_frame_counts": self.camera_frame_counts,
                "data": {
                    "timestamps": self.timestamps,
                    "joint_positions": [pos.tolist() for pos in self.joint_data],
                    "ee_positions": [pos.tolist() for pos in self.ee_position_data],
                    "ee_orientations": [quat.tolist() for quat in self.ee_orientation_data]
                }
            }
            
            # 保存到JSON文件
            json_filename = os.path.join(self.current_recording_dir, "trajectory_data.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ 轨迹数据已保存: {json_filename}")
            
            # 保存numpy格式（便于后续处理）
            np_filename = os.path.join(self.current_recording_dir, "trajectory_data.npz")
            np.savez_compressed(
                np_filename,
                timestamps=np.array(self.timestamps),
                joint_positions=np.array(self.joint_data),
                ee_positions=np.array(self.ee_position_data),
                ee_orientations=np.array(self.ee_orientation_data)
            )
            
            print(f"✓ NumPy数据已保存: {np_filename}")
            
        except Exception as e:
            print(f"❌ 保存轨迹数据时出错: {e}")
    
    def get_recording_status(self) -> Dict[str, Any]:
        """获取记录状态"""
        with self.recording_lock:
            return {
                "is_recording": self.is_recording,
                "session_name": self.current_session_name,
                "recording_dir": self.current_recording_dir,
                "frames_recorded": len(self.joint_data),
                "video_file": self.video_filename if self.video_filename else None
            }