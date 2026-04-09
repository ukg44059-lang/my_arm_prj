"""
Teaching Status Subscriber for trajectory recording
监听 /teaching_status 话题，控制轨迹记录和视频录制
"""
from typing import Optional, Callable
import numpy as np
import threading
import time
import os
import json
from datetime import datetime

ROS2_AVAILABLE = False
_import_error = ""
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError as e:
    _import_error = str(e)
    ROS2_AVAILABLE = False

def ros2_available() -> bool:
    return ROS2_AVAILABLE

class TeachingStatusSubscriber:
    """订阅 /teaching_status 话题控制轨迹记录"""
    
    def __init__(
        self,
        topic: str = "/teaching_status",
        node_name: str = "teaching_status_subscriber",
        callback: Optional[Callable[[str], None]] = None
    ) -> None:
        if not ROS2_AVAILABLE:
            raise ImportError(
                f"ROS2 not available: {_import_error}\n"
                "Please source your ROS2 environment"
            )
        
        self.callback = callback
        self.latest_status = None
        self.last_update_time = None
        self.lock = threading.Lock()
        
        # 为每个订阅器创建独立的ROS2上下文
        self.context = rclpy.Context()
        rclpy.init(context=self.context)
        
        self.node = Node(node_name, context=self.context)
        self.subscription = self.node.create_subscription(
            String,
            topic,
            self._status_callback,
            10
        )
        
        # 创建独立的执行器
        self.executor = rclpy.executors.SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        
        self.running = False
        self.spin_thread = None
        
        print(f"✓ Teaching Status Subscriber initialized")
        print(f"  - Topic: {topic}")
        print(f"  - Node: {node_name}")
    
    def _status_callback(self, msg):
        """内部回调处理teaching status消息"""
        with self.lock:
            self.latest_status = msg.data.strip()
            self.last_update_time = time.time()
            
            print(f"📚 Teaching Status: {self.latest_status}")
            
            # 调用外部回调
            if self.callback:
                self.callback(self.latest_status)
    
    def get_latest_status(self) -> Optional[str]:
        """获取最新的teaching status"""
        with self.lock:
            return self.latest_status
    
    def start_spinning(self):
        """开始spinning ROS2节点"""
        if self.running:
            print("⚠ Teaching Status Subscriber already running")
            return
        
        self.running = True
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()
        print("✓ Teaching Status Subscriber started")
    
    def stop_spinning(self):
        """停止spinning ROS2节点"""
        if not self.running:
            return
        
        self.running = False
        if self.spin_thread:
            self.spin_thread.join(timeout=1.0)
            self.spin_thread = None
        print("✓ Teaching Status Subscriber stopped")
    
    def _spin_loop(self):
        """主spin循环"""
        while self.running:
            try:
                # 使用独立的执行器避免冲突
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                if self.running:  # 只在运行时报错，避免关闭时的无关错误
                    print(f"❌ Teaching Status spin error: {e}")
                break
    
    def shutdown(self):
        """清理订阅器"""
        self.stop_spinning()
        
        try:
            if hasattr(self, 'executor'):
                self.executor.remove_node(self.node)
                self.executor.shutdown()
        except Exception:
            pass
        
        try:
            self.subscription.destroy()
            self.node.destroy_node()
        except Exception:
            pass
        
        try:
            if hasattr(self, 'context'):
                rclpy.shutdown(context=self.context)
        except Exception:
            pass
        
        print("✓ Teaching Status Subscriber shutdown complete")