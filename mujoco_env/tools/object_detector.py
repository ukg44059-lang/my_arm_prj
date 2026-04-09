#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge


class PurpleObjectDetector(Node):
    def __init__(self) -> None:
        super().__init__('purple_object_detector')

        self.bridge = CvBridge()

        # 订阅 RGB / Depth / CameraInfo
        self.rgb_sub = self.create_subscription(
            Image,
            '/external_camera/rgb/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/external_camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/external_camera/camera_info',
            self.camera_info_callback,
            10
        )

        # 发布 2D 中心点
        self.center_pub = self.create_publisher(
            Point,
            '/purple_object_center',
            10
        )

        # 发布 3D 点
        self.point3d_pub = self.create_publisher(
            PointStamped,
            '/purple_object_3d',
            10
        )

        # 缓存最新深度图和相机内参
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_encoding: Optional[str] = None

        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx0: Optional[float] = None
        self.cy0: Optional[float] = None

        self.last_log_counter = 0

        self.get_logger().info('Purple object detector started.')
        self.get_logger().info('Subscribed RGB: /external_camera/rgb/image_raw')
        self.get_logger().info('Subscribed Depth: /external_camera/depth/image_raw')
        self.get_logger().info('Subscribed CameraInfo: /external_camera/camera_info')
        self.get_logger().info('Publishing 2D center: /purple_object_center')
        self.get_logger().info('Publishing 3D point: /purple_object_3d')

    # ----------------------------
    # 回调
    # ----------------------------
    def camera_info_callback(self, msg: CameraInfo) -> None:
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx0 = float(msg.k[2])
        self.cy0 = float(msg.k[5])

    def depth_callback(self, msg: Image) -> None:
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )
            self.latest_depth_encoding = msg.encoding
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')

    def rgb_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB conversion failed: {e}')
            return

        debug_image = frame.copy()

        # 1. 颜色分割：BGR -> HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 紫色范围（先给一组通用值，后续可调）
        lower_purple = np.array([125, 80, 40], dtype=np.uint8)
        upper_purple = np.array([165, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_purple, upper_purple)

        # 2. 形态学去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 3. 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(largest))

            # 过滤小噪声
            if area > 150.0:
                found = True

                x, y, w, h = cv2.boundingRect(largest)
                cx = x + w // 2
                cy = y + h // 2

                # 发布 2D 中心点
                center_msg = Point()
                center_msg.x = float(cx)
                center_msg.y = float(cy)
                center_msg.z = 0.0
                self.center_pub.publish(center_msg)

                # 画框
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(debug_image, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    debug_image,
                    f'purple ({cx}, {cy}) area={area:.0f}',
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # 4. 读取该区域深度
                depth_value = None
                point_3d = None

                if self.latest_depth is not None:
                    depth_value = self.get_depth_median(self.latest_depth, cx, cy, win=3)

                if depth_value is not None:
                    cv2.putText(
                        debug_image,
                        f'Z={depth_value:.3f}m',
                        (x, min(debug_image.shape[0] - 10, y + h + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                # 5. 还原三维坐标
                if (
                    depth_value is not None and
                    self.fx is not None and
                    self.fy is not None and
                    self.cx0 is not None and
                    self.cy0 is not None
                ):
                    point_3d = self.pixel_to_3d(cx, cy, depth_value)

                    if point_3d is not None:
                        X, Y, Z = point_3d

                        point_msg = PointStamped()
                        point_msg.header = msg.header
                        point_msg.point.x = float(X)
                        point_msg.point.y = float(Y)
                        point_msg.point.z = float(Z)
                        self.point3d_pub.publish(point_msg)

                        cv2.putText(
                            debug_image,
                            f'3D=({X:.3f}, {Y:.3f}, {Z:.3f})m',
                            (x, min(debug_image.shape[0] - 30, y + h + 45)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 0),
                            2
                        )

                        self.rate_limited_log(
                            f'Purple center=({cx},{cy}), depth={depth_value:.3f}m, '
                            f'3D=({X:.3f}, {Y:.3f}, {Z:.3f})'
                        )
                else:
                    if self.latest_depth is None:
                        self.rate_limited_log('Waiting for depth image...')
                    elif self.fx is None:
                        self.rate_limited_log('Waiting for camera_info...')

        if not found:
            self.rate_limited_log('No purple object detected.')

        # 显示窗口
        cv2.imshow('Purple Detector - RGB', debug_image)
        cv2.imshow('Purple Detector - Mask', mask)
        cv2.waitKey(1)

    # ----------------------------
    # 工具函数
    # ----------------------------
    def get_depth_median(
        self,
        depth_image: np.ndarray,
        u: int,
        v: int,
        win: int = 3
    ) -> Optional[float]:
        """
        在 (u, v) 周围取一个小窗口，返回有效深度的中位数。
        返回单位统一为“米”。
        """
        if depth_image is None:
            return None

        h, w = depth_image.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None

        u_min = max(0, u - win)
        u_max = min(w, u + win + 1)
        v_min = max(0, v - win)
        v_max = min(h, v + win + 1)

        patch = depth_image[v_min:v_max, u_min:u_max]

        # 转成 float，便于统一处理
        patch = patch.astype(np.float32)

        # 去掉 nan / inf
        valid = patch[np.isfinite(patch)]

        # 去掉 <= 0 的无效值
        valid = valid[valid > 0.0]

        if valid.size == 0:
            return None

        depth = float(np.median(valid))

        # 若原始深度图是 uint16，通常单位为毫米，转成米
        if depth_image.dtype == np.uint16:
            depth = depth / 1000.0

        # 简单的合理范围过滤，可按项目再调
        if not (0.01 <= depth <= 20.0):
            return None

        return depth

    def pixel_to_3d(self, u: int, v: int, depth_m: float) -> Optional[Tuple[float, float, float]]:
        """
        将图像像素坐标 + 深度，转换成相机坐标系下的三维点。
        """
        if None in (self.fx, self.fy, self.cx0, self.cy0):
            return None

        X = (float(u) - self.cx0) * depth_m / self.fx
        Y = (float(v) - self.cy0) * depth_m / self.fy
        Z = depth_m
        return X, Y, Z

    def rate_limited_log(self, text: str, interval: int = 20) -> None:
        """
        限频打印，避免终端刷爆。
        每 interval 帧打印一次。
        """
        self.last_log_counter += 1
        if self.last_log_counter % interval == 0:
            self.get_logger().info(text)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PurpleObjectDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()