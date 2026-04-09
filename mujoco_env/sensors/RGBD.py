import mujoco
from mujoco.renderer import Renderer
import numpy as np
import cv2


class MuJoCo_RGBD_Sensor:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, rgb_renderer: Renderer, depth_renderer: Renderer, cam_name: str, fps: int = 30):
        """
        初始化 MuJoCo RGBD 传感器。

        参数:
            model: MuJoCo 模型。
            data: MuJoCo 数据。
            rgb_renderer: RGB 渲染器。
            depth_renderer: 深度渲染器。
            cam_name: 相机名称。
            fps: 帧率。
        """
        self.model = model
        self.rgb_renderer = rgb_renderer
        self.rgb_renderer.disable_depth_rendering()
        self.depth_renderer = depth_renderer
        self.depth_renderer.enable_depth_rendering()
        self.data = data
        self.cam_name = cam_name
        self.fps = fps
        self.camera_id = self.model.cam(cam_name).id
        self.height = 480
        self.width = 640

        # 初始化相机内参和外参
        self.intr = self._compute_intrinsics()
        self.extr = self._compute_extrinsics()

    def _compute_intrinsics(self) -> np.ndarray:
        """
        计算相机内参矩阵。
        """
        fovy = self.model.cam_fovy[self.camera_id]
        theta = np.deg2rad(fovy)
        fy = self.height / (2.0 * np.tan(theta / 2.0))
        fx = fy
        cx = (self.width) / 2.0
        cy = (self.height) / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def _compute_extrinsics(self) -> np.ndarray:
        """
        计算相机外参矩阵。
        """
        self.base_Link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

        world_2_B = np.eye(4)
        world_2_B[:3, :3] = self.data.xmat[self.base_Link_id].reshape(3, 3)
        world_2_B[:3, 3] = self.data.xpos[self.base_Link_id]

        world_2_C = np.eye(4)
        world_2_C[:3, :3] = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        world_2_C[:3, 3] = self.data.cam_xpos[self.camera_id]

        # T_base2cam = R_b.T @ R_c, and T = R_b.T @ (t_c - t_b)
        R_base2cam = world_2_B[:3, :3].T @ world_2_C[:3, :3]
        t_base2cam = world_2_B[:3, :3].T @ (world_2_C[:3, 3] - world_2_B[:3, 3])

        # Construct full 4x4 transformation matrix
        extr = np.eye(4)
        extr[:3, :3] = R_base2cam
        extr[:3, 3] = t_base2cam
        return extr

    def render_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        """
        渲染 RGB-D 图像。

        返回:
            rgb: RGB 图像 (height, width, 3)
            depth: 深度图像 (height, width)，单位：米
        """
        self.rgb_renderer.update_scene(self.data, camera=self.camera_id)
        self.depth_renderer.update_scene(self.data, camera=self.camera_id)

        depth = self.depth_renderer.render()
        rgb = self.rgb_renderer.render()
        return rgb, depth

    def rgbd_to_pointcloud(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        将 RGB-D 图像转换为点云。

        参数:
            rgb: RGB 图像。
            depth: 深度图像。

        返回:
            xyzrgb: 点云数据（包含 XYZ 和 RGB），形状 (N, 6)
        """
        cc, rr = np.meshgrid(np.arange(self.width), np.arange(self.height), sparse=True)
        valid = (depth > 0) & (depth < 5.0)  # 有效深度范围
        z = np.where(valid, depth, np.nan)
        x = np.where(valid, z * (cc - self.intr[0, 2]) / self.intr[0, 0], 0)
        y = np.where(valid, z * (rr - self.intr[1, 2]) / self.intr[1, 1], 0)
        xyz = np.vstack([e.flatten() for e in [x, y, z]]).T
        color = rgb.transpose([2, 0, 1]).reshape((3, -1)).T / 255.0
        mask = np.isnan(xyz[:, 2])
        xyz = xyz[~mask]
        color = color[~mask]
        xyzrgb = np.hstack([xyz[:, :3], color])
        return xyzrgb

    def display_rgb(self, rgb: np.ndarray, window_name: str = "RGB"):
        """
        使用 OpenCV 显示 RGB 图像。

        参数:
            rgb: RGB 图像 (height, width, 3)
            window_name: 窗口名称
        """
        # 将 RGB 转换为 BGR（OpenCV 默认使用 BGR）
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, bgr)

    def display_depth(self, depth: np.ndarray, window_name: str = "Depth"):
        """
        使用 OpenCV 显示深度图像（归一化为灰度图）。

        参数:
            depth: 深度图像 (height, width)，单位：米
            window_name: 窗口名称
        """
        # 归一化深度图到 0-255
        depth_valid = depth[depth > 0]
        if len(depth_valid) > 0:
            min_depth = np.min(depth_valid)
            max_depth = np.max(depth_valid)
            depth_normalized = np.where(
                depth > 0,
                ((depth - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8),
                0
            )
        else:
            depth_normalized = np.zeros_like(depth, dtype=np.uint8)

        # 应用彩色map
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        cv2.imshow(window_name, depth_colored)

    def display_rgbd(self, rgb: np.ndarray, depth: np.ndarray, rgb_window: str = "RGB", depth_window: str = "Depth"):
        """
        同时显示 RGB 和深度图像。

        参数:
            rgb: RGB 图像
            depth: 深度图像
            rgb_window: RGB 窗口名称
            depth_window: 深度窗口名称
        """
        self.display_rgb(rgb, rgb_window)
        self.display_depth(depth, depth_window)

    @staticmethod
    def wait_for_key(delay: int = 1) -> int:
        """
        等待按键输入。

        参数:
            delay: 延迟时间（毫秒），0 表示无限等待

        返回:
            按键 ASCII 码，-1 表示无按键
        """
        return cv2.waitKey(delay)

    @staticmethod
    def close_all_windows():
        """关闭所有 OpenCV 窗口。"""
        cv2.destroyAllWindows()

