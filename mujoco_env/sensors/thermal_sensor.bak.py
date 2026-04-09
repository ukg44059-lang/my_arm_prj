import mujoco
from mujoco import renderer
import numpy as np
import cv2


class ThermalSensor:
    def __init__(
        self, 
        model: mujoco.MjModel, 
        data: mujoco.MjData, 
        cam_name: str,
        width: int = 640,
        height: int = 480,
        body_temperatures: dict = None,
        ambient_temperature: float = 25.0,
        enable_distance_attenuation: bool = True,
        attenuation_coefficient: float = 0.1,
        enable_noise: bool = True,
        noise_stddev: float = 0.5
    ):
        """
        基于深度渲染器的红外热传感器（使用geom_id缓冲）。
        
        参数:
            model: MuJoCo 模型
            data: MuJoCo 数据
            cam_name: 相机名称
            width: 图像宽度
            height: 图像高度
            body_temperatures: body_id -> 温度(°C) 的字典，默认所有物体25°C
            ambient_temperature: 环境温度(°C)
            enable_distance_attenuation: 是否启用距离衰减
            attenuation_coefficient: 衰减系数（越大衰减越快）
            enable_noise: 是否启用噪声
            noise_stddev: 噪声标准差(°C)
        """
        self.model = model
        self.data = data
        self.cam_name = cam_name
        self.camera_id = self.model.cam(cam_name).id
        self.width = width
        self.height = height
        self.ambient_temperature = ambient_temperature
        self.enable_distance_attenuation = enable_distance_attenuation
        self.attenuation_coefficient = attenuation_coefficient
        self.enable_noise = enable_noise
        self.noise_stddev = noise_stddev
        
        self.renderer = renderer.Renderer(model, height=height, width=width)
        
        if body_temperatures is None:
            self.body_temperatures = {i: ambient_temperature for i in range(model.nbody)}
        else:
            self.body_temperatures = body_temperatures
        
    def set_body_temperature(self, body_name: str, temperature: float):
        """设置指定物体的温度"""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self.body_temperatures[body_id] = temperature
        except:
            print(f"Warning: Body '{body_name}' not found in model")
    
    def set_body_temperature_by_id(self, body_id: int, temperature: float):
        """通过body_id设置温度"""
        if 0 <= body_id < self.model.nbody:
            self.body_temperatures[body_id] = temperature
        else:
            print(f"Warning: Invalid body_id {body_id}")
    
    def render_thermal_image(self, debug=False) -> tuple[np.ndarray, np.ndarray]:
        """
        渲染热成像图像（使用MuJoCo的segmentation功能）
        
        返回:
            temperature_map: 温度图 (height, width)，单位°C
            body_id_map: body_id图 (height, width)，-1表示背景
        """
        self.renderer.update_scene(self.data, camera=self.camera_id)
        
        self.renderer.enable_segmentation_rendering()
        seg_image = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        
        self.renderer.enable_depth_rendering()
        depth_image = self.renderer.render()
        self.renderer.disable_depth_rendering()
        
        temperature_map = np.full((self.height, self.width), self.ambient_temperature, dtype=np.float32)
        body_id_map = np.full((self.height, self.width), -1, dtype=np.int32)
        
        if debug:
            print(f"分割图像形状: {seg_image.shape}, dtype: {seg_image.dtype}")
            print(f"深度图像形状: {depth_image.shape}, 范围: [{depth_image.min():.3f}, {depth_image.max():.3f}]")
            print(f"分割ID范围: [{seg_image.min()}, {seg_image.max()}]")
            unique_seg_ids = np.unique(seg_image)
            print(f"唯一的seg_id数量: {len(unique_seg_ids)}, 值: {unique_seg_ids[:20]}")
        
        geom_id_image = seg_image[:, :, 0].astype(np.int32)
        
        for v in range(self.height):
            for u in range(self.width):
                geom_id = geom_id_image[v, u]
                
                if 0 <= geom_id < self.model.ngeom:
                    body_id = self.model.geom_bodyid[geom_id]
                    body_id_map[v, u] = body_id
                    
                    object_temp = self.body_temperatures.get(body_id, self.ambient_temperature)
                    distance = depth_image[v, u]
                    
                    if self.enable_distance_attenuation and distance > 0:
                        temp_diff = object_temp - self.ambient_temperature
                        attenuated_diff = temp_diff * np.exp(-self.attenuation_coefficient * distance)
                        measured_temp = self.ambient_temperature + attenuated_diff
                    else:
                        measured_temp = object_temp
                    
                    temperature_map[v, u] = measured_temp
        
        if self.enable_noise:
            noise = np.random.normal(0, self.noise_stddev, (self.height, self.width))
            temperature_map += noise
        
        unique_bodies_found = set(body_id_map[body_id_map >= 0].flatten())
        
        if debug:
            print(f"检测到的body数量: {len(unique_bodies_found)}")
            for body_id in sorted(unique_bodies_found):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name is None:
                    body_name = f"body_{body_id}"
                object_temp = self.body_temperatures.get(body_id, self.ambient_temperature)
                mask = body_id_map == body_id
                measured_temps = temperature_map[mask]
                avg_measured = measured_temps.mean()
                avg_distance = depth_image[mask].mean() if mask.sum() > 0 else 0
                pixel_count = mask.sum()
                print(f"  Body {body_id} ({body_name}): 设定={object_temp:.1f}°C, "
                      f"测量平均={avg_measured:.1f}°C, 平均距离={avg_distance:.3f}m, 像素数={pixel_count}")
        
        return temperature_map, body_id_map
    
    def temperature_to_grayscale(self, temperature_map: np.ndarray,
                                temp_min: float = 0.0,
                                temp_max: float = 100.0) -> np.ndarray:
        """将温度图转换为灰度图（0-100度映射到0-255灰度）"""
        grayscale = np.clip(
            (temperature_map - temp_min) / (temp_max - temp_min) * 255,
            0,
            255
        ).astype(np.uint8)
        return grayscale
    
    def temperature_to_color(self, temperature_map: np.ndarray, 
                            temp_min: float = 15.0, 
                            temp_max: float = 100.0) -> np.ndarray:
        """将温度图转换为伪彩色热成像"""
        temp_normalized = np.clip(
            (temperature_map - temp_min) / (temp_max - temp_min) * 255, 
            0, 
            255
        ).astype(np.uint8)
        
        thermal_image = cv2.applyColorMap(temp_normalized, cv2.COLORMAP_JET)
        thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)
        
        return thermal_image
    
    def display_thermal(self, thermal_image: np.ndarray, window_name: str = "Thermal Camera"):
        """显示热成像图像"""
        bgr = cv2.cvtColor(thermal_image, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, bgr)
    
    @staticmethod
    def wait_for_key(delay: int = 1) -> int:
        """等待按键输入"""
        return cv2.waitKey(delay)
    
    @staticmethod
    def close_all_windows():
        """关闭所有OpenCV窗口"""
        cv2.destroyAllWindows()
