"""
PID配置文件加载器
支持YAML和JSON格式的PID参数配置文件
"""

import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


class PIDConfigLoader:
    """PID配置文件加载和保存工具类"""

    @staticmethod
    def load_from_yaml(file_path: str) -> Dict[str, Any]:
        """
        从YAML文件加载PID配置

        Args:
            file_path: YAML配置文件路径

        Returns:
            包含PID参数的字典
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"✓ Loaded PID config from: {file_path}")
        return PIDConfigLoader._process_config(config)

    @staticmethod
    def load_from_json(file_path: str) -> Dict[str, Any]:
        """
        从JSON文件加载PID配置

        Args:
            file_path: JSON配置文件路径

        Returns:
            包含PID参数的字典
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"✓ Loaded PID config from: {file_path}")
        return PIDConfigLoader._process_config(config)

    @staticmethod
    def save_to_yaml(config: Dict[str, Any], file_path: str):
        """
        保存PID配置到YAML文件

        Args:
            config: PID配置字典
            file_path: 保存路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换numpy数组为列表
        config_to_save = PIDConfigLoader._prepare_for_save(config)

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)

        print(f"✓ Saved PID config to: {file_path}")

    @staticmethod
    def save_to_json(config: Dict[str, Any], file_path: str):
        """
        保存PID配置到JSON文件

        Args:
            config: PID配置字典
            file_path: 保存路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换numpy数组为列表
        config_to_save = PIDConfigLoader._prepare_for_save(config)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved PID config to: {file_path}")

    @staticmethod
    def _process_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理配置文件，提取PID参数

        Args:
            config: 原始配置字典

        Returns:
            处理后的配置字典
        """
        processed = {}

        # 提取全局设置
        if 'global' in config:
            processed['global'] = config['global']

        # 提取关节配置
        if 'joints' in config:
            joints = config['joints']
            processed['joints'] = []

            for joint in joints:
                processed['joints'].append({
                    'index': joint['index'],
                    'name': joint.get('name', f"Joint{joint['index']}"),
                    'kp': float(joint['kp']),
                    'kd': float(joint['kd']),
                    'ki': float(joint['ki']),
                })

        # 提取积分限制
        if 'integral_limits' in config:
            processed['integral_limits'] = config['integral_limits']

        return processed

    @staticmethod
    def _prepare_for_save(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备配置用于保存（转换numpy数组等）

        Args:
            config: 配置字典

        Returns:
            可序列化的配置字典
        """
        import copy
        config_copy = copy.deepcopy(config)

        # 递归转换numpy数组为列表
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj

        return convert_numpy(config_copy)

    @staticmethod
    def create_config_from_arrays(
        kp: np.ndarray,
        kd: np.ndarray,
        ki: np.ndarray,
        joint_names: Optional[list] = None,
        use_gravity_compensation: bool = True,
        gravity_scale: float = 1.2
    ) -> Dict[str, Any]:
        """
        从numpy数组创建配置字典

        Args:
            kp: 比例增益数组
            kd: 微分增益数组
            ki: 积分增益数组
            joint_names: 关节名称列表（可选）
            use_gravity_compensation: 是否使用重力补偿
            gravity_scale: 重力补偿系数

        Returns:
            配置字典
        """
        n_joints = len(kp)

        if joint_names is None:
            joint_names = [f"Joint{i+1}" for i in range(n_joints)]

        config = {
            'global': {
                'use_gravity_compensation': use_gravity_compensation,
                'gravity_scale': gravity_scale,
                'dt': 0.001
            },
            'joints': [],
            'integral_limits': {
                'min': -1.0,
                'max': 1.0
            }
        }

        for i in range(n_joints):
            config['joints'].append({
                'name': joint_names[i],
                'index': i,
                'kp': float(kp[i]),
                'kd': float(kd[i]),
                'ki': float(ki[i])
            })

        return config


# 便捷函数
def load_pid_config(file_path: str) -> Dict[str, Any]:
    """
    自动识别文件类型并加载PID配置

    Args:
        file_path: 配置文件路径 (.yaml, .yml, 或 .json)

    Returns:
        PID配置字典
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in ['.yaml', '.yml']:
        return PIDConfigLoader.load_from_yaml(str(file_path))
    elif suffix == '.json':
        return PIDConfigLoader.load_from_json(str(file_path))
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .yaml, .yml, or .json")


def save_pid_config(config: Dict[str, Any], file_path: str):
    """
    自动识别文件类型并保存PID配置

    Args:
        config: PID配置字典
        file_path: 保存路径 (.yaml, .yml, 或 .json)
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in ['.yaml', '.yml']:
        PIDConfigLoader.save_to_yaml(config, str(file_path))
    elif suffix == '.json':
        PIDConfigLoader.save_to_json(config, str(file_path))
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .yaml, .yml, or .json")
