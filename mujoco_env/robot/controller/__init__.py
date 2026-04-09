"""
机械臂控制器模块
"""

from .gravity_compensation import GravityCompensationController
from .torque_controller import TorqueController

__all__ = ['GravityCompensationController', 'TorqueController']
