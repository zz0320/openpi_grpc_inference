# -*- coding: utf-8 -*-
"""
通用工具类
"""

import logging
import sys
from typing import List, Optional
import numpy as np

from .constants import (
    ARM_INDICES,
    GRIPPER_LEFT_INDEX,
    GRIPPER_RIGHT_INDEX,
    OPENPI_ACTION_DIM,
    LEROBOT_ACTION_DIM_NO_CHASSIS,
    LEROBOT_ACTION_DIM_WITH_CHASSIS,
)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    设置日志
    
    Args:
        level: 日志级别
        log_file: 日志文件路径 (可选)
    
    Returns:
        配置好的 logger
    """
    logger = logging.getLogger("openpi_inference")
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的 handlers
    logger.handlers = []
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出 (如果指定)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ActionSmoother:
    """
    动作平滑器 - 使用简单移动平均消除抖动
    """
    
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: 平滑窗口大小
        """
        self.window_size = window_size
        self._history: List[np.ndarray] = []
    
    def smooth(self, action: List[float]) -> List[float]:
        """平滑一个 action"""
        action_arr = np.array(action, dtype=np.float64)
        
        self._history.append(action_arr)
        if len(self._history) > self.window_size:
            self._history.pop(0)
        
        smoothed = np.mean(self._history, axis=0)
        return smoothed.tolist()
    
    def reset(self):
        """重置状态"""
        self._history = []


class VelocityLimiter:
    """
    速度限制器 - 限制相邻帧之间的最大变化量
    
    注意：只对手臂关节限制，不对夹爪限制
    """
    
    def __init__(self, max_delta: float = 0.05):
        """
        Args:
            max_delta: 每帧最大角度变化 (弧度)，默认 0.05 rad ≈ 2.9°
        """
        self.max_delta = max_delta
        self._last_action: Optional[np.ndarray] = None
    
    def limit(self, action: List[float]) -> List[float]:
        """限制速度 (只限制手臂，不限制夹爪)"""
        action_arr = np.array(action, dtype=np.float64)
        
        if self._last_action is None:
            self._last_action = action_arr.copy()
            return action_arr.tolist()
        
        limited = action_arr.copy()
        
        # 只对手臂关节进行速度限制 (前14维)
        arm_indices = list(range(min(14, len(action_arr))))
        for i in arm_indices:
            if i < len(action_arr) and i < len(self._last_action):
                delta = action_arr[i] - self._last_action[i]
                delta = np.clip(delta, -self.max_delta, self.max_delta)
                limited[i] = self._last_action[i] + delta
        
        self._last_action = limited.copy()
        return limited.tolist()
    
    def reset(self):
        """重置状态"""
        self._last_action = None


def binarize_gripper_action(
    action: List[float],
    threshold: float = 0.5,
    high_value: float = 1.0,
    low_value: float = 0.0,
    gripper_indices: List[int] = None
) -> List[float]:
    """
    将夹爪控制二值化 (0/1 控制)
    
    当夹爪值超过阈值时设为 high_value，否则设为 low_value
    
    Args:
        action: action 数组
        threshold: 阈值
        high_value: 超过阈值时的值
        low_value: 低于阈值时的值
        gripper_indices: 夹爪索引列表，默认 [14, 15]
        
    Returns:
        处理后的 action 数组
    """
    if gripper_indices is None:
        gripper_indices = [GRIPPER_LEFT_INDEX, GRIPPER_RIGHT_INDEX]
    
    action = list(action)
    
    for idx in gripper_indices:
        if idx < len(action):
            action[idx] = high_value if action[idx] > threshold else low_value
    
    return action


def expand_openpi_action_to_lerobot(
    action: List[float],
    default_head: List[float] = None,
    default_torso: List[float] = None,
    default_chassis: List[float] = None,
    include_chassis: bool = False
) -> List[float]:
    """
    将 OpenPi 16维 action 扩展为 LeRobot V2.0 格式
    
    OpenPi 格式 (16维):
        [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1)]
    
    LeRobot V2.0 格式 (22/25维):
        [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)?]
    
    Args:
        action: OpenPi action (16维)
        default_head: 默认 head 值 (2维)
        default_torso: 默认 torso 值 (4维)
        default_chassis: 默认 chassis 值 (3维)
        include_chassis: 是否包含底盘
        
    Returns:
        LeRobot V2.0 格式 action (22或25维)
    """
    if len(action) != OPENPI_ACTION_DIM:
        raise ValueError(f"OpenPi action 必须是 {OPENPI_ACTION_DIM} 维，当前: {len(action)}")
    
    default_head = default_head or [0.0, 0.0]
    default_torso = default_torso or [0.0, 0.0, 0.0, 0.0]
    default_chassis = default_chassis or [0.0, 0.0, 0.0]
    
    # 扩展: 添加 head 和 torso
    expanded = list(action) + default_head + default_torso
    
    if include_chassis:
        expanded = expanded + default_chassis
    
    return expanded


def contract_lerobot_state_to_openpi(state: List[float]) -> List[float]:
    """
    将 LeRobot V2.0 格式 state 压缩为 OpenPi 格式
    
    LeRobot V2.0 格式 (22/25维):
        [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)?]
    
    OpenPi 格式 (16维):
        [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1)]
    
    Args:
        state: LeRobot state (22或25维)
        
    Returns:
        OpenPi 格式 state (16维)
    """
    # 只取前16维
    return list(state[:OPENPI_ACTION_DIM])


def openpi_action_to_waypoint(action: List[float], include_chassis: bool = False) -> List[List[float]]:
    """
    将 OpenPi action 转换为 Astribot waypoint 格式
    
    首先扩展为 LeRobot 格式，然后转换为 waypoint
    
    Args:
        action: OpenPi action (16维)
        include_chassis: 是否包含底盘
        
    Returns:
        waypoint: [torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2), chassis(3)?]
    """
    # 先扩展为 LeRobot 格式
    expanded = expand_openpi_action_to_lerobot(
        action, 
        include_chassis=include_chassis
    )
    
    # 转换为 waypoint
    return lerobot_action_to_waypoint(expanded, include_chassis=include_chassis)


def lerobot_action_to_waypoint(action: List[float], include_chassis: bool = False) -> List[List[float]]:
    """
    将 LeRobot V2.0 格式的 action 转换为 Astribot waypoint 格式
    
    Args:
        action: LeRobot action 数组 (22或25维)
        include_chassis: 是否包含底盘控制
        
    Returns:
        waypoint: [torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2), chassis(3)?]
    """
    if isinstance(action, np.ndarray):
        action = action.tolist()
    
    action_len = len(action)
    
    if action_len < LEROBOT_ACTION_DIM_NO_CHASSIS:
        raise ValueError(f"Action 长度必须至少为 {LEROBOT_ACTION_DIM_NO_CHASSIS}，当前为 {action_len}")
    
    # 构建 waypoint
    waypoint = [
        action[18:22],         # torso (4)
        action[0:7],           # arm_left (7)
        [action[14]],          # gripper_left (1)
        action[7:14],          # arm_right (7)
        [action[15]],          # gripper_right (1)
        action[16:18],         # head (2)
    ]
    
    if include_chassis and action_len >= LEROBOT_ACTION_DIM_WITH_CHASSIS:
        waypoint.append(action[22:25])  # chassis (3)
    
    return waypoint


def waypoint_to_openpi_state(waypoint: List[List[float]]) -> List[float]:
    """
    将 Astribot waypoint 格式转换为 OpenPi state 格式
    
    Args:
        waypoint: [torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2), chassis(3)?]
        
    Returns:
        state: OpenPi state (16维)
    """
    # torso = waypoint[0]         # 4 - 不使用
    arm_left = waypoint[1]      # 7
    gripper_left = waypoint[2]  # 1
    arm_right = waypoint[3]     # 7
    gripper_right = waypoint[4] # 1
    # head = waypoint[5]          # 2 - 不使用
    
    # OpenPi 格式: [arm_left, arm_right, gripper_left, gripper_right]
    state = arm_left + arm_right + gripper_left + gripper_right
    
    return state

