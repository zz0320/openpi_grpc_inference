# -*- coding: utf-8 -*-
"""
通用工具类

支持 25 维 (含底盘) 和 22 维 (不含底盘) action 格式转换
"""

import logging
import sys
from typing import List, Optional
import numpy as np

from .constants import (
    ARM_INDICES,
    GRIPPER_LEFT_INDEX,
    GRIPPER_RIGHT_INDEX,
    HEAD_INDICES,
    TORSO_INDICES,
    CHASSIS_INDICES,
    OPENPI_ACTION_DIM_NO_CHASSIS,
    OPENPI_ACTION_DIM_WITH_CHASSIS,
    OPENPI_ACTION_INDEX,
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


def filter_action_by_config(
    action: List[float],
    execute_head: bool = True,
    execute_torso: bool = True,
    execute_chassis: bool = False,
    current_state: Optional[List[float]] = None
) -> List[float]:
    """
    根据配置过滤 action
    
    将不执行的部件设为当前值（保持不变）或零值
    
    Args:
        action: 原始 action (25维)
        execute_head: 是否执行头部
        execute_torso: 是否执行腰部
        execute_chassis: 是否执行底盘
        current_state: 当前状态 (用于保持不执行部件的值)
        
    Returns:
        过滤后的 action (22维或25维)
    """
    if len(action) < OPENPI_ACTION_DIM_NO_CHASSIS:
        return action
    
    action = list(action)
    
    # 如果有当前状态，不执行的部件保持当前值
    if current_state is not None:
        if not execute_head and len(current_state) > 16:
            action[16:18] = current_state[16:18]
        if not execute_torso and len(current_state) > 18:
            action[18:22] = current_state[18:22]
    
    # 是否包含底盘
    if execute_chassis:
        # 返回完整 25 维
        if len(action) >= OPENPI_ACTION_DIM_WITH_CHASSIS:
            return action[:25]
        else:
            # 如果 action 不足 25 维，补零
            return action + [0.0] * (25 - len(action))
    else:
        # 返回 22 维 (去掉底盘)
        return action[:22]


def openpi_action_to_waypoint(
    action: List[float], 
    include_chassis: bool = False
) -> List[List[float]]:
    """
    将 OpenPi action (25维) 转换为 Astribot waypoint 格式
    
    OpenPi 格式 (25维):
        [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)]
    
    Astribot waypoint 格式:
        [torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2), chassis(3)?]
    
    Args:
        action: OpenPi action (22或25维)
        include_chassis: 是否包含底盘控制
        
    Returns:
        waypoint: 嵌套列表格式
    """
    if isinstance(action, np.ndarray):
        action = action.tolist()
    
    action_len = len(action)
    
    if action_len < OPENPI_ACTION_DIM_NO_CHASSIS:
        raise ValueError(f"Action 长度必须至少为 {OPENPI_ACTION_DIM_NO_CHASSIS}，当前为 {action_len}")
    
    # 构建 waypoint
    waypoint = [
        action[18:22],         # torso (4)
        action[0:7],           # arm_left (7)
        [action[14]],          # gripper_left (1)
        action[7:14],          # arm_right (7)
        [action[15]],          # gripper_right (1)
        action[16:18],         # head (2)
    ]
    
    if include_chassis and action_len >= OPENPI_ACTION_DIM_WITH_CHASSIS:
        waypoint.append(action[22:25])  # chassis (3)
    
    return waypoint


def waypoint_to_openpi_state(
    waypoint: List[List[float]], 
    include_chassis: bool = False
) -> List[float]:
    """
    将 Astribot waypoint 格式转换为 OpenPi state 格式
    
    Astribot waypoint 格式:
        [torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2), chassis(3)?]
    
    OpenPi 格式 (25维):
        [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)]
    
    Args:
        waypoint: Astribot waypoint 格式
        include_chassis: 是否包含底盘
        
    Returns:
        state: OpenPi state (22或25维)
    """
    torso = waypoint[0]         # 4
    arm_left = waypoint[1]      # 7
    gripper_left = waypoint[2]  # 1
    arm_right = waypoint[3]     # 7
    gripper_right = waypoint[4] # 1
    head = waypoint[5]          # 2
    
    # 按 OpenPi 顺序组装
    state = (
        list(arm_left) +        # [0:7]
        list(arm_right) +       # [7:14]
        list(gripper_left) +    # [14:15]
        list(gripper_right) +   # [15:16]
        list(head) +            # [16:18]
        list(torso)             # [18:22]
    )
    
    if include_chassis and len(waypoint) > 6:
        chassis = waypoint[6]   # 3
        state = state + list(chassis)  # [22:25]
    
    return state


def get_ready_position(include_chassis: bool = False) -> List[float]:
    """
    获取机器人准备位置
    
    Args:
        include_chassis: 是否包含底盘
        
    Returns:
        准备位置 (22或25维)
    """
    from .constants import READY_POSITION_22, READY_POSITION_25
    
    if include_chassis:
        return list(READY_POSITION_25)
    return list(READY_POSITION_22)
