# -*- coding: utf-8 -*-
"""
配置管理
"""

from dataclasses import dataclass, field
from typing import Optional, List

from .constants import (
    DEFAULT_CONTROL_FREQ,
    DEFAULT_GRPC_PORT,
    DEFAULT_GRPC_HOST,
    OPENPI_ACTION_DIM,
    OPENPI_ACTION_DIM_NO_CHASSIS,
    OPENPI_ACTION_DIM_WITH_CHASSIS,
    OPENPI_MODEL_OUTPUT_DIM,
    DEFAULT_ACTION_HORIZON,
)


@dataclass
class ActionConfig:
    """
    Action 配置
    
    OpenPi 模型输出 25 维 action，实际使用时可以过滤为 22 维（不含底盘）
    
    分离三个概念:
    1. state_dim: 输入 state 的维度 (22 或 25)，由机器人状态采集决定
    2. action_dim: 模型输出的 action 维度 (25)，由训练时决定
    3. execute_chassis: 执行时是否控制底盘 (只影响发送给机器人的命令)
    """
    # ========== 输入配置 ==========
    # 输入 state 是否包含底盘 (影响 Client 采集的状态维度)
    state_includes_chassis: bool = False
    
    # ========== 执行配置 ==========
    # 执行 action 时是否控制底盘 (即使 action 有 25 维，也可以选择不执行底盘)
    execute_chassis: bool = False
    # 是否执行头部控制
    execute_head: bool = True
    # 是否执行腰部控制
    execute_torso: bool = True
    
    @property
    def state_dim(self) -> int:
        """输入 state 维度"""
        # 基础: arm_left(7) + arm_right(7) + gripper_left(1) + gripper_right(1) + head(2) + torso(4) = 22
        return OPENPI_ACTION_DIM_WITH_CHASSIS if self.state_includes_chassis else OPENPI_ACTION_DIM_NO_CHASSIS
    
    @property
    def output_dim(self) -> int:
        """输出 action 维度 (过滤后)"""
        return OPENPI_ACTION_DIM_WITH_CHASSIS if self.execute_chassis else OPENPI_ACTION_DIM_NO_CHASSIS
    
    # ========== 兼容旧接口 ==========
    @property
    def enable_chassis(self) -> bool:
        """兼容旧接口"""
        return self.execute_chassis
    
    @property
    def enable_head(self) -> bool:
        """兼容旧接口"""
        return self.execute_head
    
    @property
    def enable_torso(self) -> bool:
        """兼容旧接口"""
        return self.execute_torso


@dataclass
class ServerConfig:
    """
    Server 配置
    """
    host: str = DEFAULT_GRPC_HOST
    port: int = DEFAULT_GRPC_PORT
    max_workers: int = 10
    
    # 预加载模型配置 (可选)
    config_name: Optional[str] = None  # e.g., "pi05_astribot_lora"
    checkpoint_dir: Optional[str] = None  # e.g., "checkpoints/pi05_astribot_lora/exp/50000"
    
    # 默认 prompt
    default_prompt: Optional[str] = None
    
    # PyTorch 设备
    pytorch_device: Optional[str] = None  # e.g., "cuda", "cuda:0"
    
    # Action 配置 (默认值，Client 可覆盖)
    action_config: ActionConfig = field(default_factory=ActionConfig)


@dataclass
class ClientConfig:
    """Client 配置"""
    server_host: str = "localhost"
    server_port: int = DEFAULT_GRPC_PORT
    timeout: float = 30.0
    
    # 策略配置 (Client 端指定，Server 端加载)
    config_name: Optional[str] = None  # OpenPi 训练配置名称
    checkpoint_dir: Optional[str] = None  # Checkpoint 目录
    default_prompt: Optional[str] = None  # 默认语言指令
    pytorch_device: Optional[str] = None  # 推理设备
    
    # 控制配置
    control_freq: float = DEFAULT_CONTROL_FREQ
    control_way: str = "direct"  # "direct" or "filter"
    
    # 平滑配置
    smooth_window: int = 0  # 0 = 不平滑
    max_velocity: float = 0.0  # 0 = 不限制
    
    # Action 配置
    action_config: ActionConfig = field(default_factory=ActionConfig)
    
    @property
    def server_address(self) -> str:
        return f"{self.server_host}:{self.server_port}"

