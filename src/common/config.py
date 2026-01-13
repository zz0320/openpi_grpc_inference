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
    OPENPI_STATE_DIM,
    DEFAULT_ACTION_HORIZON,
)


@dataclass
class ActionConfig:
    """
    Action 配置
    
    OpenPi 模型输出 16 维 action，但执行时可能需要扩展到 22/25 维
    """
    # OpenPi 模型输出维度
    model_action_dim: int = OPENPI_ACTION_DIM
    
    # 执行时是否扩展到完整维度 (添加 head, torso 等)
    expand_to_full_dim: bool = True
    
    # 执行时是否控制底盘
    execute_chassis: bool = False
    
    # head/torso 默认值 (扩展时使用)
    default_head: List[float] = field(default_factory=lambda: [0.0, 0.0])
    default_torso: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    
    @property
    def execute_dim(self) -> int:
        """执行时的 action 维度"""
        if not self.expand_to_full_dim:
            return self.model_action_dim
        if self.execute_chassis:
            return 25
        return 22


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

