# -*- coding: utf-8 -*-
"""
OpenPi 推理服务器

运行环境: Python 3.10+ (openpi 环境)

支持:
1. 从 checkpoint 加载训练好的 Pi0/Pi0.5 模型
2. 单次推理和 chunk 推理
3. 视觉输入 (多相机图像)
4. 语言指令 (prompt)

使用方法:
    # 启动服务器 (等待 Client 配置)
    python -m src.server.inference_server --port 50052

    # 预加载模型启动
    python -m src.server.inference_server --port 50052 \
        --config pi05_astribot_lora \
        --checkpoint checkpoints/pi05_astribot_lora/exp/50000 \
        --prompt "clear up the desktop"
"""

import io
import json
import logging
import os
import signal
import sys
import time
import traceback
from concurrent import futures
from typing import Any, Dict, Iterator, Optional

import numpy as np

import grpc

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 导入生成的 protobuf 代码 (统一管理)
from src.common.proto_imports import pb2, pb2_grpc

# 导入通用模块
from src.common.config import ServerConfig, ActionConfig
from src.common.constants import (
    OPENPI_ACTION_DIM,
    OPENPI_ACTION_DIM_NO_CHASSIS,
    OPENPI_ACTION_DIM_WITH_CHASSIS,
    OPENPI_MODEL_OUTPUT_DIM,
    DEFAULT_ACTION_HORIZON,
    GRPC_MAX_MESSAGE_LENGTH,
)
from src.common.utils import setup_logging

logger = logging.getLogger("openpi_inference.server")


# ============================================================================
# OpenPi 推理相关导入
# ============================================================================
HAS_OPENPI = False

try:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    HAS_OPENPI = True
except ImportError:
    logger.warning("未找到 openpi 模块，请确保 openpi 在 Python 路径中")


class OpenPiModelInference:
    """
    OpenPi 模型推理器
    
    封装 Policy 对象，提供推理接口
    """
    
    def __init__(
        self,
        config_name: str,
        checkpoint_dir: str,
        default_prompt: Optional[str] = None,
        pytorch_device: Optional[str] = None,
    ):
        """
        初始化模型推理器
        
        Args:
            config_name: 训练配置名称 (e.g., "pi05_astribot_lora")
            checkpoint_dir: Checkpoint 目录
            default_prompt: 默认语言指令
            pytorch_device: PyTorch 设备 (e.g., "cuda", "cuda:0")
        """
        if not HAS_OPENPI:
            raise ImportError("需要安装 openpi: pip install -e /path/to/openpi")
        
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir
        self.default_prompt = default_prompt
        self.pytorch_device = pytorch_device
        
        self.policy = None
        self.train_config = None
        self._metadata = {}
        
        self._load()
    
    def _load(self):
        """加载模型"""
        logger.info(f"加载 OpenPi 模型: config={self.config_name}, checkpoint={self.checkpoint_dir}")
        
        start_time = time.time()
        
        # 获取训练配置
        self.train_config = _config.get_config(self.config_name)
        
        # 创建 Policy
        self.policy = _policy_config.create_trained_policy(
            self.train_config,
            self.checkpoint_dir,
            default_prompt=self.default_prompt,
            pytorch_device=self.pytorch_device,
        )
        
        # 获取元数据
        self._metadata = self.policy.metadata
        
        load_time = time.time() - start_time
        logger.info(f"模型加载完成，耗时: {load_time:.2f}s")
        logger.info(f"模型元数据: {self._metadata}")
    
    @property
    def metadata(self) -> dict:
        """获取模型元数据"""
        return self._metadata
    
    @property
    def action_horizon(self) -> int:
        """获取 action horizon"""
        if self.train_config and hasattr(self.train_config, 'model'):
            return getattr(self.train_config.model, 'action_horizon', DEFAULT_ACTION_HORIZON)
        return DEFAULT_ACTION_HORIZON
    
    @property
    def action_dim(self) -> int:
        """获取 action 维度"""
        if self.train_config and hasattr(self.train_config, 'model'):
            return getattr(self.train_config.model, 'action_dim', OPENPI_ACTION_DIM)
        return OPENPI_ACTION_DIM
    
    def reset(self):
        """重置策略状态"""
        # OpenPi Policy 目前没有状态需要重置
        pass
    
    def predict(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行推理
        
        Args:
            observation: 观测字典，包含:
                - "observation.state": 状态向量 (16维)
                - "observation.images.{camera}": 图像数据 (H, W, C), uint8
                - "prompt": 语言指令
                
        Returns:
            包含 "actions" 的字典，shape (action_horizon, action_dim)
        """
        if self.policy is None:
            raise RuntimeError("模型未加载")
        
        return self.policy.infer(observation)


class OpenPiInferenceServicer(pb2_grpc.OpenPiInferenceServiceServicer):
    """
    OpenPi 推理服务 gRPC 实现
    
    设计原则:
    - Server 返回模型原始输出 (25维)，不做过滤
    - Action 过滤 (head/torso/chassis) 由 Client 端负责
    - 这样设计更灵活，Client 可以根据场景动态调整
    """
    
    def __init__(
        self,
        config_name: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        default_prompt: Optional[str] = None,
        pytorch_device: Optional[str] = None,
        action_config: Optional[ActionConfig] = None,
    ):
        """
        初始化服务
        
        Args:
            config_name: 预加载的配置名称 (可选)
            checkpoint_dir: 预加载的 checkpoint 目录 (可选)
            default_prompt: 默认语言指令
            pytorch_device: PyTorch 设备
            action_config: Action 配置 (控制维度过滤)
        """
        self.model_inference: Optional[OpenPiModelInference] = None
        self.is_ready = False
        
        # Action 配置
        self.action_config = action_config or ActionConfig()
        
        # 状态
        self.current_episode = 0
        self.current_frame = 0
        self.current_prompt = default_prompt or ""
        self.model_name = "none"
        
        # 如果提供了配置，立即加载模型
        if config_name and checkpoint_dir:
            self._load_model(config_name, checkpoint_dir, default_prompt, pytorch_device)
        else:
            logger.info("Server 以空闲模式启动，等待 Client 配置...")
    
    def _load_model(
        self,
        config_name: str,
        checkpoint_dir: str,
        default_prompt: Optional[str] = None,
        pytorch_device: Optional[str] = None,
    ):
        """加载模型"""
        logger.info(f"加载模型: config={config_name}, checkpoint={checkpoint_dir}")
        
        self.model_inference = OpenPiModelInference(
            config_name=config_name,
            checkpoint_dir=checkpoint_dir,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )
        
        self.model_name = config_name
        self.current_prompt = default_prompt or ""
        self.is_ready = True
        self.current_frame = 0
        
        logger.info("模型加载成功")
    
    def _build_observation_dict(self, request: "pb2.Observation") -> Dict[str, Any]:
        """
        将 gRPC 请求转换为 OpenPi 观测字典格式
        
        OpenPi 期望的格式:
        - "observation.state": (state_dim,) numpy array, float32
        - "observation.images.{camera_name}": (H, W, C) numpy array, uint8
        - "prompt": string
        """
        obs_dict = {}
        
        # 处理状态 (25维: arm_left(7) + arm_right(7) + grippers(2) + head(2) + torso(4) + chassis(3))
        if request.state:
            state = np.array(list(request.state), dtype=np.float32)
            obs_dict["observation.state"] = state
        
        # 处理图像
        for img_data in request.images:
            if img_data.data:
                try:
                    image = self._decode_image(img_data)
                    if image is not None:
                        key = f"observation.images.{img_data.camera_name}"
                        obs_dict[key] = image
                        logger.debug(f"处理图像 {img_data.camera_name}: shape={image.shape}")
                except Exception as e:
                    logger.warning(f"图像解码失败 ({img_data.camera_name}): {e}")
        
        # 处理 prompt
        if request.prompt:
            obs_dict["prompt"] = request.prompt
        elif self.current_prompt:
            obs_dict["prompt"] = self.current_prompt
        
        return obs_dict
    
    def _decode_image(self, img_data: "pb2.ImageData") -> Optional[np.ndarray]:
        """
        解码图像数据
        
        Returns:
            numpy array: shape (H, W, C), dtype uint8
        """
        from PIL import Image
        
        encoding = img_data.encoding.lower()
        
        if encoding in ['jpeg', 'jpg', 'png']:
            image = Image.open(io.BytesIO(img_data.data))
            image = image.convert('RGB')
        elif encoding == 'raw':
            if img_data.width > 0 and img_data.height > 0:
                image = Image.frombytes('RGB', (img_data.width, img_data.height), img_data.data)
            else:
                logger.warning("raw 格式需要指定 width 和 height")
                return None
        else:
            logger.warning(f"不支持的图像编码格式: {encoding}")
            return None
        
        return np.array(image, dtype=np.uint8)
    
    def Configure(self, request: "pb2.PolicyConfig", context) -> "pb2.ServiceStatus":
        """配置策略"""
        logger.info(f"收到配置请求: config={request.config_name}, checkpoint={request.checkpoint_dir}")
        
        try:
            # 解析 action 配置
            action_config = self._parse_action_config(request)
            if action_config:
                self.action_config = action_config
                logger.info(f"Action 配置: head={action_config.enable_head}, torso={action_config.enable_torso}, chassis={action_config.enable_chassis}")
            
            self._load_model(
                config_name=request.config_name,
                checkpoint_dir=request.checkpoint_dir,
                default_prompt=request.default_prompt or None,
                pytorch_device=request.device or None,
            )
            return self._get_status(f"已加载模型: {request.config_name}")
            
        except Exception as e:
            logger.error(f"配置失败: {e}")
            traceback.print_exc()
            return self._get_status(f"配置失败: {e}")
    
    def _parse_action_config(self, request: "pb2.PolicyConfig") -> Optional[ActionConfig]:
        """
        解析 action 配置 (仅用于日志记录，Server 不再做过滤)
        
        protobuf 字段映射到 ActionConfig:
        - enable_chassis -> execute_chassis
        - enable_head -> execute_head  
        - enable_torso -> execute_torso
        """
        if hasattr(request, 'action_config') and request.HasField('action_config'):
            ac = request.action_config
            return ActionConfig(
                # protobuf enable_* 字段映射到 execute_* 参数
                execute_chassis=ac.enable_chassis if hasattr(ac, 'enable_chassis') else False,
                execute_head=ac.enable_head if hasattr(ac, 'enable_head') else True,
                execute_torso=ac.enable_torso if hasattr(ac, 'enable_torso') else True,
            )
        return None
    
    def Predict(self, request: "pb2.Observation", context) -> "pb2.Action":
        """
        单次推理 - 返回 action chunk 的第一个 action
        
        Server 返回模型原始输出 (25维)，Action 过滤由 Client 端负责。
        """
        if not self.is_ready:
            return pb2.Action(
                status=pb2.NOT_READY,
                error_message="服务未就绪，请先调用 Configure"
            )
        
        try:
            obs_dict = self._build_observation_dict(request)
            
            result = self.model_inference.predict(obs_dict)
            
            # 获取 actions (shape: action_horizon, action_dim)
            actions = result.get("actions", None)
            if actions is None:
                return pb2.Action(
                    status=pb2.ERROR,
                    error_message="模型未返回 actions"
                )
            
            # 返回第一个 action (模型原始输出，不做过滤)
            first_action = actions[0] if actions.ndim > 1 else actions
            
            self.current_frame += 1
            
            logger.debug(f"推理结果: dim={len(first_action)}")
            
            return pb2.Action(
                values=first_action.tolist(),
                is_terminal=False,
                status=pb2.OK,
                server_frame_index=self.current_frame
            )
            
        except Exception as e:
            logger.error(f"推理错误: {e}")
            traceback.print_exc()
            return pb2.Action(
                status=pb2.ERROR,
                error_message=str(e)
            )
    
    def PredictChunk(self, request: "pb2.Observation", context) -> "pb2.ActionChunk":
        """
        Chunk 推理 - 返回完整的 action chunk
        
        Server 返回模型原始输出 (25维)，Action 过滤由 Client 端负责。
        """
        if not self.is_ready:
            return pb2.ActionChunk(
                status=pb2.NOT_READY,
                error_message="服务未就绪，请先调用 Configure"
            )
        
        try:
            obs_dict = self._build_observation_dict(request)
            
            result = self.model_inference.predict(obs_dict)
            
            # 获取 actions (shape: action_horizon, action_dim)
            actions = result.get("actions", None)
            if actions is None:
                return pb2.ActionChunk(
                    status=pb2.ERROR,
                    error_message="模型未返回 actions"
                )
            
            # 确保是 2D
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            
            # 返回模型原始输出，不做过滤
            chunk_size = actions.shape[0]
            action_dim = actions.shape[1]
            
            # 构建响应
            action_steps = []
            for i in range(chunk_size):
                action_steps.append(pb2.ActionStep(values=actions[i].tolist()))
            
            self.current_frame += 1
            
            logger.debug(f"返回 action chunk: size={chunk_size}, dim={action_dim}")
            
            return pb2.ActionChunk(
                actions=action_steps,
                chunk_size=chunk_size,
                action_dim=action_dim,
                is_terminal=False,
                status=pb2.OK,
                server_frame_index=self.current_frame
            )
            
        except Exception as e:
            logger.error(f"Chunk 推理错误: {e}")
            traceback.print_exc()
            return pb2.ActionChunk(
                status=pb2.ERROR,
                error_message=str(e)
            )
    
    def StreamPredict(
        self,
        request_iterator: Iterator["pb2.Observation"],
        context
    ) -> Iterator["pb2.Action"]:
        """流式推理"""
        logger.info("开始流式推理")
        
        for obs in request_iterator:
            if context.is_active():
                action_response = self.Predict(obs, context)
                yield action_response
                
                if action_response.is_terminal:
                    break
            else:
                break
        
        logger.info("流式推理结束")
    
    def Control(self, request: "pb2.ControlCommand", context) -> "pb2.ServiceStatus":
        """控制命令"""
        cmd_type = request.type
        params = dict(request.params)
        
        if cmd_type == pb2.CMD_RESET:
            self._reset_state()
            return self._get_status("已重置")
            
        elif cmd_type == pb2.CMD_SET_EPISODE:
            ep = int(params.get("episode", "0"))
            self.current_episode = ep
            self._reset_state()
            return self._get_status(f"切换到 episode {ep}")
        
        elif cmd_type == pb2.CMD_SET_PROMPT:
            prompt = params.get("prompt", "")
            self.current_prompt = prompt
            return self._get_status(f"设置 prompt: {prompt}")
        
        return self._get_status("未知命令")
    
    def _reset_state(self):
        """重置内部状态"""
        self.current_frame = 0
        if self.model_inference:
            self.model_inference.reset()
    
    def GetStatus(self, request: "pb2.Empty", context) -> "pb2.ServiceStatus":
        """获取状态"""
        return self._get_status()
    
    def Reset(self, request: "pb2.Empty", context) -> "pb2.ServiceStatus":
        """重置"""
        self._reset_state()
        return self._get_status("已重置")
    
    def _get_status(self, message: str = "") -> "pb2.ServiceStatus":
        """构建状态响应"""
        action_horizon = DEFAULT_ACTION_HORIZON
        # 返回模型原始输出维度 (25维)，过滤由 Client 端负责
        action_dim = OPENPI_MODEL_OUTPUT_DIM
        metadata_json = "{}"
        
        if self.model_inference:
            action_horizon = self.model_inference.action_horizon
            action_dim = self.model_inference.action_dim
            metadata_json = json.dumps(self.model_inference.metadata)
        
        return pb2.ServiceStatus(
            is_ready=self.is_ready,
            model_name=self.model_name,
            current_episode=self.current_episode,
            current_frame=self.current_frame,
            action_horizon=action_horizon,
            action_dim=action_dim,
            message=message,
            current_prompt=self.current_prompt,
            metadata_json=metadata_json
        )


class InferenceServer:
    """gRPC 推理服务器"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.server = None
        self.servicer = None
        self._stopped = False
    
    def start(self):
        """启动服务器"""
        if pb2 is None or pb2_grpc is None:
            raise RuntimeError("未找到 protobuf 生成文件，请先运行 scripts/generate_proto.sh")
        
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
            options=[
                ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_LENGTH),
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
            ]
        )
        
        self.servicer = OpenPiInferenceServicer(
            config_name=self.config.config_name,
            checkpoint_dir=self.config.checkpoint_dir,
            default_prompt=self.config.default_prompt,
            pytorch_device=self.config.pytorch_device,
            action_config=self.config.action_config,
        )
        pb2_grpc.add_OpenPiInferenceServiceServicer_to_server(self.servicer, self.server)
        
        address = f'{self.config.host}:{self.config.port}'
        self.server.add_insecure_port(address)
        
        self.server.start()
        logger.info(f"gRPC 服务器已启动: {address}")
        
        if self.config.config_name:
            logger.info(f"预加载模型: {self.config.config_name}")
        else:
            logger.info("等待 Client 配置...")
        
        return self
    
    def wait_for_termination(self, timeout: Optional[float] = None):
        """等待服务器终止"""
        if self.server:
            self.server.wait_for_termination(timeout)
    
    def stop(self, grace: float = 5.0):
        """停止服务器"""
        if self.server and not self._stopped:
            logger.info("正在停止服务器...")
            self.server.stop(grace)
            self._stopped = True
            logger.info("服务器已停止")


def run_server(config: ServerConfig):
    """运行推理服务器"""
    setup_logging("INFO")
    
    server = InferenceServer(config)
    
    def signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在停止...")
        server.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop()


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='OpenPi gRPC 推理服务器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # [推荐] 启动空闲模式服务器，等待 Client 指定模型
  python -m src.server.inference_server --port 50052 --device cuda
  
  # [可选] 预加载模型 (适用于调试)
  python -m src.server.inference_server --port 50052 \\
      --config pi05_astribot_lora \\
      --checkpoint checkpoints/pi05_astribot_lora/exp/50000 \\
      --prompt "clear up the desktop"

说明:
  推荐使用空闲模式启动，让 Client 端指定模型配置。
  这样可以方便地切换不同模型，无需重启 Server。
        """
    )
    
    parser.add_argument('--host', default='0.0.0.0', help='监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=50052, help='监听端口 (默认: 50052)')
    parser.add_argument('--workers', type=int, default=10, help='工作线程数 (默认: 10)')
    
    parser.add_argument('--config', type=str, default=None,
                        help='训练配置名称 (e.g., pi05_astribot_lora)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint 目录')
    parser.add_argument('--prompt', type=str, default=None,
                        help='默认语言指令')
    parser.add_argument('--device', type=str, default=None,
                        help='PyTorch 设备 (e.g., cuda, cuda:0)')
    
    args = parser.parse_args()
    
    config = ServerConfig(
        host=args.host,
        port=args.port,
        max_workers=args.workers,
        config_name=args.config,
        checkpoint_dir=args.checkpoint,
        default_prompt=args.prompt,
        pytorch_device=args.device,
    )
    
    run_server(config)


if __name__ == '__main__':
    main()

