# -*- coding: utf-8 -*-
"""
OpenPi 推理客户端

运行环境: 机器人侧 (Astribot SDK 环境)

通过 gRPC 连接远程推理服务器获取 action
支持:
1. 单步推理模式
2. Chunk 推理模式 (Action Chunking)
3. 语言指令 (prompt) 控制
4. Astribot S1 机器人控制
5. 完整的推理日志记录
"""

import os
import sys
import time
import signal
import logging
from collections import deque
from typing import List, Optional, Iterator, Callable, Dict
import numpy as np

import grpc

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 导入生成的 protobuf 代码
try:
    from src.generated import openpi_inference_pb2 as pb2
    from src.generated import openpi_inference_pb2_grpc as pb2_grpc
except ImportError:
    try:
        from generated import openpi_inference_pb2 as pb2
        from generated import openpi_inference_pb2_grpc as pb2_grpc
    except ImportError:
        pb2 = None
        pb2_grpc = None
        print("警告: 未找到 protobuf 生成文件，请先运行 scripts/generate_proto.sh")

# 导入通用模块
from src.common.config import ClientConfig, ActionConfig
from src.common.constants import (
    OPENPI_ACTION_DIM,
    OPENPI_STATE_DIM,
    DEFAULT_ACTION_HORIZON,
    ASTRIBOT_NAMES_LIST,
    ASTRIBOT_NAMES_LIST_WITH_CHASSIS,
    ASTRIBOT_IMAGE_TOPICS,
    ASTRIBOT_IMAGE_SHAPES,
    GRPC_MAX_MESSAGE_LENGTH,
    READY_POSITION_16,
    READY_POSITION_22,
)
from src.common.utils import (
    setup_logging,
    ActionSmoother,
    VelocityLimiter,
    binarize_gripper_action,
    expand_openpi_action_to_lerobot,
    contract_lerobot_state_to_openpi,
    lerobot_action_to_waypoint,
    waypoint_to_openpi_state,
)

# 导入日志记录器
from src.client.inference_logger import InferenceLogger

logger = logging.getLogger("openpi_inference.client")


# ============================================================================
# Astribot SDK 导入
# ============================================================================
HAS_ASTRIBOT = False
try:
    from core.astribot_api.astribot_client import Astribot
    HAS_ASTRIBOT = True
except ImportError:
    pass

# ROS 相机
HAS_ROS = False
try:
    import rospy
    from sensor_msgs.msg import CompressedImage
    HAS_ROS = True
except ImportError:
    pass


# 全局中断标志
_interrupted = False

def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    logger.warning(f"收到中断信号 {signum}")

signal.signal(signal.SIGINT, _signal_handler)


class AstribotCameraSubscriber:
    """
    Astribot ROS 图像话题订阅器
    
    从 ROS 话题获取压缩图像数据
    """
    
    def __init__(self, camera_names: List[str] = None):
        """
        Args:
            camera_names: 要订阅的相机名称列表，默认 ['head', 'wrist_left', 'wrist_right']
        """
        if not HAS_ROS:
            raise ImportError("需要安装 ROS: rospy, sensor_msgs")
        
        self.camera_names = camera_names or ['head', 'wrist_left', 'wrist_right']
        self._images: Dict[str, bytes] = {}
        self._subscribers = []
        self._initialized = False
    
    def start(self, init_node: bool = True):
        """启动订阅"""
        if init_node:
            try:
                rospy.init_node('openpi_camera_subscriber', anonymous=True)
            except rospy.exceptions.ROSException:
                pass
        
        for topic, cam_name in ASTRIBOT_IMAGE_TOPICS.items():
            if cam_name in self.camera_names:
                sub = rospy.Subscriber(
                    topic,
                    CompressedImage,
                    self._callback,
                    callback_args=cam_name,
                    queue_size=1
                )
                self._subscribers.append(sub)
                logger.info(f"订阅相机: {cam_name} <- {topic}")
        
        self._initialized = True
    
    def _callback(self, msg: "CompressedImage", cam_name: str):
        """ROS 回调函数"""
        self._images[cam_name] = bytes(msg.data)
    
    def get_image(self, camera_name: str) -> Optional[bytes]:
        """获取指定相机的图像 (JPEG bytes)"""
        return self._images.get(camera_name)
    
    def get_all_images(self) -> Dict[str, bytes]:
        """获取所有相机的图像"""
        return {k: v for k, v in self._images.items() if v is not None}
    
    def get_images_for_inference(self) -> List[dict]:
        """获取用于推理的编码图像列表"""
        images = []
        for cam_name, img_bytes in self._images.items():
            if img_bytes:
                images.append({
                    'name': cam_name,
                    'data': img_bytes,
                    'width': ASTRIBOT_IMAGE_SHAPES[cam_name][1],
                    'height': ASTRIBOT_IMAGE_SHAPES[cam_name][0],
                    'encoding': 'jpeg'
                })
        return images
    
    def wait_for_images(self, timeout: float = 5.0) -> bool:
        """等待所有相机图像就绪"""
        start = time.time()
        while time.time() - start < timeout:
            if all(cam in self._images for cam in self.camera_names):
                return True
            time.sleep(0.1)
        return False
    
    def stop(self):
        """停止订阅"""
        for sub in self._subscribers:
            sub.unregister()
        self._subscribers = []
        self._images = {}
        self._initialized = False


class ActionChunkManager:
    """
    Action Chunk 管理器
    
    在 Client 端管理 action queue，实现：
    1. 从 Server 获取完整的 action chunk
    2. 在本地逐步消费 action
    3. 当 queue 用完时，自动请求新的 chunk
    """
    
    def __init__(
        self,
        client: "InferenceClient",
        n_action_steps: Optional[int] = None,
        auto_refill_threshold: float = 0.0
    ):
        """
        初始化 Action Chunk 管理器
        
        Args:
            client: InferenceClient 实例
            n_action_steps: 每个 chunk 实际使用的 action 数量
            auto_refill_threshold: 自动补充阈值 (0.0-1.0)
        """
        self.client = client
        self.n_action_steps = n_action_steps
        self.auto_refill_threshold = auto_refill_threshold
        
        self._action_queue: deque = deque()
        self._chunk_size = 0
        self._action_dim = 0
        
        self._current_chunk_start_frame = 0
        self._actions_consumed = 0
        self._total_actions_consumed = 0
        self._is_terminal = False
        self._last_action_triggered_inference = False
    
    @property
    def queue_size(self) -> int:
        return len(self._action_queue)
    
    @property
    def is_empty(self) -> bool:
        return len(self._action_queue) == 0
    
    @property
    def is_terminal(self) -> bool:
        return self._is_terminal and self.is_empty
    
    @property
    def last_action_triggered_inference(self) -> bool:
        return self._last_action_triggered_inference
    
    def reset(self):
        """重置状态"""
        self._action_queue.clear()
        self._actions_consumed = 0
        self._total_actions_consumed = 0
        self._is_terminal = False
        self._current_chunk_start_frame = 0
        self._last_action_triggered_inference = False
    
    def _should_refill(self) -> bool:
        if self._is_terminal:
            return False
        if self._chunk_size == 0:
            return True
        
        effective_size = self.n_action_steps or self._chunk_size
        remaining_ratio = len(self._action_queue) / effective_size
        return remaining_ratio <= self.auto_refill_threshold
    
    def _fetch_chunk(
        self,
        state: List[float],
        episode_id: int,
        frame_index: int,
        images: Optional[List[dict]] = None,
        prompt: Optional[str] = None
    ) -> bool:
        """从 Server 获取新的 action chunk"""
        try:
            chunk_response = self.client.predict_chunk(
                state=state,
                episode_id=episode_id,
                frame_index=frame_index,
                images=images,
                prompt=prompt
            )
            
            if chunk_response.status == pb2.EPISODE_END:
                self._is_terminal = True
                return False
            
            if chunk_response.status != pb2.OK:
                logger.error(f"获取 chunk 失败: {chunk_response.error_message}")
                return False
            
            self._chunk_size = chunk_response.chunk_size
            self._action_dim = chunk_response.action_dim
            self._current_chunk_start_frame = frame_index
            
            self._action_queue.clear()
            self._actions_consumed = 0
            
            n_to_use = self.n_action_steps if self.n_action_steps else self._chunk_size
            n_to_use = min(n_to_use, len(chunk_response.actions))
            
            for i in range(n_to_use):
                action = list(chunk_response.actions[i].values)
                self._action_queue.append(action)
            
            logger.debug(f"获取到 chunk: size={self._chunk_size}, 使用={n_to_use}")
            return True
            
        except Exception as e:
            logger.error(f"获取 chunk 异常: {e}")
            return False
    
    def get_action(
        self,
        state: List[float],
        episode_id: int = 0,
        frame_index: int = 0,
        images: Optional[List[dict]] = None,
        prompt: Optional[str] = None
    ) -> Optional[List[float]]:
        """获取下一个 action"""
        self._last_action_triggered_inference = False
        
        if self.is_empty or self._should_refill():
            chunk_frame = self._current_chunk_start_frame + self._actions_consumed
            if self.is_empty:
                chunk_frame = frame_index
            
            if self._fetch_chunk(state, episode_id, chunk_frame, images, prompt):
                self._last_action_triggered_inference = True
            else:
                if self._is_terminal and not self.is_empty:
                    pass
                elif self.is_empty:
                    return None
        
        if self.is_empty:
            return None
        
        action = self._action_queue.popleft()
        self._actions_consumed += 1
        self._total_actions_consumed += 1
        
        return action


class InferenceClient:
    """
    gRPC 推理客户端
    
    负责与远程推理服务器通信
    """
    
    def __init__(
        self, 
        server_address: str = "localhost:50052", 
        timeout: float = 30.0
    ):
        self.server_address = server_address
        self.timeout = timeout
        self.channel = None
        self.stub = None
        self._connected = False
        
        self._connect()
    
    def _connect(self):
        """连接到服务器"""
        if pb2 is None or pb2_grpc is None:
            raise RuntimeError("未找到 protobuf 生成文件")
        
        logger.info(f"连接推理服务器: {self.server_address}")
        
        self.channel = grpc.insecure_channel(
            self.server_address,
            options=[
                ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_LENGTH),
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
            ]
        )
        self.stub = pb2_grpc.OpenPiInferenceServiceStub(self.channel)
        
        try:
            grpc.channel_ready_future(self.channel).result(timeout=self.timeout)
            self._connected = True
            logger.info(f"已连接到推理服务器")
        except grpc.FutureTimeoutError:
            raise ConnectionError(f"无法连接到服务器: {self.server_address}")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def get_status(self) -> "pb2.ServiceStatus":
        """获取服务状态"""
        return self.stub.GetStatus(pb2.Empty())
    
    def configure(
        self,
        config_name: str,
        checkpoint_dir: str,
        default_prompt: str = "",
        device: str = ""
    ) -> "pb2.ServiceStatus":
        """配置 Server 使用的模型"""
        config = pb2.PolicyConfig(
            config_name=config_name,
            checkpoint_dir=checkpoint_dir,
            default_prompt=default_prompt,
            device=device
        )
        return self.stub.Configure(config)
    
    @staticmethod
    def encode_image(image, camera_name: str = "cam", encoding: str = "jpeg", quality: int = 85) -> dict:
        """将图像编码为可发送的格式"""
        import io
        from PIL import Image as PILImage
        
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = PILImage.fromarray(image)
        elif isinstance(image, bytes):
            return {
                'name': camera_name,
                'data': image,
                'width': 0,
                'height': 0,
                'encoding': encoding
            }
        elif hasattr(image, 'mode'):
            pil_image = image
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
        
        width, height = pil_image.size
        
        if encoding.lower() in ['jpeg', 'jpg']:
            buffer = io.BytesIO()
            pil_image.convert('RGB').save(buffer, format='JPEG', quality=quality)
            data = buffer.getvalue()
        elif encoding.lower() == 'png':
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            data = buffer.getvalue()
        else:
            data = pil_image.convert('RGB').tobytes()
        
        return {
            'name': camera_name,
            'data': data,
            'width': width,
            'height': height,
            'encoding': encoding
        }
    
    def predict(
        self,
        state: List[float],
        episode_id: int = 0,
        frame_index: int = 0,
        images: Optional[List[dict]] = None,
        prompt: Optional[str] = None,
        extra_state: str = ""
    ) -> "pb2.Action":
        """单次推理"""
        obs = pb2.Observation(
            state=state,
            timestamp=time.time(),
            episode_id=episode_id,
            frame_index=frame_index,
            prompt=prompt or "",
            extra_state=extra_state
        )
        
        if images:
            for img in images:
                obs.images.append(pb2.ImageData(
                    camera_name=img.get('name', 'cam'),
                    data=img.get('data', b''),
                    width=img.get('width', 0),
                    height=img.get('height', 0),
                    encoding=img.get('encoding', 'jpeg')
                ))
        
        return self.stub.Predict(obs)
    
    def predict_chunk(
        self,
        state: List[float],
        episode_id: int = 0,
        frame_index: int = 0,
        images: Optional[List[dict]] = None,
        prompt: Optional[str] = None,
        extra_state: str = ""
    ) -> "pb2.ActionChunk":
        """Chunk 推理 - 一次性获取完整的 action chunk"""
        obs = pb2.Observation(
            state=state,
            timestamp=time.time(),
            episode_id=episode_id,
            frame_index=frame_index,
            prompt=prompt or "",
            extra_state=extra_state
        )
        
        if images:
            for img in images:
                obs.images.append(pb2.ImageData(
                    camera_name=img.get('name', 'cam'),
                    data=img.get('data', b''),
                    width=img.get('width', 0),
                    height=img.get('height', 0),
                    encoding=img.get('encoding', 'jpeg')
                ))
        
        return self.stub.PredictChunk(obs)
    
    def set_prompt(self, prompt: str) -> "pb2.ServiceStatus":
        """设置语言指令"""
        cmd = pb2.ControlCommand(
            type=pb2.CMD_SET_PROMPT,
            params={"prompt": prompt}
        )
        return self.stub.Control(cmd)
    
    def reset(self) -> "pb2.ServiceStatus":
        """重置推理状态"""
        return self.stub.Reset(pb2.Empty())
    
    def set_episode(self, episode: int) -> "pb2.ServiceStatus":
        """设置当前 episode"""
        cmd = pb2.ControlCommand(
            type=pb2.CMD_SET_EPISODE,
            params={"episode": str(episode)}
        )
        return self.stub.Control(cmd)
    
    def close(self):
        """关闭连接"""
        if self.channel:
            self.channel.close()
            self._connected = False
        logger.info("已断开推理服务器连接")


class AstribotController:
    """
    Astribot 机器人控制器
    
    整合 gRPC 客户端、机器人 SDK 和相机订阅
    
    支持两种推理模式:
    1. 单步模式 (use_chunk=False): 每次请求获取一个 action
    2. Chunk 模式 (use_chunk=True): 一次获取完整 action chunk，本地消费
    """
    
    def __init__(
        self, 
        config: ClientConfig, 
        enable_camera: bool = False, 
        camera_names: List[str] = None,
        use_chunk: bool = False,
        n_action_steps: Optional[int] = None,
        inference_logger: Optional[InferenceLogger] = None,
        binarize_gripper: bool = False,
        gripper_threshold: float = 0.5
    ):
        """
        初始化控制器
        
        Args:
            config: 客户端配置
            enable_camera: 是否启用相机订阅
            camera_names: 要订阅的相机名称列表
            use_chunk: 是否使用 chunk 模式
            n_action_steps: chunk 模式下每个 chunk 使用的 action 数量
            inference_logger: 推理日志记录器
            binarize_gripper: 是否启用夹爪二值化控制
            gripper_threshold: 夹爪二值化阈值
        """
        self.config = config
        self._use_chunk = use_chunk
        self._n_action_steps = n_action_steps
        self.inference_logger = inference_logger
        self._binarize_gripper = binarize_gripper
        self._gripper_threshold = gripper_threshold
        
        logger.info(f"初始化 AstribotController")
        logger.info(f"  - 服务器: {config.server_address}")
        logger.info(f"  - 控制频率: {config.control_freq} Hz")
        logger.info(f"  - 推理模式: {'Chunk' if use_chunk else '单步'}")
        
        # 当前 prompt
        self._current_prompt = config.default_prompt or ""
        
        # 初始化推理客户端
        self.inference_client = InferenceClient(
            server_address=config.server_address,
            timeout=config.timeout
        )
        
        # 配置 Server
        self._configure_server()
        
        # 控制参数
        self.control_freq = config.control_freq
        self.control_period = 1.0 / config.control_freq
        self.control_way = config.control_way
        
        # 平滑器和速度限制器
        self.smoother = None
        self.velocity_limiter = None
        
        if config.smooth_window > 0:
            self.smoother = ActionSmoother(window_size=config.smooth_window)
        
        if config.max_velocity > 0:
            self.velocity_limiter = VelocityLimiter(max_delta=config.max_velocity)
        
        # 初始化 Astribot SDK
        self.astribot = None
        if HAS_ASTRIBOT:
            self.astribot = Astribot(freq=config.control_freq)
            logger.info("Astribot SDK 已初始化")
        else:
            logger.warning("Astribot SDK 不可用，将以模拟模式运行")
        
        # 初始化相机订阅器
        self.camera_subscriber = None
        self._enable_camera = enable_camera
        if enable_camera:
            if HAS_ROS:
                cam_names = camera_names or ['head', 'wrist_left', 'wrist_right']
                self.camera_subscriber = AstribotCameraSubscriber(cam_names)
                self.camera_subscriber.start(init_node=True)
            else:
                logger.warning("ROS 不可用，无法启用相机订阅")
                self._enable_camera = False
        
        # 状态
        self._current_waypoint = None
        self._episode_id = 0
        self._frame_index = 0
        self._use_wbc = False
        
        # Action 配置
        self._action_config = config.action_config
        
        # Chunk 模式管理器
        self._chunk_manager: Optional[ActionChunkManager] = None
        if self._use_chunk:
            self._chunk_manager = ActionChunkManager(
                client=self.inference_client,
                n_action_steps=self._n_action_steps,
                auto_refill_threshold=0.0
            )
    
    def _configure_server(self):
        """配置 Server 端"""
        if self.config.config_name and self.config.checkpoint_dir:
            logger.info(f"配置 Server: config={self.config.config_name}")
            status = self.inference_client.configure(
                config_name=self.config.config_name,
                checkpoint_dir=self.config.checkpoint_dir,
                default_prompt=self.config.default_prompt or "",
                device=self.config.pytorch_device or ""
            )
            logger.info(f"Server 配置结果: {status.message}")
        else:
            status = self.inference_client.get_status()
            if status.is_ready:
                logger.info(f"Server 已就绪: {status.model_name}")
            else:
                logger.warning("Server 未就绪，请确保已配置模型")
    
    def set_prompt(self, prompt: str):
        """设置语言指令"""
        self._current_prompt = prompt
        self.inference_client.set_prompt(prompt)
        logger.info(f"设置 prompt: {prompt}")
    
    def get_current_state(self) -> List[float]:
        """
        获取当前状态 (OpenPi 格式: 16维)
        
        从机器人本体读取真实关节反馈
        """
        if self.astribot is not None:
            try:
                names_list = [
                    'astribot_arm_left',
                    'astribot_gripper_left',
                    'astribot_arm_right',
                    'astribot_gripper_right',
                ]
                
                positions = self.astribot.get_current_joints_position(names_list)
                
                # positions: [[arm_left(7)], [gripper_left(1)], [arm_right(7)], [gripper_right(1)]]
                arm_left = positions[0]
                gripper_left = positions[1]
                arm_right = positions[2]
                gripper_right = positions[3]
                
                # OpenPi 格式: [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1)]
                state = arm_left + arm_right + gripper_left + gripper_right
                return state
                
            except Exception as e:
                logger.warning(f"从机器人读取状态失败: {e}")
        
        # 回退: 使用追踪的命令位置
        if self._current_waypoint:
            return waypoint_to_openpi_state(self._current_waypoint)
        
        return [0.0] * OPENPI_STATE_DIM
    
    def move_to_ready_position(self, duration: float = 5.0) -> bool:
        """移动到准备位置"""
        logger.info("=" * 60)
        logger.info("移动到准备位置")
        logger.info("=" * 60)
        
        # 使用 OpenPi 16维准备位置，扩展到 22 维执行
        ready_action_16 = READY_POSITION_16
        ready_action_22 = expand_openpi_action_to_lerobot(
            ready_action_16,
            default_head=[-0.013, 0.786],  # 固定头部位置
            default_torso=[0.598, -1.195, 0.597, 0.009],  # 固定腰部位置
        )
        
        waypoint = lerobot_action_to_waypoint(ready_action_22, include_chassis=False)
        
        if self.astribot:
            self.astribot.move_joints_waypoints(
                ASTRIBOT_NAMES_LIST,
                [waypoint],
                [duration],
                use_wbc=self._use_wbc
            )
        else:
            time.sleep(duration)
        
        self._current_waypoint = waypoint
        logger.info("✓ 已到达准备位置")
        return True
    
    def get_current_images(self) -> Optional[List[dict]]:
        """获取当前相机图像"""
        if self.camera_subscriber and self._enable_camera:
            return self.camera_subscriber.get_images_for_inference()
        return None
    
    def step(self, with_images: bool = None) -> bool:
        """
        执行一步推理和控制
        
        Returns:
            True 继续, False 结束
        """
        # 获取本体状态 (16维)
        state = self.get_current_state()
        
        # 获取图像
        images = None
        send_images = with_images if with_images is not None else self._enable_camera
        if send_images:
            images = self.get_current_images()
        
        inference_start_time = time.time()
        is_inference_frame = True
        
        # 根据模式获取 action (16维)
        if self._use_chunk and self._chunk_manager is not None:
            action = self._chunk_manager.get_action(
                state=state,
                images=images,
                episode_id=self._episode_id,
                frame_index=self._frame_index,
                prompt=self._current_prompt
            )
            
            if action is None:
                return False
            
            is_inference_frame = self._chunk_manager.last_action_triggered_inference
        else:
            response = self.inference_client.predict(
                state=state,
                images=images,
                episode_id=self._episode_id,
                frame_index=self._frame_index,
                prompt=self._current_prompt
            )
            
            if response.status == pb2.EPISODE_END or response.is_terminal:
                return False
            
            if response.status != pb2.OK:
                logger.error(f"推理错误: {response.error_message}")
                return False
            
            action = list(response.values)
        
        inference_latency_ms = (time.time() - inference_start_time) * 1000
        
        # 记录推理日志
        if self.inference_logger:
            self.inference_logger.log_step(
                frame_index=self._frame_index,
                state=state,
                action=action,
                prompt=self._current_prompt,
                images=images,
                episode_id=self._episode_id,
                latency_ms=inference_latency_ms,
                extra_info={
                    "use_chunk": self._use_chunk,
                    "is_inference_frame": is_inference_frame
                },
                save_images_this_step=is_inference_frame
            )
        
        # 应用速度限制
        if self.velocity_limiter:
            action = self.velocity_limiter.limit(action)
        
        # 应用平滑
        if self.smoother:
            action = self.smoother.smooth(action)
        
        # 应用夹爪二值化
        if self._binarize_gripper:
            action = binarize_gripper_action(
                action, 
                threshold=self._gripper_threshold,
                gripper_indices=[14, 15]  # OpenPi 格式夹爪索引
            )
        
        # 扩展为 22 维并转换为 waypoint
        action_22 = expand_openpi_action_to_lerobot(action)
        waypoint = lerobot_action_to_waypoint(action_22, include_chassis=False)
        
        # 发送到机器人
        if self.astribot:
            self.astribot.set_joints_position(
                ASTRIBOT_NAMES_LIST,
                waypoint,
                control_way=self.control_way,
                use_wbc=self._use_wbc
            )
        
        self._current_waypoint = waypoint
        self._frame_index += 1
        
        return True
    
    def set_episode(self, episode: int):
        """设置当前 episode"""
        self._episode_id = episode
        self._frame_index = 0
        self.inference_client.set_episode(episode)
        
        if self.smoother:
            self.smoother.reset()
        if self.velocity_limiter:
            self.velocity_limiter.reset()
        if self._chunk_manager:
            self._chunk_manager.reset()
    
    def close(self):
        """关闭控制器"""
        if self.camera_subscriber:
            self.camera_subscriber.stop()
        self.inference_client.close()
        logger.info("控制器已关闭")


def run_inference_loop(
    controller: AstribotController,
    episode: int = 0,
    max_frames: int = 10000,
    move_to_ready: bool = True,
    ready_move_duration: float = 5.0
):
    """
    运行推理控制循环
    
    Args:
        controller: 控制器
        episode: episode 索引
        max_frames: 最大帧数
        move_to_ready: 是否先移动到准备位置
        ready_move_duration: 移动到准备位置的时间 (秒)
    """
    global _interrupted
    _interrupted = False
    
    logger.info("=" * 60)
    logger.info(f"开始推理控制 (episode={episode})")
    logger.info("=" * 60)
    
    # 启动推理日志会话
    if controller.inference_logger:
        controller.inference_logger.start_session(
            episode_id=episode,
            config_name=controller.config.config_name,
            checkpoint_dir=controller.config.checkpoint_dir,
            default_prompt=controller.config.default_prompt,
            config={
                "control_freq": controller.config.control_freq,
                "use_chunk": controller._use_chunk,
                "n_action_steps": controller._n_action_steps,
            }
        )
    
    controller.set_episode(episode)
    
    # 获取服务状态
    status = controller.inference_client.get_status()
    logger.info(f"服务状态: {'就绪' if status.is_ready else '未就绪'}")
    logger.info(f"模型: {status.model_name}")
    logger.info(f"Action horizon: {status.action_horizon}, Action dim: {status.action_dim}")
    
    # 等待图像就绪
    if controller._enable_camera and controller.camera_subscriber:
        logger.info("等待相机图像就绪...")
        if controller.camera_subscriber.wait_for_images(timeout=10.0):
            logger.info("所有相机图像已就绪")
        else:
            logger.warning("部分相机图像未就绪")
    
    # 移动到准备位置
    if move_to_ready:
        if not controller.move_to_ready_position(duration=ready_move_duration):
            logger.error("移动到准备位置失败")
            return
        
        if _interrupted:
            return
    
    # 实时推理控制
    logger.info("=" * 60)
    logger.info(f"开始实时推理 ({max_frames} 帧 @ {controller.control_freq}Hz)")
    logger.info("=" * 60)
    
    controller._frame_index = 0
    control_period = controller.control_period
    
    start_time = time.time()
    frame_count = 0
    
    while not _interrupted and frame_count < max_frames:
        loop_start = time.time()
        
        if not controller.step():
            logger.info("Episode 结束")
            break
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            actual_freq = frame_count / elapsed if elapsed > 0 else 0
            logger.info(f"进度: {frame_count}/{max_frames} | 实际频率: {actual_freq:.1f}Hz")
        
        # 频率控制
        elapsed = time.time() - loop_start
        if elapsed < control_period:
            time.sleep(control_period - elapsed)
    
    total_time = time.time() - start_time
    if frame_count > 0:
        logger.info(f"推理完成! 帧数: {frame_count}, 耗时: {total_time:.2f}s, "
                   f"平均频率: {frame_count/total_time:.1f}Hz")
    
    # 结束推理日志会话
    if controller.inference_logger:
        controller.inference_logger.end_session()


def main():
    """命令行入口"""
    import argparse
    global _interrupted
    
    parser = argparse.ArgumentParser(
        description='OpenPi 推理控制客户端',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用模型推理
  python -m src.client.inference_client --server localhost:50052 \\
      --config pi05_astribot_lora \\
      --checkpoint /path/to/checkpoint/50000 \\
      --prompt "clear up the desktop"
  
  # 启用 Chunk 模式
  python -m src.client.inference_client --server localhost:50052 \\
      --config pi05_astribot_lora \\
      --checkpoint /path/to/checkpoint/50000 \\
      --prompt "clear up the desktop" \\
      --use-chunk --n-action-steps 50
        """
    )
    
    # Server 连接
    parser.add_argument('--server', type=str, default='localhost:50052',
                        help='推理服务器地址')
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='连接超时')
    
    # 策略配置
    parser.add_argument('--config', type=str, default=None,
                        help='OpenPi 训练配置名称')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint 目录')
    parser.add_argument('--prompt', type=str, default="",
                        help='语言指令')
    parser.add_argument('--device', type=str, default=None,
                        help='PyTorch 设备')
    
    # 相机配置
    parser.add_argument('--enable-camera', action='store_true',
                        help='启用相机订阅')
    parser.add_argument('--cameras', type=str, default='head,wrist_left,wrist_right',
                        help='要订阅的相机列表')
    
    # 回放配置
    parser.add_argument('--episode', type=int, default=0,
                        help='Episode 索引')
    parser.add_argument('--max-frames', type=int, default=10000,
                        help='最大帧数')
    
    # 控制配置
    parser.add_argument('--control-freq', type=float, default=30.0,
                        help='控制频率 Hz')
    parser.add_argument('--control-way', type=str, default='direct',
                        choices=['filter', 'direct'],
                        help='控制方式')
    parser.add_argument('--smooth', type=int, default=0,
                        help='平滑窗口大小')
    parser.add_argument('--max-velocity', type=float, default=0.0,
                        help='最大速度 rad/frame')
    
    # Chunk 模式配置
    parser.add_argument('--use-chunk', action='store_true',
                        help='使用 chunk 模式')
    parser.add_argument('--n-action-steps', type=int, default=None,
                        help='每个 chunk 使用的 action 数量')
    
    # 夹爪二值化
    parser.add_argument('--binarize-gripper', action='store_true',
                        help='启用夹爪二值化控制')
    parser.add_argument('--gripper-threshold', type=float, default=0.5,
                        help='夹爪二值化阈值')
    
    # 准备位置
    parser.add_argument('--move-to-ready', action='store_true', default=True,
                        help='启动时移动到准备位置')
    parser.add_argument('--no-move-to-ready', action='store_true',
                        help='禁用移动到准备位置')
    parser.add_argument('--ready-duration', type=float, default=5.0,
                        help='移动到准备位置的时间')
    
    # 日志配置
    parser.add_argument('--enable-logging', action='store_true', default=True,
                        help='启用推理日志记录')
    parser.add_argument('--no-logging', action='store_true',
                        help='禁用推理日志记录')
    parser.add_argument('--log-dir', type=str, default='./inference_logs',
                        help='日志保存目录')
    
    args = parser.parse_args()
    
    setup_logging("INFO")
    
    # 解析服务器地址
    if ':' in args.server:
        host, port = args.server.rsplit(':', 1)
        port = int(port)
    else:
        host = args.server
        port = 50052
    
    # 构建配置
    config = ClientConfig(
        server_host=host,
        server_port=port,
        timeout=args.timeout,
        config_name=args.config,
        checkpoint_dir=args.checkpoint,
        default_prompt=args.prompt,
        pytorch_device=args.device,
        control_freq=args.control_freq,
        control_way=args.control_way,
        smooth_window=args.smooth,
        max_velocity=args.max_velocity,
    )
    
    # 解析相机列表
    camera_names = [c.strip() for c in args.cameras.split(',') if c.strip()]
    
    # 是否移动到准备位置
    move_to_ready = args.move_to_ready and not args.no_move_to_ready
    
    # 创建推理日志记录器
    inference_logger = None
    enable_logging = args.enable_logging and not args.no_logging
    if enable_logging:
        inference_logger = InferenceLogger(
            log_dir=args.log_dir,
            save_images=True,
            enabled=True
        )
    
    controller = None
    
    try:
        controller = AstribotController(
            config,
            enable_camera=args.enable_camera,
            camera_names=camera_names,
            use_chunk=args.use_chunk,
            n_action_steps=args.n_action_steps,
            inference_logger=inference_logger,
            binarize_gripper=args.binarize_gripper,
            gripper_threshold=args.gripper_threshold
        )
        
        run_inference_loop(
            controller,
            episode=args.episode,
            max_frames=args.max_frames,
            move_to_ready=move_to_ready,
            ready_move_duration=args.ready_duration
        )
        
    except KeyboardInterrupt:
        logger.warning("程序被中断")
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if inference_logger:
            inference_logger.end_session()
        if controller:
            controller.close()
        logger.info("程序退出")


if __name__ == '__main__':
    main()

