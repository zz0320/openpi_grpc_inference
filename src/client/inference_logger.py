# -*- coding: utf-8 -*-
"""
推理日志记录器

在 Client 侧记录全部推理信息:
- State (关节位置)
- Action (推理输出)
- Image (图像，保存为图片文件)
- Prompt (语言指令)

日志目录结构:
    inference_logs/
        session_2025-01-09_12-30-45/
            metadata.json          # 会话元信息
            inference_log.jsonl    # 推理数据 (每行一条 JSON 记录)
            images/
                frame_000000/
                    head.jpg
                    wrist_left.jpg
                    wrist_right.jpg
                frame_000001/
                    ...
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger("openpi_inference.client.logger")


@dataclass
class InferenceLogEntry:
    """
    单步推理日志条目
    
    记录完整的 action 处理流水线:
    - raw_action: 模型原始输出 (25维)
    - filtered_action: 部件过滤后 (head/torso/chassis)
    - smoothed_action: 速度限制 + 平滑后
    - action: 最终发送给机器人的 action
    """
    timestamp: float                    # Unix 时间戳
    episode_id: int                     # Episode ID
    frame_index: int                    # 帧索引
    state: List[float]                  # 关节状态
    action: List[float]                 # 最终发送给机器人的 action
    prompt: Optional[str] = None        # 语言指令
    image_paths: Dict[str, str] = None  # 图像文件路径 {camera_name: path}
    latency_ms: Optional[float] = None  # 推理延迟 (毫秒)
    raw_action: Optional[List[float]] = None        # 模型原始输出 (25维)
    filtered_action: Optional[List[float]] = None   # 部件过滤后
    smoothed_action: Optional[List[float]] = None   # 速度限制 + 平滑后
    extra_info: Optional[Dict] = None   # 额外信息
    
    def __post_init__(self):
        if self.image_paths is None:
            self.image_paths = {}


class InferenceLogger:
    """
    推理日志记录器
    
    记录 Client 侧的全部推理信息，包括 state、action、prompt 和图像。
    """
    
    def __init__(
        self,
        log_dir: str = "./inference_logs",
        session_name: Optional[str] = None,
        save_images: bool = True,
        image_format: str = "jpg",
        image_quality: int = 95,
        flush_interval: int = 10,
        enabled: bool = True
    ):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志根目录
            session_name: 会话名称 (默认: 自动生成时间戳)
            save_images: 是否保存图像
            image_format: 图像格式 ("jpg", "png")
            image_quality: 图像质量 (仅 JPEG, 1-100)
            flush_interval: 每多少步刷新一次文件
            enabled: 是否启用日志记录
        """
        self.log_dir = log_dir
        self.save_images = save_images
        self.image_format = image_format.lower()
        self.image_quality = image_quality
        self.flush_interval = flush_interval
        self.enabled = enabled
        
        # 会话目录 (自动附加时间戳)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if session_name:
            self.session_name = f"{session_name}_{timestamp}"
        else:
            self.session_name = f"session_{timestamp}"
        
        self.session_dir = os.path.join(log_dir, self.session_name)
        self.images_dir = os.path.join(self.session_dir, "images")
        
        # 日志文件
        self.metadata_path = os.path.join(self.session_dir, "metadata.json")
        self.log_path = os.path.join(self.session_dir, "inference_log.jsonl")
        
        # 状态
        self._log_file = None
        self._session_started = False
        self._step_count = 0
        self._metadata = {}
        
        # 统计
        self._total_frames = 0
        self._total_images_saved = 0
    
    def start_session(
        self,
        episode_id: int = 0,
        config_name: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        default_prompt: Optional[str] = None,
        config: Optional[Dict] = None,
        **extra_metadata
    ):
        """
        开始新的日志会话
        
        Args:
            episode_id: Episode ID
            config_name: OpenPi 配置名称
            checkpoint_dir: Checkpoint 目录
            default_prompt: 默认 prompt
            config: 配置信息
            **extra_metadata: 额外的元信息
        """
        if not self.enabled:
            return
        
        # 创建目录
        os.makedirs(self.session_dir, exist_ok=True)
        if self.save_images:
            os.makedirs(self.images_dir, exist_ok=True)
        
        # 构建元信息
        self._metadata = {
            "session_name": self.session_name,
            "start_time": datetime.now().isoformat(),
            "start_timestamp": time.time(),
            "episode_id": episode_id,
            "config_name": config_name,
            "checkpoint_dir": checkpoint_dir,
            "default_prompt": default_prompt,
            "config": config or {},
            "save_images": self.save_images,
            "image_format": self.image_format,
            **extra_metadata
        }
        
        # 写入元信息
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        
        # 打开日志文件
        self._log_file = open(self.log_path, 'w', encoding='utf-8')
        
        self._session_started = True
        self._step_count = 0
        self._total_frames = 0
        self._total_images_saved = 0
        
        logger.info(f"推理日志会话已开始: {self.session_dir}")
    
    def log_step(
        self,
        frame_index: int,
        state: List[float],
        action: List[float],
        prompt: Optional[str] = None,
        images: Optional[List[Dict]] = None,
        episode_id: int = 0,
        latency_ms: Optional[float] = None,
        raw_action: Optional[List[float]] = None,
        filtered_action: Optional[List[float]] = None,
        smoothed_action: Optional[List[float]] = None,
        extra_info: Optional[Dict] = None,
        save_images_this_step: bool = True
    ):
        """
        记录单步推理信息
        
        Args:
            frame_index: 帧索引
            state: 关节状态 (float list)
            action: 最终发送给机器人的 action (float list)
            prompt: 当前使用的语言指令
            images: 图像列表 [{'name': 'head', 'data': bytes, ...}, ...]
            episode_id: Episode ID
            latency_ms: 推理延迟 (毫秒)
            raw_action: 模型原始输出 (25维)
            filtered_action: 部件过滤后的 action
            smoothed_action: 速度限制 + 平滑后的 action
            extra_info: 额外信息
            save_images_this_step: 是否在当前步保存图像
        """
        if not self.enabled or not self._session_started:
            return
        
        # 保存图像
        image_paths = {}
        if self.save_images and images and save_images_this_step:
            image_paths = self._save_images(frame_index, images)
        
        # 构建日志条目
        entry = InferenceLogEntry(
            timestamp=time.time(),
            episode_id=episode_id,
            frame_index=frame_index,
            state=list(state) if isinstance(state, np.ndarray) else state,
            action=list(action) if isinstance(action, np.ndarray) else action,
            prompt=prompt,
            image_paths=image_paths,
            latency_ms=latency_ms,
            raw_action=list(raw_action) if raw_action is not None else None,
            filtered_action=list(filtered_action) if filtered_action is not None else None,
            smoothed_action=list(smoothed_action) if smoothed_action is not None else None,
            extra_info=extra_info
        )
        
        # 写入 JSONL
        entry_dict = asdict(entry)
        self._log_file.write(json.dumps(entry_dict, ensure_ascii=False) + '\n')
        
        self._step_count += 1
        self._total_frames += 1
        
        # 定期刷新
        if self._step_count % self.flush_interval == 0:
            self._log_file.flush()
    
    def _save_images(self, frame_index: int, images: List[Dict]) -> Dict[str, str]:
        """保存图像到文件"""
        image_paths = {}
        
        # 创建帧目录
        frame_dir = os.path.join(self.images_dir, f"frame_{frame_index:06d}")
        os.makedirs(frame_dir, exist_ok=True)
        
        for img_info in images:
            camera_name = img_info.get('name', 'unknown')
            img_data = img_info.get('data', b'')
            encoding = img_info.get('encoding', 'jpeg').lower()
            
            if not img_data:
                continue
            
            # 确定文件扩展名
            ext = 'jpg' if encoding in ['jpeg', 'jpg'] else 'png' if encoding == 'png' else self.image_format
            
            # 文件路径
            filename = f"{camera_name}.{ext}"
            filepath = os.path.join(frame_dir, filename)
            relative_path = os.path.join("images", f"frame_{frame_index:06d}", filename)
            
            try:
                if encoding in ['jpeg', 'jpg', 'png']:
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                else:
                    self._save_raw_image(img_data, img_info, filepath)
                
                image_paths[camera_name] = relative_path
                self._total_images_saved += 1
                
            except Exception as e:
                logger.warning(f"保存图像失败 [{camera_name}]: {e}")
        
        return image_paths
    
    def _save_raw_image(self, img_data: bytes, img_info: Dict, filepath: str):
        """保存 raw 格式图像"""
        try:
            from PIL import Image
            
            width = img_info.get('width', 0)
            height = img_info.get('height', 0)
            
            if width <= 0 or height <= 0:
                return
            
            img = Image.frombytes('RGB', (width, height), img_data)
            
            if filepath.endswith('.jpg'):
                img.save(filepath, 'JPEG', quality=self.image_quality)
            else:
                img.save(filepath, 'PNG')
                
        except Exception as e:
            logger.warning(f"保存 raw 图像失败: {e}")
    
    def end_session(self):
        """结束日志会话"""
        if not self.enabled or not self._session_started:
            return
        
        # 更新元信息
        self._metadata["end_time"] = datetime.now().isoformat()
        self._metadata["end_timestamp"] = time.time()
        self._metadata["total_frames"] = self._total_frames
        self._metadata["total_images_saved"] = self._total_images_saved
        
        duration = self._metadata["end_timestamp"] - self._metadata["start_timestamp"]
        self._metadata["duration_seconds"] = duration
        
        if duration > 0 and self._total_frames > 0:
            self._metadata["avg_fps"] = self._total_frames / duration
        
        # 写入更新后的元信息
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        
        # 关闭日志文件
        if self._log_file:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None
        
        self._session_started = False
        
        logger.info(f"推理日志会话已结束: {self._total_frames} 帧")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_session()


class InferenceLogReader:
    """推理日志读取器"""
    
    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.metadata_path = os.path.join(session_dir, "metadata.json")
        self.log_path = os.path.join(session_dir, "inference_log.jsonl")
    
    def get_metadata(self) -> Dict:
        """读取元信息"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def iter_entries(self):
        """迭代日志条目"""
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    
    def load_as_arrays(self) -> tuple:
        """将 state 和 action 加载为 numpy 数组"""
        states = []
        actions = []
        
        for entry in self.iter_entries():
            states.append(entry['state'])
            actions.append(entry['action'])
        
        return np.array(states), np.array(actions)

