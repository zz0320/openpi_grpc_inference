# OpenPi gRPC 推理框架

基于 `lerobot_grpc_inference` 架构的 OpenPi (Pi0/Pi0.5) 模型 gRPC 推理服务。

支持将 OpenPi 训练的模型通过 Client/Server 分离的方式进行远程推理，适用于 Astribot S1 机器人控制。

## 📋 特性

- ✅ 支持 OpenPi (Pi0/Pi0.5) 训练模型的远程推理
- ✅ Client/Server 分离架构，支持跨机器部署
- ✅ **Client 端指定模型**，Server 以空闲模式启动
- ✅ 支持 Action Chunking，减少网络延迟
- ✅ 支持语言指令 (prompt) 控制
- ✅ 支持视觉输入 (多相机图像: head, wrist_left, wrist_right)
- ✅ 支持 Astribot S1 机器人控制集成
- ✅ 完整的推理日志记录 (state, action, image)
- ✅ 兼容 OpenPi 的 uv 环境

## 📁 项目结构

```
openpi_grpc_inference/
├── config/
│   └── default.json              # 默认配置
├── proto/
│   └── openpi_inference.proto    # gRPC 协议定义
├── scripts/
│   ├── generate_proto.sh         # 生成 protobuf 代码
│   ├── run_server.sh             # 启动服务器脚本
│   └── run_client.sh             # 启动客户端脚本
├── src/
│   ├── client/
│   │   ├── inference_client.py   # 推理客户端 + Astribot 控制器
│   │   └── inference_logger.py   # 推理日志记录器
│   ├── common/
│   │   ├── config.py             # 配置管理
│   │   ├── constants.py          # 常量定义
│   │   └── utils.py              # 工具函数
│   ├── generated/                # protobuf 生成代码
│   │   ├── openpi_inference_pb2.py
│   │   └── openpi_inference_pb2_grpc.py
│   └── server/
│       └── inference_server.py   # 推理服务器 (使用 OpenPi Policy)
├── requirements.txt
├── requirements-server.txt
├── requirements-client.txt
├── setup.py
└── README.md
```

## 🔧 安装配置

### 前置条件

- OpenPi 项目已配置好 uv 环境 (`/root/openpi`)
- 已训练好的 checkpoint (如 `pi05_astribot_lora`)

### Server 端 (GPU 服务器)

```bash
# 1. 在 OpenPi 项目中安装 gRPC 依赖
cd /root/openpi
uv add grpcio grpcio-tools

# 2. 生成 protobuf 代码
cd /root/openpi_grpc_inference
OPENPI_DIR=/root/openpi ./scripts/generate_proto.sh
```

### Client 端 (机器人侧)

```bash
cd /root/openpi_grpc_inference
pip install -r requirements-client.txt

# 如果需要单独生成 protobuf
pip install grpcio-tools
python3 -m grpc_tools.protoc \
    --proto_path=proto \
    --python_out=src/generated \
    --grpc_python_out=src/generated \
    proto/openpi_inference.proto
```

## 🚀 使用方法

### 1. 启动 Server (GPU 服务器) - 空闲模式

Server 以**空闲模式**启动，等待 Client 指定要加载的模型：

```bash
cd /root/openpi

# 启动空闲模式 Server (等待 Client 配置)
PYTHONPATH=/root/openpi_grpc_inference:$PYTHONPATH \
uv run python /root/openpi_grpc_inference/src/server/inference_server.py \
    --port 50052 \
    --device cuda
```

**Server 参数说明:**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--port` | 监听端口 | 50052 |
| `--device` | PyTorch 设备 | cuda |
| `--host` | 监听地址 | 0.0.0.0 |
| `--workers` | 工作线程数 | 10 |

> **注意**: Server 启动后会显示 "等待 Client 配置..."，直到 Client 连接并指定模型。

### 2. 启动 Client (机器人侧) - 指定模型

Client 连接 Server 并**指定要加载的模型**：

```bash
cd /root/openpi_grpc_inference

# Client 指定模型路径和配置
python3 -m src.client.inference_client \
    --server <GPU服务器IP>:50052 \
    --config pi05_astribot_lora \
    --checkpoint /path/to/checkpoints/pi05_astribot_lora/astribot_lora_exp1/79999 \
    --prompt "clear up the desktop" \
    --device cuda

# 完整示例 (启用 Chunk 模式 + 相机)
python3 -m src.client.inference_client \
    --server <GPU服务器IP>:50052 \
    --config pi05_astribot_lora \
    --checkpoint /path/to/checkpoints/pi05_astribot_lora/astribot_lora_exp1/79999 \
    --prompt "clear up the desktop" \
    --device cuda \
    --use-chunk \
    --n-action-steps 50 \
    --enable-camera \
    --control-freq 30
```

**Client 参数说明:**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--server` | Server 地址 | localhost:50052 |
| **`--config`** | **OpenPi 训练配置名称** | - |
| **`--checkpoint`** | **Checkpoint 目录路径** | - |
| `--prompt` | 语言指令 | - |
| `--device` | Server 端推理设备 | cuda |
| `--use-chunk` | 启用 Chunk 模式 | False |
| `--n-action-steps` | 每个 chunk 使用的 action 数 | 50 |
| `--enable-camera` | 启用相机订阅 | False |
| `--control-freq` | 控制频率 (Hz) | 30 |
| `--episode` | Episode 索引 | 0 |
| `--max-frames` | 最大帧数 | 10000 |
| `--smooth` | 平滑窗口大小 | 0 |
| `--max-velocity` | 最大速度限制 (rad/frame) | 0 |
| `--binarize-gripper` | 夹爪二值化 | False |
| `--move-to-ready` | 先移动到准备位置 | True |
| `--enable-logging` | 启用推理日志 | True |
| `--log-dir` | 日志保存目录 | ./inference_logs |

## 🏗️ 架构说明

### 工作流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              工作流程                                        │
│                                                                             │
│  1. Server 启动 (空闲模式)                                                   │
│     └── 等待 Client 连接                                                     │
│                                                                             │
│  2. Client 连接并发送 Configure 请求                                         │
│     └── 包含: config_name, checkpoint_dir, device, prompt                   │
│                                                                             │
│  3. Server 收到配置后加载模型                                                 │
│     └── 使用 openpi.policies.policy_config.create_trained_policy()          │
│                                                                             │
│  4. Server 返回就绪状态                                                      │
│                                                                             │
│  5. Client 开始发送观测数据，Server 返回 action                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 整体架构

```
┌─────────────────────────┐         gRPC          ┌─────────────────────────┐
│      Client 端          │ ◄──────────────────►  │      Server 端          │
│     (机器人侧)           │                       │    (GPU 服务器)          │
│                         │                       │                         │
│  • 指定模型配置          │   Configure           │  • 空闲模式启动          │
│  • 采集机器人状态 (16维)  │ ──────────────────►   │  • 按需加载模型          │
│  • 采集相机图像          │                       │  • 执行 GPU 推理         │
│  • 发送观测数据          │   Observation         │  • 返回 action chunk     │
│  • 接收 action          │ ──────────────────►   │                         │
│  • 控制机器人执行        │   ActionChunk         │                         │
│                         │ ◄──────────────────   │                         │
└─────────────────────────┘                       └─────────────────────────┘
```

### 数据流

```
Client 指定模型 ─────► Server 加载模型 (首次连接)
       │
       ▼
机器人状态 (16维)  ─┐
                   ├──► gRPC 请求 ──► OpenPi Policy ──► actions (10, 16)
相机图像 (3张)     ─┤                                          │
                   │                                           │
语言指令 (prompt) ─┘                                           ▼
                                                         ActionChunk
                                                               │
                      ┌────────────────────────────────────────┘
                      ▼
              Client 本地消费 action
                      │
                      ▼
              扩展为 22 维 (添加 head, torso)
                      │
                      ▼
              转换为 waypoint 格式
                      │
                      ▼
              发送到 Astribot 执行
```

### Action 维度说明

| 格式 | 维度 | 内容 |
|------|------|------|
| OpenPi 模型输出 | 16 | arm_left(7) + arm_right(7) + gripper(2) |
| LeRobot V2.0 | 22 | 上述 + head(2) + torso(4) |
| LeRobot V2.0 (含底盘) | 25 | 上述 + chassis(3) |

## 📊 性能指标

| 指标 | 典型值 |
|------|--------|
| 模型加载时间 | ~10-30s (首次) |
| 模型推理延迟 | ~50-100ms |
| 网络传输延迟 | ~5-20ms (局域网) |
| 控制频率 | 30 Hz |
| Action Horizon | 10 步 |

## 📝 推理日志

启用日志记录后，会在 `--log-dir` 目录下创建日志：

```
inference_logs/
└── session_2025-01-13_12-30-45/
    ├── metadata.json          # 会话元信息
    ├── inference_log.jsonl    # 推理数据 (JSONL 格式)
    └── images/                # 保存的图像
        ├── frame_000000/
        │   ├── head.jpg
        │   ├── wrist_left.jpg
        │   └── wrist_right.jpg
        └── ...
```

## 🔗 与 OpenPi 的关系

本框架 Server 端依赖 OpenPi 项目的以下组件：

| 组件 | 用途 |
|------|------|
| `openpi.training.config` | 获取训练配置 (如 `pi05_astribot_lora`) |
| `openpi.policies.policy_config` | 创建 Policy 对象 |
| `openpi.policies.policy` | 执行模型推理 |

## 🆚 与 lerobot_grpc_inference 对比

| 特性 | lerobot_grpc_inference | openpi_grpc_inference |
|------|------------------------|----------------------|
| 模型框架 | LeRobot (ACT/Diffusion) | OpenPi (Pi0/Pi0.5) |
| 模型配置方式 | **Client 端指定** | **Client 端指定** |
| Action 维度 | 22/25 维 | 16 维 (自动扩展) |
| 语言指令 | ❌ 不支持 | ✅ 支持 prompt |
| 默认端口 | 50051 | 50052 |
| 数据集回放 | ✅ 支持 | ❌ 暂不支持 |

## ❓ 常见问题

### 1. Server 显示 "等待 Client 配置..."

这是正常的！Server 以空闲模式启动，等待 Client 连接并指定模型。

### 2. Client 连接后 Server 加载模型失败

检查：
- `--config` 配置名称是否正确 (如 `pi05_astribot_lora`)
- `--checkpoint` 路径是否存在
- Server 端是否能访问该 checkpoint 路径

### 3. 找不到 openpi 模块

```bash
# 确保在 openpi 目录下使用 uv run，并设置 PYTHONPATH
cd /root/openpi
PYTHONPATH=/root/openpi_grpc_inference:$PYTHONPATH uv run python ...
```

### 4. protobuf 代码未生成

```bash
# 重新生成
cd /root/openpi_grpc_inference
OPENPI_DIR=/root/openpi ./scripts/generate_proto.sh
```

### 5. 连接 Server 超时

- 检查 Server 是否正常启动
- 检查端口是否开放 (防火墙)
- 检查 IP 地址是否正确

### 6. 推理延迟过高

- 确保 Server 使用 GPU (`--device cuda`)
- 使用 Chunk 模式减少网络调用
- 检查网络延迟

## 📜 License

MIT License
