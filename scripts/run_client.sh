#!/bin/bash
# 启动 OpenPi 推理客户端
# Client 连接 Server 并指定要加载的模型

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 必填参数 - 模型配置
CONFIG=${CONFIG:-"pi05_astribot_lora"}
CHECKPOINT=${CHECKPOINT:-""}  # 必须指定

# Server 连接
SERVER=${SERVER:-"localhost:50052"}

# 推理配置
PROMPT=${PROMPT:-"clear up the desktop"}
DEVICE=${DEVICE:-"cuda"}
EPISODE=${EPISODE:-0}
CONTROL_FREQ=${CONTROL_FREQ:-30}

# Chunk 模式配置
USE_CHUNK=${USE_CHUNK:-true}
N_ACTION_STEPS=${N_ACTION_STEPS:-50}

# 相机配置
ENABLE_CAMERA=${ENABLE_CAMERA:-false}
CAMERAS=${CAMERAS:-"head,wrist_left,wrist_right"}

# 检查必填参数
if [ -z "$CHECKPOINT" ]; then
    echo "错误: 必须指定 CHECKPOINT 路径"
    echo ""
    echo "使用方法:"
    echo "  CHECKPOINT=/path/to/checkpoint ./scripts/run_client.sh"
    echo ""
    echo "示例:"
    echo "  CHECKPOINT=/root/openpi/checkpoints/pi05_astribot_lora/astribot_lora_exp1/79999 \\"
    echo "  SERVER=192.168.1.100:50052 \\"
    echo "  PROMPT=\"clear up the desktop\" \\"
    echo "  ./scripts/run_client.sh"
    exit 1
fi

echo "=========================================="
echo "OpenPi 推理客户端"
echo "=========================================="
echo "Server: $SERVER"
echo "配置: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Prompt: $PROMPT"
echo "设备: $DEVICE"
echo "Episode: $EPISODE"
echo "控制频率: $CONTROL_FREQ Hz"
echo "Chunk 模式: $USE_CHUNK"
if [ "$USE_CHUNK" = "true" ]; then
    echo "N Action Steps: $N_ACTION_STEPS"
fi
echo "启用相机: $ENABLE_CAMERA"
echo "=========================================="

# 构建命令
CMD="python3 -m src.client.inference_client"
CMD="$CMD --server $SERVER"
CMD="$CMD --config $CONFIG"
CMD="$CMD --checkpoint $CHECKPOINT"
CMD="$CMD --prompt \"$PROMPT\""
CMD="$CMD --device $DEVICE"
CMD="$CMD --episode $EPISODE"
CMD="$CMD --control-freq $CONTROL_FREQ"

if [ "$USE_CHUNK" = "true" ]; then
    CMD="$CMD --use-chunk"
    if [ -n "$N_ACTION_STEPS" ]; then
        CMD="$CMD --n-action-steps $N_ACTION_STEPS"
    fi
fi

if [ "$ENABLE_CAMERA" = "true" ]; then
    CMD="$CMD --enable-camera"
    CMD="$CMD --cameras $CAMERAS"
fi

echo ""
echo "执行: $CMD"
echo ""

eval $CMD
