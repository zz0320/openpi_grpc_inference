#!/bin/bash
# 启动 OpenPi 推理服务器 (空闲模式)
# Server 启动后等待 Client 指定模型配置
# 使用 uv 环境运行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# OpenPi 项目路径 (uv 环境所在位置)
OPENPI_DIR=${OPENPI_DIR:-"/root/openpi"}

cd "$PROJECT_DIR"

# 默认参数
PORT=${PORT:-50052}
DEVICE=${DEVICE:-"cuda"}
HOST=${HOST:-"0.0.0.0"}
WORKERS=${WORKERS:-10}

echo "=========================================="
echo "OpenPi 推理服务器 (空闲模式)"
echo "=========================================="
echo "OpenPi 目录: $OPENPI_DIR"
echo "监听地址: $HOST:$PORT"
echo "设备: $DEVICE"
echo "工作线程: $WORKERS"
echo ""
echo "Server 将以空闲模式启动，等待 Client 指定模型"
echo "=========================================="

# 设置 PYTHONPATH 包含当前项目
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# 构建命令参数 (不包含 config 和 checkpoint)
ARGS="--host $HOST --port $PORT --device $DEVICE --workers $WORKERS"

# 使用 uv run 在 openpi 环境中执行
echo ""
echo "执行命令:"
echo "  cd $OPENPI_DIR"
echo "  PYTHONPATH=$PROJECT_DIR:\$PYTHONPATH uv run python $PROJECT_DIR/src/server/inference_server.py $ARGS"
echo ""

cd "$OPENPI_DIR"
uv run python "$PROJECT_DIR/src/server/inference_server.py" $ARGS
