#!/bin/bash
# 生成 protobuf 代码
# 使用 uv 环境运行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# OpenPi 项目路径 (uv 环境所在位置)
OPENPI_DIR=${OPENPI_DIR:-"/root/openpi"}

PROTO_DIR="$PROJECT_DIR/proto"
OUTPUT_DIR="$PROJECT_DIR/src/generated"

echo "生成 protobuf 代码..."
echo "  OpenPi 目录: $OPENPI_DIR"
echo "  Proto 目录: $PROTO_DIR"
echo "  输出目录: $OUTPUT_DIR"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 使用 uv run 在 openpi 环境中生成 protobuf 代码
cd "$OPENPI_DIR"
uv run python -m grpc_tools.protoc \
    --proto_path="$PROTO_DIR" \
    --python_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    "$PROTO_DIR/openpi_inference.proto"

if [ $? -eq 0 ]; then
    echo "✓ 生成成功!"
    
    # 创建 __init__.py
    touch "$OUTPUT_DIR/__init__.py"
    
    # 修复 import 路径问题
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' 's/import openpi_inference_pb2/from . import openpi_inference_pb2/' "$OUTPUT_DIR/openpi_inference_pb2_grpc.py"
    else
        # Linux
        sed -i 's/import openpi_inference_pb2/from . import openpi_inference_pb2/' "$OUTPUT_DIR/openpi_inference_pb2_grpc.py"
    fi
    
    echo "  生成文件:"
    ls -la "$OUTPUT_DIR"
else
    echo "✗ 生成失败!"
    exit 1
fi

