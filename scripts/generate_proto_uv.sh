#!/bin/bash
# 生成 protobuf 代码
# 使用 OpenPi 的 uv 环境 (适用于 Server 端)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# OpenPi 目录
OPENPI_DIR=${OPENPI_DIR:-"/root/openpi"}

PROTO_DIR="$PROJECT_DIR/proto"
OUTPUT_DIR="$PROJECT_DIR/src/generated"

echo "生成 protobuf 代码 (uv 环境)..."
echo "  OpenPi 目录: $OPENPI_DIR"
echo "  Proto 目录: $PROTO_DIR"
echo "  输出目录: $OUTPUT_DIR"

# 检查 OpenPi 目录
if [ ! -d "$OPENPI_DIR" ]; then
    echo "✗ OpenPi 目录不存在: $OPENPI_DIR"
    echo "  请设置 OPENPI_DIR 环境变量或使用 generate_proto.sh"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# 使用 uv 环境
cd "$OPENPI_DIR"
uv run python -m grpc_tools.protoc \
    --proto_path="$PROTO_DIR" \
    --python_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    "$PROTO_DIR/openpi_inference.proto"

if [ $? -eq 0 ]; then
    echo "✓ 生成成功!"
    touch "$OUTPUT_DIR/__init__.py"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/import openpi_inference_pb2/from . import openpi_inference_pb2/' "$OUTPUT_DIR/openpi_inference_pb2_grpc.py"
    else
        sed -i 's/import openpi_inference_pb2/from . import openpi_inference_pb2/' "$OUTPUT_DIR/openpi_inference_pb2_grpc.py"
    fi
    
    echo "  生成文件:"
    ls -la "$OUTPUT_DIR"
else
    echo "✗ 生成失败!"
    exit 1
fi

