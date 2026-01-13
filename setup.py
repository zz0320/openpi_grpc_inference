#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenPi gRPC 推理框架安装脚本
"""

from setuptools import setup, find_packages

setup(
    name="openpi_grpc_inference",
    version="0.1.0",
    description="OpenPi 模型 gRPC 推理服务",
    author="OpenPi Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "grpcio>=1.50.0",
        "grpcio-tools>=1.50.0",
        "protobuf>=4.21.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "server": [
            # openpi 需要单独安装
        ],
        "client": [
            # Astribot SDK 需要单独安装
        ],
    },
    entry_points={
        "console_scripts": [
            "openpi-server=src.server.inference_server:main",
            "openpi-client=src.client.inference_client:main",
        ],
    },
)

