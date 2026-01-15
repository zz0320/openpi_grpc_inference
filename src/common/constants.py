# -*- coding: utf-8 -*-
"""
常量定义

OpenPi + Astribot S1 机器人配置

模型输出维度: 25维 [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)]
实际使用维度: 22维 (不含底盘)
"""

# ============================================================================
# OpenPi 模型维度配置
# ============================================================================

# 模型输出维度 (完整 25 维)
OPENPI_MODEL_OUTPUT_DIM = 25

# 执行维度 (不含底盘 22 维)
OPENPI_ACTION_DIM_NO_CHASSIS = 22

# 执行维度 (含底盘 25 维)
OPENPI_ACTION_DIM_WITH_CHASSIS = 25

# 默认使用不含底盘的维度
OPENPI_ACTION_DIM = OPENPI_ACTION_DIM_NO_CHASSIS
OPENPI_STATE_DIM = OPENPI_ACTION_DIM_NO_CHASSIS

# 含底盘的状态维度
OPENPI_STATE_DIM_WITH_CHASSIS = OPENPI_ACTION_DIM_WITH_CHASSIS

# Action horizon (Pi0/Pi0.5 默认)
DEFAULT_ACTION_HORIZON = 10

# ============================================================================
# LeRobot 数据集维度配置 (用于兼容)
# ============================================================================

# V2.0 数据集 (不含底盘): [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4)]
LEROBOT_ACTION_DIM_NO_CHASSIS = 22

# V2.0 数据集 (含底盘): 上述 + chassis(3)
LEROBOT_ACTION_DIM_WITH_CHASSIS = 25

# 默认使用不含底盘的维度
LEROBOT_ACTION_DIM = LEROBOT_ACTION_DIM_NO_CHASSIS

# ============================================================================
# Action 各部件维度配置
# ============================================================================

ACTION_DIM_CONFIG = {
    'arm_left': 7,
    'arm_right': 7,
    'gripper_left': 1,
    'gripper_right': 1,
    'head': 2,
    'torso': 4,
    'chassis': 3,
}

# OpenPi 模型输出维度配置 (完整 25 维)
OPENPI_DIM_CONFIG = {
    'arm_left': 7,
    'arm_right': 7,
    'gripper_left': 1,
    'gripper_right': 1,
    'head': 2,
    'torso': 4,
    'chassis': 3,
}

# 各部件在 OpenPi action 数组中的起始索引 (25 维格式)
OPENPI_ACTION_INDEX = {
    'arm_left': (0, 7),        # [0:7]
    'arm_right': (7, 14),      # [7:14]
    'gripper_left': (14, 15),  # [14:15]
    'gripper_right': (15, 16), # [15:16]
    'head': (16, 18),          # [16:18]
    'torso': (18, 22),         # [18:22]
    'chassis': (22, 25),       # [22:25]
}

# 各部件在 LeRobot V2.0 action 数组中的起始索引 (与 OpenPi 一致)
LEROBOT_ACTION_INDEX = {
    'arm_left': (0, 7),        # [0:7]
    'arm_right': (7, 14),      # [7:14]
    'gripper_left': (14, 15),  # [14:15]
    'gripper_right': (15, 16), # [15:16]
    'head': (16, 18),          # [16:18]
    'torso': (18, 22),         # [18:22]
    'chassis': (22, 25),       # [22:25]
}

# ============================================================================
# 关节索引
# ============================================================================

ARM_LEFT_INDICES = list(range(0, 7))
ARM_RIGHT_INDICES = list(range(7, 14))
ARM_INDICES = ARM_LEFT_INDICES + ARM_RIGHT_INDICES

# 夹爪索引
GRIPPER_LEFT_INDEX = 14
GRIPPER_RIGHT_INDEX = 15

# 头部索引
HEAD_INDICES = list(range(16, 18))

# 腰部索引
TORSO_INDICES = list(range(18, 22))

# 底盘索引
CHASSIS_INDICES = list(range(22, 25))

# ============================================================================
# Astribot 部件配置
# ============================================================================

# 不含底盘的部件列表 (默认使用)
ASTRIBOT_NAMES_LIST = [
    'astribot_torso',
    'astribot_arm_left',
    'astribot_gripper_left',
    'astribot_arm_right',
    'astribot_gripper_right',
    'astribot_head',
]

# 含底盘的部件列表
ASTRIBOT_NAMES_LIST_WITH_CHASSIS = [
    'astribot_torso',
    'astribot_arm_left',
    'astribot_gripper_left',
    'astribot_arm_right',
    'astribot_gripper_right',
    'astribot_head',
    'astribot_chassis',
]

ASTRIBOT_DOF_CONFIG = {
    'astribot_torso': 4,
    'astribot_arm_left': 7,
    'astribot_gripper_left': 1,
    'astribot_arm_right': 7,
    'astribot_gripper_right': 1,
    'astribot_head': 2,
    'astribot_chassis': 3,
}

# ============================================================================
# 相机配置
# ============================================================================

# OpenPi 模型期望的相机名称
OPENPI_CAMERA_NAMES = ['head', 'wrist_left', 'wrist_right']

# ROS 图像话题 -> 相机名称
ASTRIBOT_IMAGE_TOPICS = {
    '/astribot_camera/head_rgbd/color_compress/compressed': 'head',
    '/astribot_camera/left_wrist_rgbd/color_compress/compressed': 'wrist_left',
    '/astribot_camera/right_wrist_rgbd/color_compress/compressed': 'wrist_right',
    '/astribot_camera/torso_rgbd/color_compress/compressed': 'torso',
}

# 图像尺寸 (H, W, C)
ASTRIBOT_IMAGE_SHAPES = {
    'head': (720, 1280, 3),
    'wrist_left': (360, 640, 3),
    'wrist_right': (360, 640, 3),
    'torso': (720, 1280, 3),
}

# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_CONTROL_FREQ = 30.0
DEFAULT_GRPC_PORT = 50052  # 使用不同于 lerobot 的端口
DEFAULT_GRPC_HOST = "0.0.0.0"

# gRPC 配置
GRPC_MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100MB (图像较大)
GRPC_KEEPALIVE_TIME_MS = 10000
GRPC_KEEPALIVE_TIMEOUT_MS = 5000

# ============================================================================
# 机器人准备位置 (Ready Position)
# ============================================================================

# 22维准备位置 (不含底盘)
# 格式: [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4)]
READY_POSITION_22 = [
    # arm_left (7)
    0.154849, -0.022670, -1.421605, 1.660323, -0.346889, 0.115219, 0.126036,
    # arm_right (7)
    -0.161952, -0.022760, 1.418778, 1.660055, 0.343307, 0.115222, -0.123617,
    # gripper_left (1)
    0.0,
    # gripper_right (1)
    0.0,
    # head (2)
    -0.013063, 0.786349,
    # torso (4)
    0.597646, -1.195333, 0.597043, 0.009469,
]

# 25维准备位置 (含底盘)
# 格式: [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)]
READY_POSITION_25 = READY_POSITION_22 + [
    # chassis (3)
    -0.000426, 0.002229, -0.069377,
]

