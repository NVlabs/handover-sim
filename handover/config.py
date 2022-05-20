# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import easysim

from yacs.config import CfgNode as CN

_C = easysim.cfg

# ---------------------------------------------------------------------------- #
# Simulation config
# ---------------------------------------------------------------------------- #
_C.SIM.USE_DEFAULT_STEP_PARAMS = False

_C.SIM.TIME_STEP = 0.001

# ---------------------------------------------------------------------------- #
# Environment config
# ---------------------------------------------------------------------------- #
_C.ENV = CN()


_C.ENV.TABLE_BASE_POSITION = (0.61, 0.28, 0.0)

_C.ENV.TABLE_BASE_ORIENTATION = (0, 0, 0, 1)

_C.ENV.TABLE_HEIGHT = 0.92


_C.ENV.PANDA_BASE_POSITION = (0.61, -0.50, 0.875)

_C.ENV.PANDA_BASE_ORIENTATION = (0.0, 0.0, 0.7071068, 0.7071068)

_C.ENV.PANDA_INITIAL_POSITION = (0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04)

_C.ENV.PANDA_MAX_FORCE = (250.0,) * 9

_C.ENV.PANDA_POSITION_GAIN = (0.01,) * 9

_C.ENV.PANDA_VELOCITY_GAIN = (1.0,) * 9


_C.ENV.YCB_TRANSLATION_MAX_FORCE = (50.0,) * 3

_C.ENV.YCB_TRANSLATION_POSITION_GAIN = (0.2,) * 3

_C.ENV.YCB_TRANSLATION_VELOCITY_GAIN = (1.0,) * 3

_C.ENV.YCB_ROTATION_MAX_FORCE = (5.0,) * 3

_C.ENV.YCB_ROTATION_POSITION_GAIN = (0.2,) * 3

_C.ENV.YCB_ROTATION_VELOCITY_GAIN = (1.0,) * 3


_C.ENV.MANO_TRANSLATION_MAX_FORCE = (50.0,) * 3

_C.ENV.MANO_TRANSLATION_POSITION_GAIN = (0.2,) * 3

_C.ENV.MANO_TRANSLATION_VELOCITY_GAIN = (1.0,) * 3

_C.ENV.MANO_ROTATION_MAX_FORCE = (5.0,) * 3

_C.ENV.MANO_ROTATION_POSITION_GAIN = (0.2,) * 3

_C.ENV.MANO_ROTATION_VELOCITY_GAIN = (1.0,) * 3

_C.ENV.MANO_JOINT_MAX_FORCE = (0.5,) * 45

_C.ENV.MANO_JOINT_POSITION_GAIN = (0.1,) * 45

_C.ENV.MANO_JOINT_VELOCITY_GAIN = (1.0,) * 45


_C.ENV.COLLISION_FILTER_TABLE = 2 ** 0

_C.ENV.COLLISION_FILTER_PANDA = -1

_C.ENV.COLLISION_FILTER_YCB = [2 ** (i + 1) for i in range(21)]

_C.ENV.COLLISION_FILTER_MANO = 2 ** 22

_C.ENV.COLLISION_FILTER_YCB_RELEASE = -1 - 2 ** 22


_C.ENV.RELEASE_FORCE_THRESH = 0.0

_C.ENV.RELEASE_TIME_THRESH = 0.1

_C.ENV.RELEASE_CONTACT_REGION_RANGE_X = (-0.0110, +0.0110)

_C.ENV.RELEASE_CONTACT_REGION_RANGE_Y = (-0.0090, +0.0090)

_C.ENV.RELEASE_CONTACT_REGION_RANGE_Z = (+0.0000, +0.0550)

_C.ENV.DRAW_RELEASE_CONTACT = False

_C.ENV.RELEASE_CONTACT_REGION_COLOR = [0.85, 0.19, 0.21, 0.5]

_C.ENV.RELEASE_CONTACT_VERTEX_RADIUS = 0.001

_C.ENV.RELEASE_CONTACT_VERTEX_COLOR = [0.85, 0.19, 0.21, 1.0]


_C.ENV.ID = "HandoverStateEnv-v1"

# ---------------------------------------------------------------------------- #
# Benchmark config
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CN()


_C.BENCHMARK.SETUP = "s0"

_C.BENCHMARK.SPLIT = "test"


_C.BENCHMARK.GOAL_CENTER = (0.61, -0.20, 1.25)

_C.BENCHMARK.GOAL_RADIUS = 0.15

_C.BENCHMARK.DRAW_GOAL = False

_C.BENCHMARK.GOAL_COLOR = (0.85, 0.19, 0.21, 0.5)

_C.BENCHMARK.CONTACT_FORCE_THRESH = 0.0

_C.BENCHMARK.SUCCESS_TIME_THRESH = 0.1

_C.BENCHMARK.MAX_EPISODE_TIME = 13.0


_C.BENCHMARK.RESULT_DIR = "results"

_C.BENCHMARK.SAVE_RESULT = False

_C.BENCHMARK.SAVE_HEADLESS_RENDER = False


get_cfg = easysim.get_cfg

get_config_from_args = easysim.get_config_from_args
