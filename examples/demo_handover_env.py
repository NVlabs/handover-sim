# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import gym
import numpy as np

from handover.config import get_config_from_args

scene_id = 105


def main():
    cfg = get_config_from_args()

    env = gym.make(cfg.ENV.ID, cfg=cfg)

    while True:
        env.reset(scene_id=scene_id)
        for _ in range(int(3.0 / cfg.SIM.TIME_STEP)):
            action = np.array(cfg.ENV.PANDA_INITIAL_POSITION, dtype=np.float32)
            action += np.random.uniform(low=-1.0, high=+1.0, size=len(action))
            env.step(action)


if __name__ == "__main__":
    main()
