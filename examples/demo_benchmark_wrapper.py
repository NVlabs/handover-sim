# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import abc
import gym
import numpy as np
import pybullet

from handover.config import get_config_from_args
from handover.benchmark_wrapper import HandoverBenchmarkWrapper

from demo_trajectory import (
    start_conf,
    traj,
    time_wait,
    time_action_repeat,
    time_close_gripper,
    back_step_size,
)

setup = "s0"
split = "train"

scene_id = 105


class SimplePolicy(abc.ABC):
    def __init__(
        self,
        cfg,
        start_conf=start_conf,
        time_wait=time_wait,
        time_action_repeat=time_action_repeat,
        time_close_gripper=time_close_gripper,
        back_step_size=back_step_size,
    ):
        self._cfg = cfg
        self._start_conf = start_conf
        self._steps_wait = int(time_wait / self._cfg.SIM.TIME_STEP)
        self._steps_action_repeat = int(time_action_repeat / self._cfg.SIM.TIME_STEP)
        self._steps_close_gripper = int(time_close_gripper / self._cfg.SIM.TIME_STEP)
        self._back_step_size = back_step_size

    @property
    @abc.abstractmethod
    def name(self):
        """ """

    def reset(self):
        self._done = False
        self._done_frame = None
        self._done_action = None
        self._back = None

    def forward(self, obs):
        info = {}

        if obs["frame"] < self._steps_wait:
            # Wait.
            action = self._start_conf.copy()
        elif not self._done:
            # Approach object until reaching grasp pose.
            action, done, info = self.plan(obs)
            if done:
                self._done = True
                self._done_frame = obs["frame"] + 1
                self._done_action = action.copy()
        else:
            # Close gripper and back out.
            if obs["frame"] < self._done_frame + self._steps_close_gripper:
                action = self._done_action.copy()
                action[7:9] = 0.0
            else:
                if self._back is None:
                    self._back = []
                    pos = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3].numpy()
                    dpos_goal = self._cfg.BENCHMARK.GOAL_CENTER - pos
                    dpos_step = dpos_goal / np.linalg.norm(dpos_goal) * self._back_step_size
                    num_steps = int(np.ceil(np.linalg.norm(dpos_goal) / self._back_step_size))
                    for _ in range(num_steps):
                        pos += dpos_step
                        conf = pybullet.calculateInverseKinematics(
                            obs["panda_body"].contact_id[0], obs["panda_link_ind_hand"] - 1, pos
                        )
                        conf = np.array(conf)
                        conf[7:9] = 0.0
                        self._back.append(conf)

                i = (
                    obs["frame"] - self._done_frame - self._steps_close_gripper
                ) // self._steps_action_repeat
                i = min(i, len(self._back) - 1)
                action = self._back[i].copy()

        return action, info

    @abc.abstractmethod
    def plan(self, obs):
        """ """


class DemoPolicy(SimplePolicy):
    @property
    def name(self):
        return "demo"

    def plan(self, obs):
        i = (obs["frame"] - self._steps_wait) // self._steps_action_repeat
        action = traj[i].copy()
        done = obs["frame"] == self._steps_wait + len(traj) * self._steps_action_repeat - 1
        return action, done, {}


def main():
    cfg = get_config_from_args()

    cfg.BENCHMARK.SETUP = setup
    cfg.BENCHMARK.SPLIT = split

    env = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))

    policy = DemoPolicy(cfg)

    while True:
        obs = env.reset(scene_id=scene_id)
        policy.reset()

        while True:
            action, _ = policy.forward(obs)
            obs, _, _, info = env.step(action)

            if info["status"] != 0:
                print("Done. Status: {:d}".format(info["status"]))
                break


if __name__ == "__main__":
    main()
