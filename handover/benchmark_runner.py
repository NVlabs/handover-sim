# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import gym
import os
import functools
import time
import cv2
import numpy as np

from datetime import datetime

from handover.benchmark_wrapper import EpisodeStatus, HandoverBenchmarkWrapper


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        return value, elapsed_time

    return wrapper_timer


class BenchmarkRunner:
    def __init__(self, cfg):
        self._cfg = cfg

        self._env = HandoverBenchmarkWrapper(gym.make(self._cfg.ENV.ID, cfg=self._cfg))

    def run(self, policy, res_dir=None, index=None):
        if self._cfg.BENCHMARK.SAVE_OFFSCREEN_RENDER:
            if not self._cfg.ENV.RENDER_OFFSCREEN:
                raise ValueError(
                    "SAVE_OFFSCREEN_RENDER can only be True when RENDER_OFFSCREEN is set to True"
                )
            if not self._cfg.BENCHMARK.SAVE_RESULT and res_dir is None:
                raise ValueError(
                    "SAVE_OFFSCREEN_RENDER can only be True when SAVE_RESULT is set to True or "
                    "`res_dir` is not None"
                )
            if 1.0 / self._cfg.BENCHMARK.OFFSCREEN_RENDER_FRAME_RATE < self._cfg.SIM.TIME_STEP:
                raise ValueError("Offscreen render time step must not be smaller than TIME_STEP")

            self._render_steps = (
                1.0 / self._cfg.BENCHMARK.OFFSCREEN_RENDER_FRAME_RATE / self._cfg.SIM.TIME_STEP
            )

        if self._cfg.BENCHMARK.SAVE_RESULT:
            if res_dir is not None:
                raise ValueError("SAVE_RESULT can only be True when `res_dir` is None")

            dt = datetime.now()
            dt = dt.strftime("%Y-%m-%d_%H-%M-%S")
            res_dir = os.path.join(
                self._cfg.BENCHMARK.RESULT_DIR,
                "{}_{}_{}_{}".format(
                    dt, policy.name, self._cfg.BENCHMARK.SETUP, self._cfg.BENCHMARK.SPLIT
                ),
            )
            os.makedirs(res_dir, exist_ok=True)

            cfg_file = os.path.join(res_dir, "config.yaml")
            with open(cfg_file, "w") as f:
                self._cfg.dump(stream=f, default_flow_style=None)

        if index is None:
            indices = range(self._env.num_scenes)
        else:
            indices = [index]

        for idx in indices:
            print(
                "{:04d}/{:04d}: scene {}".format(
                    idx + 1, self._env.num_scenes, self._env._scene_ids[idx]
                )
            )

            kwargs = {}
            if self._cfg.BENCHMARK.SAVE_OFFSCREEN_RENDER:
                kwargs["render_dir"] = os.path.join(res_dir, "{:03d}".format(idx))
                os.makedirs(kwargs["render_dir"], exist_ok=True)

            result, elapsed_time = self._run_scene(idx, policy, **kwargs)

            print("time:   {:6.2f}".format(elapsed_time))
            print("frame:  {:5d}".format(result["elapsed_frame"]))
            if result["result"] == EpisodeStatus.SUCCESS:
                print("result:  success")
            else:
                failure_1 = (
                    result["result"] & EpisodeStatus.FAILURE_HUMAN_CONTACT
                    == EpisodeStatus.FAILURE_HUMAN_CONTACT
                )
                failure_2 = (
                    result["result"] & EpisodeStatus.FAILURE_OBJECT_DROP
                    == EpisodeStatus.FAILURE_OBJECT_DROP
                )
                failure_3 = (
                    result["result"] & EpisodeStatus.FAILURE_TIMEOUT
                    == EpisodeStatus.FAILURE_TIMEOUT
                )
                print("result:  failure {:d} {:d} {:d}".format(failure_1, failure_2, failure_3))

            if self._cfg.BENCHMARK.SAVE_RESULT:
                res_file = os.path.join(res_dir, "{:03d}.npz".format(idx))
                np.savez_compressed(res_file, **result)

    @timer
    def _run_scene(self, idx, policy, render_dir=None):
        obs = self._env.reset(idx=idx)
        policy.reset()

        result = {}
        result["action"] = []
        result["elapsed_time"] = []

        if self._cfg.BENCHMARK.SAVE_OFFSCREEN_RENDER:
            self._render_offscreen_and_save(render_dir)

        while True:
            action, elapsed_time = self._run_policy(policy, obs)

            result["action"].append(action)
            result["elapsed_time"].append(elapsed_time)

            obs, _, _, info = self._env.step(action)

            if (
                self._cfg.BENCHMARK.SAVE_OFFSCREEN_RENDER
                and (self._env.frame % self._render_steps)
                <= (self._env.frame - 1) % self._render_steps
            ):
                self._render_offscreen_and_save(render_dir)

            if info["status"] != 0:
                break

        result["action"] = np.array(result["action"])
        result["elapsed_time"] = np.array(result["elapsed_time"])
        result["elapsed_frame"] = self._env.frame
        result["result"] = info["status"]

        return result

    def _render_offscreen_and_save(self, render_dir):
        render_file = os.path.join(render_dir, "{:06d}.jpg".format(self._env.frame))
        cv2.imwrite(render_file, self._env.render_offscreen()[:, :, [2, 1, 0, 3]])

    @timer
    def _run_policy(self, policy, obs):
        return policy.forward(obs)
