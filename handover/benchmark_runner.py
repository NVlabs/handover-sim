import gym
import os
import functools
import time
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

    def run(self, policy):
        if self._cfg.BENCHMARK.SAVE_HEADLESS_RENDER and not self._cfg.BENCHMARK.SAVE_RESULT:
            raise ValueError(
                "SAVE_HEADLESS_RENDER can only be set to True when SAVE_RESULT is set to True"
            )

        if self._cfg.BENCHMARK.SAVE_RESULT:
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

        for idx in range(self._env.num_scenes):
            print(
                "{:04d}/{:04d}: scene {}".format(
                    idx + 1, self._env.num_scenes, self._env._scene_ids[idx]
                )
            )

            result, elapsed_time = self._run_scene(idx, policy)

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
    def _run_scene(self, idx, policy):
        obs = self._env.reset(idx=idx)
        policy.reset()

        result = {}
        result["action"] = []
        result["elapsed_time"] = []

        while True:
            action, elapsed_time = self._run_policy(policy, obs)

            result["action"].append(action)
            result["elapsed_time"].append(elapsed_time)

            obs, _, _, info = self._env.step(action)

            if info["status"] != 0:
                break

        result["action"] = np.array(result["action"])
        result["elapsed_time"] = np.array(result["elapsed_time"])
        result["elapsed_frame"] = self._env.frame
        result["result"] = info["status"]

        return result

    @timer
    def _run_policy(self, policy, obs):
        return policy.forward(obs)
