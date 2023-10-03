# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import argparse
import os
import numpy as np

from handover.config import get_cfg
from handover.benchmark_runner import BenchmarkRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Render benchmark.")
    parser.add_argument("--res_dir", help="result directory produced by benchmark runner")
    parser.add_argument("--index", type=int, help="index of the scene")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help=(
            """modify config options at the end of the command; use space-separated """
            """"PATH.KEY VALUE" pairs; only certain options are allowed to be changed---see """
            """main()"""
        ),
    )
    args = parser.parse_args()
    return args


class ResultLoaderPolicy:
    def __init__(self, res_dir, index=None):
        self._res_dir = res_dir

        if index is None:
            self._idx = -1
        else:
            self._idx = index - 1

    def reset(self):
        self._idx += 1

        res_file = os.path.join(self._res_dir, "{:03d}.npz".format(self._idx))
        self._action = np.load(res_file)["action"]

    def forward(self, obs):
        return self._action[obs["frame"]], {}


def main():
    args = parse_args()

    cfg_file = os.path.join(args.res_dir, "config.yaml")
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)

    cfg.BENCHMARK.SAVE_RESULT = False

    for full_key in args.opts[0::2]:
        assert full_key in (
            "SIM.RENDER",
            "SIM.BULLET.USE_EGL",
            "ENV.RENDER_OFFSCREEN",
            "BENCHMARK.SAVE_OFFSCREEN_RENDER",
            "BENCHMARK.OFFSCREEN_RENDER_FRAME_RATE",
        ) or full_key.startswith(
            "ENV.OFFSCREEN_RENDERER_CAMERA_"
        ), "Key not allowed to be changed: {}".format(
            full_key
        )
    cfg.merge_from_list(args.opts)

    policy = ResultLoaderPolicy(args.res_dir, index=args.index)

    benchmark_runner = BenchmarkRunner(cfg)
    benchmark_runner.run(policy, res_dir=args.res_dir, index=args.index)


if __name__ == "__main__":
    main()
