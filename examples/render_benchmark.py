# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import argparse
import os
import numpy as np
import pybullet
import types

from contextlib import contextmanager

from handover.config import get_cfg
from handover.benchmark_runner import BenchmarkRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Render benchmark.")
    parser.add_argument("--res_dir", help="result directory produced by benchmark runner")
    parser.add_argument(
        "--bullet_disable_cov_rendering",
        action="store_true",
        help="disable cov rendering for Bullet",
    )
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
    def __init__(self, res_dir):
        self._res_dir = res_dir

        self._idx = -1

    def reset(self):
        self._idx += 1

        res_file = os.path.join(self._res_dir, "{:03d}.npz".format(self._idx))
        self._result = np.load(res_file)

    def forward(self, obs):
        return self._result["action"][obs["frame"]]


def bullet_disable_cov_rendering(cfg, env):
    if not cfg.SIM.RENDER:
        raise ValueError(
            "--bullet_disable_cov_rendering can only be used when RENDER is set to True"
        )

    @contextmanager
    def _disable_cov_rendering(self):
        try:
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
            yield
        finally:
            pass

    env.simulator._disable_cov_rendering = types.MethodType(_disable_cov_rendering, env.simulator)


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

    policy = ResultLoaderPolicy(args.res_dir)

    benchmark_runner = BenchmarkRunner(cfg)

    if args.bullet_disable_cov_rendering:
        bullet_disable_cov_rendering(cfg, benchmark_runner.env)

    benchmark_runner.run(policy, args.res_dir)


if __name__ == "__main__":
    main()
