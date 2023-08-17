# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import os
import sys
import pybullet
import random
import numpy as np

from handover.config import get_config_from_args
from handover.benchmark_runner import BenchmarkRunner

from demo_benchmark_wrapper import SimplePolicy, start_conf

assert "OMG_PLANNER_DIR" in os.environ, "Environment variable 'OMG_PLANNER_DIR' is not set"
sys.path.append(os.environ["OMG_PLANNER_DIR"])

from omg.config import cfg
from omg.core import PlanningScene


class OMGPlanner:
    def __init__(self):
        cfg.ik_parallel = False
        cfg.vis = False
        cfg.scene_file = ""
        cfg.cam_V = None

        # Enforce determinism. This accounts for the call of random.sample() in
        # `Robot.load_collision_points()` in `OMG-Planner/omg/core.py`.
        random.seed(0)

        self._scene = PlanningScene(None)

    def reset_scene(self, names, poses):
        for name in list(self._scene.env.names):
            self._scene.env.remove_object(name)
        assert len(self._scene.env.objects) == 0

        for name, pose in zip(names, poses):
            self._scene.env.add_object(name, pose[:3], pose[3:], compute_grasp=False)
        self._scene.env.combine_sdfs()

    def plan_to_target(self, start_conf, target_name):
        # Enfore determinism. This accounts for the call of `np.random.choice()` in
        # `Planner.setup_goal_set()` in `OMG-Planner/omg/planner.py`.
        np.random.seed(0)

        self._scene.traj.start = start_conf
        self._scene.env.set_target(target_name)

        if not hasattr(self._scene, "planner"):
            self._scene.reset()
        else:
            self._scene.update_planner()

        info = self._scene.step()
        traj = self._scene.planner.history_trajectories[-1]

        if len(info) == 0:
            traj = None

        return traj, info


class OMGPlannerPolicy(SimplePolicy):
    def __init__(self, cfg):
        super().__init__(cfg, time_close_gripper=0.5)

        self._panda_base_invert_transform = pybullet.invertTransform(
            self._cfg.ENV.PANDA_BASE_POSITION, self._cfg.ENV.PANDA_BASE_ORIENTATION
        )

        self._omg_planner = OMGPlanner()

    @property
    def name(self):
        return "omg-planner"

    def reset(self):
        super().reset()

        self._traj = None

    def plan(self, obs):
        if self._traj is None:
            traj, _ = self._run_omg_planner(obs)
            if traj is None:
                print("Planning not run due to empty goal set. Stay in start conf.")
                self._traj = []
            else:
                assert len(traj) > 0
                self._traj = traj

        if len(self._traj) == 0:
            action = start_conf.copy()
            done = False
        else:
            i = (obs["frame"] - self._steps_wait) // self._steps_action_repeat
            action = self._traj[i].copy()
            done = (
                obs["frame"] == self._steps_wait + len(self._traj) * self._steps_action_repeat - 1
            )

        return action, done, {}

    def _run_omg_planner(self, obs):
        poses = []
        for i in obs["ycb_bodies"]:
            pos = obs["ycb_bodies"][i].link_state[0, 6, 0:3]
            orn = obs["ycb_bodies"][i].link_state[0, 6, 3:7]
            pos, orn = pybullet.multiplyTransforms(*self._panda_base_invert_transform, pos, orn)
            poses += [orn + pos]
        names = [obs["ycb_classes"][i] for i in obs["ycb_bodies"]]
        poses = [p[4:] + (p[3], p[0], p[1], p[2]) for p in poses]
        self._omg_planner.reset_scene(names, poses)

        target_name = names[0]
        traj, info = self._omg_planner.plan_to_target(start_conf, target_name)

        return traj, info


def main():
    cfg = get_config_from_args()

    policy = OMGPlannerPolicy(cfg)

    benchmark_runner = BenchmarkRunner(cfg)
    benchmark_runner.run(policy)


if __name__ == "__main__":
    main()
