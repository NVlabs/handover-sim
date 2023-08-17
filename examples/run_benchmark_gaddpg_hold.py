# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import os
import sys
import numpy as np
import pybullet

from scipy.spatial.transform import Rotation as Rot

from handover.benchmark_runner import timer, BenchmarkRunner
from handover.config import get_config_from_args

from demo_benchmark_wrapper import SimplePolicy, time_wait

assert "GADDPG_DIR" in os.environ, "Environment variable 'GADDPG_DIR' is not set"
sys.path.append(os.environ["GADDPG_DIR"])

from experiments.config import cfg_from_file, cfg
from env.panda_scene import PandaTaskSpace6D
from core.ddpg import DDPG
from core.utils import (
    make_nets_opts_schedulers,
    unpack_pose,
    tf_quat,
    se3_inverse,
    se3_transform_pc,
    regularize_pc_point_count,
    hand_finger_point,
    unpack_action,
)

seed = 123456
pretrained = os.path.join(os.environ["GADDPG_DIR"], "output", "demo_model")


class PointListener:
    def __init__(self, pretrained=pretrained, seed=seed):
        self._pretrained = pretrained
        self._seed = seed

        cfg_from_file(os.path.join(self._pretrained, "config.yaml"), reset_model_spec=False)
        cfg.RL_MODEL_SPEC = os.path.join(self._pretrained, cfg.RL_MODEL_SPEC.split("/")[-1])

        input_dim = cfg.RL_TRAIN.feature_input_dim
        action_space = PandaTaskSpace6D()
        net_dict = make_nets_opts_schedulers(cfg.RL_MODEL_SPEC, cfg.RL_TRAIN)

        self._agent = DDPG(input_dim, action_space, cfg.RL_TRAIN)
        self._agent.setup_feature_extractor(net_dict)
        self._agent.load_model(self._pretrained)

        # Warm up network. The first pass will be slower than later ones so we want to exclude it
        # from benchmark time.
        _, warm_up_time = self._warm_up_network()
        print("warn up time: {:6.2f}".format(warm_up_time))

        self._cage_point_threshold = 50

    @timer
    def _warm_up_network(self):
        state = [(np.zeros((4, 1)), np.array([])), None, None, None]
        self._agent.select_action(state)

    def reset(self):
        self._remaining_step = cfg.RL_MAX_STEP
        self._acc_points = np.zeros((3, 0))
        self._acc_mean = np.zeros((3, 1))

        # Enforce determinism.
        np.random.seed(self._seed)

    @property
    def remaining_step(self):
        return self._remaining_step

    @property
    def acc_points(self):
        return self._acc_points

    def run_network(self, point_state, ef_pose):
        state = self._point_state_to_state(point_state, ef_pose)
        state = [state, None, None, None]
        action, _, _, _ = self._agent.select_action(state, remain_timestep=self._remaining_step)

        ef_pose_delta = unpack_action(action)
        ef_pose_new = np.matmul(ef_pose, ef_pose_delta)

        self._remaining_step = max(self._remaining_step - 1, 0)

        return ef_pose_new

    def _point_state_to_state(self, point_state, ef_pose):
        point_state = self._process_pointcloud(point_state, ef_pose)
        image_state = np.array([])
        obs = (point_state, image_state)
        return obs

    def _process_pointcloud(self, point_state, ef_pose):
        if point_state.shape[1] > 0:
            self._acc_points = np.zeros((3, 0))
            self._update_curr_acc_points(point_state, ef_pose)

        inv_ef_pose = se3_inverse(ef_pose)
        point_state = se3_transform_pc(inv_ef_pose, self._acc_points)

        point_state = regularize_pc_point_count(point_state.T, cfg.RL_TRAIN.uniform_num_pts).T

        point_state = np.concatenate((hand_finger_point, point_state), axis=1)
        point_state_ = np.zeros((4, point_state.shape[1]))
        point_state_[:3] = point_state
        point_state_[3, : hand_finger_point.shape[1]] = 1
        point_state = point_state_

        return point_state

    def _update_curr_acc_points(self, new_points, ef_pose):
        new_points = se3_transform_pc(ef_pose, new_points)

        step = cfg.RL_MAX_STEP - self._remaining_step
        aggr_sample_point_num = min(
            int(cfg.RL_TRAIN.pt_accumulate_ratio ** step * cfg.RL_TRAIN.uniform_num_pts),
            new_points.shape[1],
        )
        index = np.random.choice(
            range(new_points.shape[1]), size=aggr_sample_point_num, replace=False
        )
        new_points = new_points[:, index]

        self._acc_points = np.concatenate((new_points, self._acc_points), axis=1)

        acc_mean = np.mean(self._acc_points, axis=1)
        acc_diff = np.linalg.norm(self._acc_mean - acc_mean)
        if acc_diff > 0.1 and self._remaining_step < cfg.RL_MAX_STEP:
            self._remaining_step += 5
            self._remaining_step = min(self._remaining_step, cfg.RL_MAX_STEP)
        self._acc_mean = acc_mean

    def termination_heuristics(self, ef_pose):
        inv_ef_pose = se3_inverse(ef_pose)
        point_state = se3_transform_pc(inv_ef_pose, self._acc_points)
        cage_points_mask = (
            (point_state[2] > +0.06)
            & (point_state[2] < +0.11)
            & (point_state[1] > -0.05)
            & (point_state[1] < +0.05)
            & (point_state[0] > -0.02)
            & (point_state[0] < +0.02)
        )
        cage_points_mask_reg = regularize_pc_point_count(
            cage_points_mask[:, None], cfg.RL_TRAIN.uniform_num_pts
        )
        return np.sum(cage_points_mask_reg) > self._cage_point_threshold


class GADDPGPolicy(SimplePolicy):
    def __init__(self, cfg, time_wait=time_wait):
        super().__init__(
            cfg,
            start_conf=np.array(cfg.ENV.PANDA_INITIAL_POSITION),
            time_wait=time_wait,
            time_action_repeat=0.15,
            time_close_gripper=0.5,
        )

        self._panda_base_invert_transform = pybullet.invertTransform(
            self._cfg.ENV.PANDA_BASE_POSITION, self._cfg.ENV.PANDA_BASE_ORIENTATION
        )

        self._point_listener = PointListener()

    def reset(self):
        super().reset()

        self._traj = []
        self._point_listener.reset()

    def plan(self, obs):
        info = {}

        if (obs["frame"] - self._steps_wait) % self._steps_action_repeat == 0:
            point_state, obs_time = self._get_point_state_from_callback(obs)
            info["obs_time"] = obs_time

            if point_state.shape[1] == 0 and self._point_listener.acc_points.shape[1] == 0:
                action = np.array(self._cfg.ENV.PANDA_INITIAL_POSITION)
            else:
                ef_pose = self._get_ef_pose(obs)
                ef_pose_new = self._point_listener.run_network(point_state, ef_pose)

                pos = ef_pose_new[:3, 3]
                orn = Rot.from_matrix(ef_pose_new[:3, :3]).as_quat()
                pos, orn = pybullet.multiplyTransforms(
                    self._cfg.ENV.PANDA_BASE_POSITION,
                    self._cfg.ENV.PANDA_BASE_ORIENTATION,
                    pos,
                    orn,
                )
                action = pybullet.calculateInverseKinematics(
                    obs["panda_body"].contact_id[0], obs["panda_link_ind_hand"] - 1, pos, orn
                )
                action = np.array(action)
                action[7:9] = 0.04

            self._traj.append(action)
        else:
            action = self._traj[-1].copy()

        if (obs["frame"] - self._steps_wait + 1) % self._steps_action_repeat == 0:
            ef_pose = self._get_ef_pose(obs)
            done = (
                self._point_listener.remaining_step == 0
                or self._point_listener._acc_points.shape[1] > 0
                and self._point_listener.termination_heuristics(ef_pose)
            )
        else:
            done = False

        return action, done, info

    @timer
    def _get_point_state_from_callback(self, obs):
        point_state = obs["callback_get_point_state"]()
        return point_state.T

    def _get_ef_pose(self, obs):
        pos = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3]
        orn = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 3:7]
        pos, orn = pybullet.multiplyTransforms(*self._panda_base_invert_transform, pos, orn)
        return unpack_pose(np.hstack((pos, tf_quat(orn))))


class GADDPGHoldPolicy(GADDPGPolicy):
    @property
    def name(self):
        return "ga-ddpg-hold"


def main():
    cfg = get_config_from_args()

    policy = GADDPGHoldPolicy(cfg)

    benchmark_runner = BenchmarkRunner(cfg)
    benchmark_runner.run(policy)


if __name__ == "__main__":
    main()
