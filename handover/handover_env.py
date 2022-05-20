# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import easysim
import abc
import numpy as np

from handover.table import Table
from handover.panda import Panda, PandaHandCamera
from handover.dex_ycb import DexYCB
from handover.ycb import YCB
from handover.mano import MANO
from handover.transform3d import get_t3d_from_qt


class HandoverEnv(easysim.SimulatorEnv):
    def init(self):
        self._table = Table(self.cfg, self.scene)

        self._panda = self._get_panda_cls()(self.cfg, self._scene)

        self._dex_ycb = DexYCB(self.cfg)
        self._ycb = YCB(self.cfg, self.scene, self._dex_ycb)
        self._mano = MANO(self.cfg, self.scene, self._dex_ycb)

        self._release_step_thresh = self.cfg.ENV.RELEASE_TIME_THRESH / self.cfg.SIM.TIME_STEP

    @abc.abstractmethod
    def _get_panda_cls(self):
        """ """

    @property
    def table(self):
        return self._table

    @property
    def panda(self):
        return self._panda

    @property
    def dex_ycb(self):
        return self._dex_ycb

    @property
    def ycb(self):
        return self._ycb

    @property
    def mano(self):
        return self._mano

    def pre_reset(self, env_ids, scene_id):
        self.ycb.reset(scene_id)
        self.mano.reset(scene_id)

        self._release_reset()
        if self.cfg.ENV.DRAW_RELEASE_CONTACT:
            self._release_draw_reset()

    def post_reset(self, env_ids, scene_id):
        self._frame = 0
        return self._get_observation()

    def pre_step(self, action):
        self.panda.step(action)
        self.ycb.step()
        self.mano.step()

    def post_step(self, action):
        if not self.ycb.released and self._release_check():
            self.ycb.release()

        if self.cfg.ENV.DRAW_RELEASE_CONTACT:
            self._release_draw_step()

        self._frame += 1

        observation = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()

        return observation, reward, done, info

    @property
    def frame(self):
        return self._frame

    @abc.abstractmethod
    def _get_observation(self):
        """ """

    @abc.abstractmethod
    def _get_reward(self):
        """ """

    @abc.abstractmethod
    def _get_done(self):
        """ """

    @abc.abstractmethod
    def _get_info(self):
        """ """

    def _release_reset(self):
        self._release_step_counter_passive = 0
        self._release_step_counter_active = 0

    def _release_check(self):
        contact = self.contact[0]

        contact = contact[contact["force"] > self.cfg.ENV.RELEASE_FORCE_THRESH]

        if len(contact) == 0:
            contact_panda_release_region = [False] * len(self.panda.LINK_IND_FINGERS)
            contact_panda_body = False
        else:
            contact_1 = contact[
                (contact["body_id_a"] == self.ycb.bodies[self.ycb.ids[0]].contact_id[0])
                & (contact["body_id_b"] == self.panda.body.contact_id[0])
            ]
            contact_2 = contact[
                (contact["body_id_a"] == self.panda.body.contact_id[0])
                & (contact["body_id_b"] == self.ycb.bodies[self.ycb.ids[0]].contact_id[0])
            ]
            contact_2[["body_id_a", "body_id_b"]] = contact_2[["body_id_b", "body_id_a"]]
            contact_2[["link_id_a", "link_id_b"]] = contact_2[["link_id_b", "link_id_a"]]
            contact_2[["position_a_world", "position_b_world"]] = contact_2[
                ["position_b_world", "position_a_world"]
            ]
            contact_2[["position_a_link", "position_b_link"]] = contact_2[
                ["position_b_link", "position_a_link"]
            ]
            contact_2["normal"]["x"] *= -1
            contact_2["normal"]["y"] *= -1
            contact_2["normal"]["z"] *= -1
            contact = np.concatenate((contact_1, contact_2))

            contact_panda_body = len(contact) > 0
            contact_panda_release_region = []

            for link_index in self.panda.LINK_IND_FINGERS:
                contact_link = contact[contact["link_id_b"] == link_index]

                if len(contact_link) == 0:
                    contact_panda_release_region.append(False)
                else:
                    if np.any(np.isnan(contact_link["position_b_link"]["x"])):
                        pos = self.panda.body.link_state[0, link_index, 0:3]
                        orn = self.panda.body.link_state[0, link_index, 3:7]
                        t3d = get_t3d_from_qt(orn, pos)
                        t3d = t3d.inverse()
                        position = (
                            np.ascontiguousarray(contact_link["position_b_world"])
                            .view(np.float32)
                            .reshape(-1, 3)
                        )
                        position = t3d.transform_points(position)
                    else:
                        position = (
                            np.ascontiguousarray(contact_link["position_b_link"])
                            .view(np.float32)
                            .reshape(-1, 3)
                        )

                    is_in_release_region = (
                        (position[:, 0] > self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_X[0])
                        & (position[:, 0] < self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_X[1])
                        & (position[:, 1] > self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Y[0])
                        & (position[:, 1] < self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Y[1])
                        & (position[:, 2] > self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Z[0])
                        & (position[:, 2] < self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Z[1])
                    )
                    contact_panda_release_region.append(np.any(is_in_release_region))

        if not any(contact_panda_release_region) and contact_panda_body:
            self._release_step_counter_passive += 1
        else:
            if self._release_step_counter_passive != 0:
                self._release_step_counter_passive = 0

        if all(contact_panda_release_region):
            self._release_step_counter_active += 1
        else:
            if self._release_step_counter_active != 0:
                self._release_step_counter_active = 0

        return (
            self._release_step_counter_passive >= self._release_step_thresh
            or self._release_step_counter_active >= self._release_step_thresh
        )

    def _release_draw_reset(self):
        raise NotImplementedError

    def _release_draw_step(self):
        raise NotImplementedError


class HandoverStateEnv(HandoverEnv):
    def _get_panda_cls(self):
        return Panda

    def _get_observation(self):
        observation = {}
        observation["frame"] = self.frame
        observation["panda_link_ind_hand"] = self.panda.LINK_IND_HAND
        observation["panda_body"] = self.panda.body
        observation["ycb_classes"] = self.ycb.CLASSES
        observation["ycb_bodies"] = self.ycb.bodies
        observation["mano_body"] = self.mano.body
        return observation

    def _get_reward(self):
        return None

    def _get_done(self):
        return False

    def _get_info(self):
        return {}


import pybullet

from scipy.spatial.transform import Rotation as Rot


class HandoverHandCameraPointStateEnv(HandoverEnv):
    def init(self):
        super().init()

        # Create projection matrix.
        self._width = 224
        self._height = 224
        self._fov = 90.0
        self._aspect = self._width / self._height
        self._near = 0.035
        self._far = 2.0
        self._projection_matrix = pybullet.computeProjectionMatrixFOV(
            self._fov, self._aspect, self._near, self._far
        )

        # Get deproject points before depth multiplication.
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = self._width * self._projection_matrix[0] / 2
        K[1, 1] = self._height * self._projection_matrix[5] / 2
        K[0, 2] = self._width / 2
        K[1, 2] = self._height / 2
        K_inv = np.linalg.inv(K)
        x, y = np.meshgrid(np.arange(self._width), np.arange(self._height))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        ones = np.ones((self._height, self._width), dtype=np.float32)
        xy1s = np.stack((x, y, ones), axis=2).reshape(self._width * self._height, 3).T
        self._deproject_p = np.matmul(K_inv, xy1s).T

        # Get transform from hand to pinhole camera frame.
        pos = (+0.036, 0.0, +0.036)
        orn = Rot.from_euler("XYZ", (0.0, 0.0, +np.pi / 2)).as_quat().astype(np.float32)
        self._t3d_hand_to_camera = get_t3d_from_qt(orn, pos)

    def _get_panda_cls(self):
        return PandaHandCamera

    def post_reset(self, env_ids, scene_id):
        self._point_state = None

        return super().post_reset(env_ids, scene_id)

    def post_step(self, action):
        self._point_state = None

        return super().post_step(action)

    def _get_observation(self):
        observation = {}
        observation["frame"] = self.frame
        observation["panda_link_ind_hand"] = self.panda.LINK_IND_HAND
        observation["panda_body"] = self.panda.body
        observation["callback_get_point_state"] = lambda: self._get_point_state()
        return observation

    def _get_reward(self):
        return None

    def _get_done(self):
        return False

    def _get_info(self):
        return {}

    def _get_point_state(self):
        if self._point_state is not None:
            return self._point_state

        # Get OpenGL view frame from URDF camera frame.
        pos = self.panda.body.link_state[0, self.panda.LINK_IND_CAMERA, 0:3]
        orn = self.panda.body.link_state[0, self.panda.LINK_IND_CAMERA, 3:7]
        R = np.matmul(
            Rot.from_quat(orn).as_matrix().astype(np.float32),
            Rot.from_euler("XYZ", (-np.pi / 2, 0.0, -np.pi)).as_matrix().astype(np.float32),
        )

        # Create view matrix.
        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[:3, :3] = R.T
        view_matrix[:3, 3] = -1 * np.matmul(R.T, pos)
        view_matrix = view_matrix.T

        # Render camera image.
        _, _, _, depth, mask = pybullet.getCameraImage(
            self._width,
            self._height,
            viewMatrix=tuple(view_matrix.flatten()),
            projectionMatrix=self._projection_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )

        # Convert depth buffer to real depth.
        depth = self._far * self._near / (self._far - (self._far - self._near) * depth)

        # Get point state in pinhole camera frame.
        mask = mask == self.ycb.bodies[self.ycb.ids[0]].contact_id[0]
        point_state = (
            np.tile(depth[mask].reshape(-1, 1), (1, 3)) * self._deproject_p[mask.ravel(), :]
        )

        # Transform point state to hand frame.
        point_state = self._t3d_hand_to_camera.transform_points(point_state)

        self._point_state = point_state

        return self._point_state
