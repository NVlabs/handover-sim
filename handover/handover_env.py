import easysim
import numpy as np

from handover.dex_ycb import DexYCB
from handover.table import Table
from handover.panda import Panda
from handover.ycb import YCB
from handover.mano import MANO
from handover.transform3d import get_t3d_from_qt


class HandoverEnv(easysim.SimulatorEnv):
    def init(self):
        self._dex_ycb = DexYCB(self.cfg)
        self._table = Table(self.cfg, self.scene)
        self._panda = Panda(self.cfg, self.scene)
        self._ycb = YCB(self.cfg, self.scene, self._dex_ycb)
        self._mano = MANO(self.cfg, self.scene, self._dex_ycb)

        self._table.add()
        self._panda.add()

        self._cur_scene_id = None
        self._release_step_thresh = self.cfg.ENV.RELEASE_TIME_THRESH / self.cfg.SIM.TIME_STEP

    def pre_reset(self, env_ids, scene_id):
        if scene_id != self._cur_scene_id:
            self._ycb.clean()
            self._mano.clean()
            self._cur_scene_id = scene_id

        self._ycb.reset(scene_id)
        self._mano.reset(scene_id)

        self._release_reset()
        if self.cfg.ENV.IS_DRAW_RELEASE:
            self._release_draw_reset()

    def post_reset(self, env_ids, scene_id):
        return None

    def pre_step(self, action):
        self._panda.step(action)
        self._ycb.step()
        self._mano.step()

    def post_step(self, action):
        if not self._ycb.released and self._release_check():
            self._ycb.release()

        if self.cfg.ENV.IS_DRAW_RELEASE:
            self._release_draw_step()

        return None, None, False, {}

    def _release_reset(self):
        self._release_step_counter_passive = 0
        self._release_step_counter_active = 0

    def _release_check(self):
        contact = self.contact[0]

        contact = contact[contact["force"] > self.cfg.ENV.RELEASE_FORCE_THRESH]

        if len(contact) == 0:
            contact_panda_release_region = [False] * len(self._panda.LINK_IND_FINGERS)
            contact_panda_body = False
        else:
            contact_1 = contact[
                (contact["body_id_a"] == self._ycb.grasped_body.contact_id)
                & (contact["body_id_b"] == self._panda.body.contact_id)
            ]
            contact_2 = contact[
                (contact["body_id_a"] == self._panda.body.contact_id)
                & (contact["body_id_b"] == self._ycb.grasped_body.contact_id)
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

            for link_index in self._panda.LINK_IND_FINGERS:
                contact_link = contact[contact["link_id_b"] == link_index]

                if len(contact_link) == 0:
                    contact_panda_release_region.append(False)
                else:
                    if np.any(np.isnan(contact_link["position_b_link"]["x"])):
                        pos = self._panda.body.link_state[0][link_index, 0:3]
                        orn = self._panda.body.link_state[0][link_index, 3:7]
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
                        (position[:, 0] > self.cfg.ENV.RELEASE_FINGER_CONTACT_X_RANGE[0])
                        & (position[:, 0] < self.cfg.ENV.RELEASE_FINGER_CONTACT_X_RANGE[1])
                        & (position[:, 1] > self.cfg.ENV.RELEASE_FINGER_CONTACT_Y_RANGE[0])
                        & (position[:, 1] < self.cfg.ENV.RELEASE_FINGER_CONTACT_Y_RANGE[1])
                        & (position[:, 2] > self.cfg.ENV.RELEASE_FINGER_CONTACT_Z_RANGE[0])
                        & (position[:, 2] < self.cfg.ENV.RELEASE_FINGER_CONTACT_Z_RANGE[1])
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
