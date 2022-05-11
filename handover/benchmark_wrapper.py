import easysim
import numpy as np


class HandoverStatusWrapper(easysim.SimulatorWrapper):
    _FAILURE_ROBOT_HUMAN_CONTACT = -1
    _FAILURE_OBJECT_DROP = -2
    _FAILURE_TIMEOUT = -4

    def __init__(self, env):
        super().__init__(env)

        self._success_step_thresh = self.cfg.BENCHMARK.SUCCESS_TIME_THRESH / self.cfg.SIM.TIME_STEP
        self._max_episode_steps = self.cfg.BENCHMARK.MAX_EPISODE_TIME / self.cfg.SIM.TIME_STEP

    def reset(self, env_ids=None, **kwargs):
        observation = super().reset(env_ids=env_ids, **kwargs)

        if self.cfg.BENCHMARK.IS_DRAW_GOAL:
            raise NotImplementedError

        self._elapsed_steps = 0

        self._dropped = False
        self._success_step_counter = 0

        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)

        self._elapsed_steps += 1

        status = self._check_status()

        if self._elapsed_steps >= self._max_episode_steps and status != 1:
            status += self._FAILURE_TIMEOUT

        done |= status != 0

        info["status"] = status

        return observation, reward, done, info

    def _check_status(self):
        status = 0

        if self.mano.body is not None:
            contact = self.contact[0]

            contact_1 = contact[
                (contact["body_id_a"] == self.mano.body.contact_id[0])
                & (contact["body_id_b"] == self.panda.body.contact_id[0])
            ]
            contact_2 = contact[
                (contact["body_id_a"] == self.panda.body.contact_id[0])
                & (contact["body_id_b"] == self.mano.body.contact_id[0])
            ]
            contact = np.concatenate((contact_1, contact_2))

            for x in contact:
                if x["force"] > self.cfg.BENCHMARK.CONTACT_FORCE_THRESH:
                    status += self._FAILURE_ROBOT_HUMAN_CONTACT
                    break

        if not self.ycb.released:
            return status

        if not self._dropped:
            contact = self.contact[0]

            contact_1 = contact[
                contact["body_id_a"] == self.ycb.bodies[self.ycb.ids[0]].contact_id[0]
            ]
            contact_2 = contact[
                contact["body_id_b"] == self.ycb.bodies[self.ycb.ids[0]].contact_id[0]
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

            contact_panda = contact[contact["body_id_b"] == self.panda.body.contact_id[0]]
            contact_table = contact[contact["body_id_b"] == self.table.body.contact_id[0]]
            contact_ycb_other = contact[
                np.any(
                    [
                        contact["body_id_b"] == self.ycb.bodies[i].contact_id[0]
                        for i in self.ycb.ids[1:]
                    ],
                    axis=0,
                )
            ]

            panda_link_ind = contact_panda["link_id_b"][
                contact_panda["force"] > self.cfg.BENCHMARK.CONTACT_FORCE_THRESH
            ]
            contact_panda_fingers = set(self.panda.LINK_IND_FINGERS).issubset(panda_link_ind)
            contact_table = np.any(contact_table["force"] > self.cfg.BENCHMARK.CONTACT_FORCE_THRESH)
            contact_ycb_other = np.any(
                contact_ycb_other["force"] > self.cfg.BENCHMARK.CONTACT_FORCE_THRESH
            )

            is_below_table = (
                self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 2].item()
                < self.cfg.ENV.TABLE_HEIGHT
            )

            if not contact_panda_fingers and (contact_table or contact_ycb_other or is_below_table):
                self._dropped = True

        if self._dropped:
            status += self._FAILURE_OBJECT_DROP

        if status < 0:
            return status

        if not contact_panda_fingers:
            if self._success_step_counter != 0:
                self._success_step_counter = 0
            return 0

        pos = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 0:3].numpy()
        dist = np.linalg.norm(self.cfg.BENCHMARK.GOAL_CENTER - pos)
        is_within_goal = dist < self.cfg.BENCHMARK.GOAL_RADIUS

        if not is_within_goal:
            if self._success_step_counter != 0:
                self._success_step_counter = 0
            return 0

        self._success_step_counter += 1

        if self._success_step_counter >= self._success_step_thresh:
            return 1
        else:
            return 0


class HandoverBenchmarkWrapper(HandoverStatusWrapper):
    _EVAL_SKIP_OBJECT = [0, 15]

    def __init__(self, env):
        super().__init__(env)

        # Seen subjects, camera views, grasped objects.
        if self.cfg.BENCHMARK.SETUP == "s0":
            if self.cfg.BENCHMARK.SPLIT == "train":
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                sequence_ind = [i for i in range(100) if i % 5 != 4]
            if self.cfg.BENCHMARK.SPLIT == "val":
                subject_ind = [0, 1]
                sequence_ind = [i for i in range(100) if i % 5 == 4]
            if self.cfg.BENCHMARK.SPLIT == "test":
                subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
                sequence_ind = [i for i in range(100) if i % 5 == 4]
            mano_side = ["right", "left"]

        # Unseen subjects.
        if self.cfg.BENCHMARK.SETUP == "s1":
            if self.cfg.BENCHMARK.SPLIT == "train":
                subject_ind = [0, 1, 2, 3, 4, 5, 9]
            if self.cfg.BENCHMARK.SPLIT == "val":
                subject_ind = [6]
            if self.cfg.BENCHMARK.SPLIT == "test":
                subject_ind = [7, 8]
            sequence_ind = [*range(100)]
            mano_side = ["right", "left"]

        # Unseen handedness.
        if self.cfg.BENCHMARK.SETUP == "s2":
            if self.cfg.BENCHMARK.SPLIT == "train":
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                mano_side = ["right"]
            if self.cfg.BENCHMARK.SPLIT == "val":
                subject_ind = [0, 1]
                mano_side = ["left"]
            if self.cfg.BENCHMARK.SPLIT == "test":
                subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
                mano_side = ["left"]
            sequence_ind = [*range(100)]

        # Unseen grasped objects.
        if self.cfg.BENCHMARK.SETUP == "s3":
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            if self.cfg.BENCHMARK.SPLIT == "train":
                sequence_ind = [i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)]
            if self.cfg.BENCHMARK.SPLIT == "val":
                sequence_ind = [i for i in range(100) if i // 5 in (3, 19)]
            if self.cfg.BENCHMARK.SPLIT == "test":
                sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]
            mano_side = ["right", "left"]

        self._scene_ids = []
        for i in range(1000):
            if i // 5 % 20 in self._EVAL_SKIP_OBJECT:
                continue
            if i // 100 in subject_ind and i % 100 in sequence_ind:
                if mano_side == ["right", "left"]:
                    self._scene_ids.append(i)
                else:
                    if i % 5 != 4:
                        if (
                            i % 5 in (0, 1)
                            and mano_side == ["right"]
                            or i % 5 in (2, 3)
                            and mano_side == ["left"]
                        ):
                            self._scene_ids.append(i)
                    elif mano_side == self.dex_ycb.load_meta_from_cache(i)["mano_sides"]:
                        self._scene_ids.append(i)

    @property
    def num_scenes(self):
        return len(self._scene_ids)

    def reset(self, env_ids=None, **kwargs):
        if "idx" in kwargs:
            assert "scene_id" not in kwargs
            kwargs["scene_id"] = self._scene_ids[kwargs["idx"]]
            del kwargs["idx"]
        else:
            assert kwargs["scene_id"] in self._scene_ids

        return super().reset(env_ids=env_ids, **kwargs)
