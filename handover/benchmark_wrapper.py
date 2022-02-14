import numpy as np

from handover.handover_env import HandoverEnv


class HandoverStatusEnv(HandoverEnv):
  _FAILURE_ROBOT_HUMAN_CONTACT = -1
  _FAILURE_OBJECT_DROP = -2
  _FAILURE_TIMEOUT = -4

  def init(self):
    super().init()

    self._success_step_thresh = self.cfg.BENCHMARK.SUCCESS_TIME_THRESH / self.cfg.SIM.TIME_STEP
    self._max_episode_steps = self.cfg.BENCHMARK.MAX_EPISODE_TIME / self.cfg.SIM.TIME_STEP

  @property
  def goal_center(self):
    return self.cfg.BENCHMARK.GOAL_CENTER

  @property
  def goal_radius(self):
    return self.cfg.BENCHMARK.GOAL_RADIUS

  def pre_reset(self, env_ids, scene_id):
    super().pre_reset(env_ids, scene_id)

    if self.cfg.BENCHMARK.IS_DRAW_GOAL:
      raise NotImplementedError

    self._elapsed_steps = 0

    self._dropped = False
    self._success_step_counter = 0

  def post_step(self, action):
    observation, reward, done, info = super().post_step(action)

    self._elapsed_steps += 1

    status = self.check_status()

    if self._elapsed_steps >= self._max_episode_steps and status != 1:
      status += self._FAILURE_TIMEOUT

    done |= status != 0

    info['status'] = status

    return observation, reward, done, info

  def check_status(self):
    status = 0

    if self._mano.body is not None:
      contact = self.contact[0]

      contact_1 = contact[(contact['body_id_a'] == self._mano.body.contact_id) &
                          (contact['body_id_b'] == self._panda.body.contact_id)]
      contact_2 = contact[(contact['body_id_a'] == self._panda.body.contact_id)
                          &
                          (contact['body_id_b'] == self._mano.body.contact_id)]
      contact = np.concatenate((contact_1, contact_2))

      for x in contact:
        if x['force'] > self.cfg.BENCHMARK.CONTACT_FORCE_THRESH:
          status += self._FAILURE_ROBOT_HUMAN_CONTACT
          break

    if not self._ycb.released:
      return status

    if not self._dropped:
      contact = self.contact[0]
      contact_1 = contact[contact['body_id_a'] ==
                          self._ycb.grasped_body.contact_id]
      contact_2 = contact[contact['body_id_b'] ==
                          self._ycb.grasped_body.contact_id]
      contact_2[['body_id_a',
                 'body_id_b']] = contact_2[['body_id_b', 'body_id_a']]
      contact_2[['link_id_a',
                 'link_id_b']] = contact_2[['link_id_b', 'link_id_a']]
      contact_2[['position_a_world', 'position_b_world'
                ]] = contact_2[['position_b_world', 'position_a_world']]
      contact_2[['position_a_link', 'position_b_link'
                ]] = contact_2[['position_b_link', 'position_a_link']]
      contact_2['normal']['x'] *= -1
      contact_2['normal']['y'] *= -1
      contact_2['normal']['z'] *= -1
      contact = np.concatenate((contact_1, contact_2))

      contact_panda = contact[contact['body_id_b'] ==
                              self._panda.body.contact_id]
      contact_table = contact[contact['body_id_b'] ==
                              self._table.body.contact_id]
      contact_non_grasped_ycb = contact[np.any([
          contact['body_id_b'] == x.contact_id
          for x in self._ycb.non_grasped_bodies
      ],
                                               axis=0)]

      panda_link_ind = contact_panda['link_id_b'][
          contact_panda['force'] > self.cfg.BENCHMARK.CONTACT_FORCE_THRESH]
      contact_panda_fingers = set(
          self._panda.LINK_IND_FINGERS).issubset(panda_link_ind)
      contact_table = np.any(
          contact_table['force'] > self.cfg.BENCHMARK.CONTACT_FORCE_THRESH)
      contact_non_grasped_ycb = np.any(contact_non_grasped_ycb['force'] >
                                       self.cfg.BENCHMARK.CONTACT_FORCE_THRESH)

      is_below_table = self._ycb.grasped_body.link_state[0][
          6, 2] < self.cfg.ENV.TABLE_HEIGHT

      if not contact_panda_fingers and (contact_table or contact_non_grasped_ycb
                                        or is_below_table):
        self._dropped = True

    if self._dropped:
      status += self._FAILURE_OBJECT_DROP

    if status < 0:
      return status

    if not contact_panda_fingers:
      if self._success_step_counter != 0:
        self._success_step_counter = 0
      return 0

    pos = self._panda.body.link_state[0][self._panda.LINK_IND_HAND, 0:3].numpy()
    dist = np.linalg.norm(pos - self.goal_center)
    is_within_goal = dist < self.goal_radius

    if not is_within_goal:
      if self._success_step_counter != 0:
        self._success_step_counter = 0
      return 0

    self._success_step_counter += 1

    if self._success_step_counter >= self._success_step_thresh:
      return 1
    else:
      return 0


class HandoverBenchmarkEnv(HandoverStatusEnv):
  _EVAL_SKIP_OBJECT = [0, 15]

  def init(self):
    super().init()

    # TODO(ywchao): move scene_ids calculation to dex_ycb.py.
    # Seen subjects, camera views, grasped objects.
    if self.cfg.BENCHMARK.SETUP == 's0':
      if self.cfg.BENCHMARK.SPLIT == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        sequence_ind = [i for i in range(100) if i % 5 != 4]
      if self.cfg.BENCHMARK.SPLIT == 'val':
        subject_ind = [0, 1]
        sequence_ind = [i for i in range(100) if i % 5 == 4]
      if self.cfg.BENCHMARK.SPLIT == 'test':
        subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
        sequence_ind = [i for i in range(100) if i % 5 == 4]

    # Unseen subjects.
    if self.cfg.BENCHMARK.SETUP == 's1':
      if self.cfg.BENCHMARK.SPLIT == 'train':
        raise NotImplementedError
      if self.cfg.BENCHMARK.SPLIT == 'val':
        raise NotImplementedError
      if self.cfg.BENCHMARK.SPLIT == 'test':
        raise NotImplementedError

    # Unseen handedness.
    if self.cfg.BENCHMARK.SETUP == 's2':
      if self.cfg.BENCHMARK.SPLIT == 'train':
        raise NotImplementedError
      if self.cfg.BENCHMARK.SPLIT == 'val':
        raise NotImplementedError
      if self.cfg.BENCHMARK.SPLIT == 'test':
        raise NotImplementedError

    # Unseen grasped objects.
    if self.cfg.BENCHMARK.SETUP == 's3':
      if self.cfg.BENCHMARK.SPLIT == 'train':
        raise NotImplementedError
      if self.cfg.BENCHMARK.SPLIT == 'val':
        raise NotImplementedError
      if self.cfg.BENCHMARK.SPLIT == 'test':
        raise NotImplementedError

    self._scene_ids = []
    for i in range(1000):
      if i // 100 in subject_ind and i % 100 in sequence_ind:
        if i // 5 % 20 in self._EVAL_SKIP_OBJECT:
          continue
        self._scene_ids.append(i)

  @property
  def num_scenes(self):
    return len(self._scene_ids)

  def pre_reset(self, env_ids, idx=None, scene_id=None):
    if scene_id is None:
      scene_id = self._scene_ids[idx]
    else:
      assert scene_id in self._scene_ids

    super().pre_reset(env_ids, scene_id)

  def post_reset(self, env_ids, idx=None, scene_id=None):
    observation = super().post_reset(env_ids, scene_id)
    return observation
