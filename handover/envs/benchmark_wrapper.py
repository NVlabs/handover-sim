import numpy as np

from handover.envs.handover_env import HandoverEnv


class HandoverStatusEnv(HandoverEnv):
  _FAILURE_ROBOT_HUMAN_CONTACT = -1
  _FAILURE_OBJECT_DROP = -2
  _FAILURE_TIMEOUT = -4

  def __init__(self, is_render=False, is_draw_goal=False):
    super().__init__(is_render=is_render)

    self._is_draw_goal = is_draw_goal

    self._contact_force_threshold = 0.0

    self._goal_center = [0.61, -0.20, 1.25]
    self._goal_radius = 0.15
    self._goal_color = [0.85, 0.19, 0.21, 0.5]

    self._success_time_threshold = 0.1
    self._success_step_threshold = self._success_time_threshold / self._time_step

    self._max_episode_time = 13.0
    self._max_episode_steps = self._max_episode_time / self._time_step

  @property
  def goal_center(self):
    return self._goal_center

  @property
  def goal_radius(self):
    return self._goal_radius

  def reset(self, scene_id, hard_reset=False):
    if self._p is None:
      hard_reset = True

    observation = super().reset(scene_id, hard_reset=hard_reset)

    if self._is_render:
      self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

    if hard_reset:
      if self._is_draw_goal:
        visual_id = self._p.createVisualShape(self._p.GEOM_SPHERE,
                                              radius=self._goal_radius,
                                              rgbaColor=self._goal_color)
        self._goal = self._p.createMultiBody(baseMass=0.0,
                                             baseVisualShapeIndex=visual_id,
                                             basePosition=self._goal_center)

    if self._is_render:
      self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

    self._static_ycb_body_id = [
        self._ycb.body_id[y]
        for y in self._ycb.body_id
        if y != self._ycb.ycb_ids[self._ycb.ycb_grasp_ind]
    ]

    self._elapsed_steps = 0

    self._dropped = False
    self._success_step_counter = 0

    return observation

  def step(self, action):
    observation, reward, done, info = super().step(action)

    self._elapsed_steps += 1

    status = self.check_status()

    if self._elapsed_steps >= self._max_episode_steps and status != 1:
      status += self._FAILURE_TIMEOUT

    done |= status != 0

    info['status'] = status

    return observation, reward, done, info

  def check_status(self):
    status = 0

    if self._mano.body_id is not None:
      pts = self._p.getContactPoints(bodyA=self._mano.body_id,
                                     bodyB=self._panda.body_id)
      for x in pts:
        if x[9] > self._contact_force_threshold:
          status += self._FAILURE_ROBOT_HUMAN_CONTACT
          break

    if not self._ycb.released:
      return status

    if not self._dropped:
      pts = self._p.getContactPoints(
          bodyA=self._ycb.body_id[self._ycb.ycb_ids[self._ycb.ycb_grasp_ind]])

      pts_panda = [x for x in pts if x[2] == self._panda.body_id]
      pts_table = [x for x in pts if x[2] == self._table.body_id]
      pts_static_ycb = [x for x in pts if x[2] in self._static_ycb_body_id]

      pts_panda_link_id = [
          x[4] for x in pts_panda if x[9] > self._contact_force_threshold
      ]
      is_contact_panda_fingers = set(
          self._panda.LINK_ID_FINGERS).issubset(pts_panda_link_id)
      is_contact_table = any(
          [x[9] > self._contact_force_threshold for x in pts_table])
      is_contact_static_ycb = any(
          [x[9] > self._contact_force_threshold for x in pts_static_ycb])

      pos, _ = self._ycb.get_base_state(
          self._ycb.ycb_ids[self._ycb.ycb_grasp_ind])
      is_below_table = pos[6] < self._table.HEIGHT

      if not is_contact_panda_fingers and (is_contact_table or
                                           is_contact_static_ycb or
                                           is_below_table):
        self._dropped = True

    if self._dropped:
      status += self._FAILURE_OBJECT_DROP

    if status < 0:
      return status

    if not is_contact_panda_fingers:
      if self._success_step_counter != 0:
        self._success_step_counter = 0
      return 0

    pos = self._p.getLinkState(self._panda.body_id, self._panda.LINK_ID_HAND)[4]
    dist = np.linalg.norm(np.array(pos, dtype=np.float32) - self._goal_center)
    is_within_goal = dist < self._goal_radius

    if not is_within_goal:
      if self._success_step_counter != 0:
        self._success_step_counter = 0
      return 0

    self._success_step_counter += 1

    if self._success_step_counter >= self._success_step_threshold:
      return 1
    else:
      return 0


class HandoverBenchmarkEnv(HandoverStatusEnv):
  _EVAL_SKIP_OBJECT = [0, 15]

  def __init__(self, setup, split, is_render=False, is_draw_goal=False):
    super().__init__(is_render=is_render, is_draw_goal=is_draw_goal)

    self._setup = setup
    self._split = split

    # TODO(ywchao): move scene_ids calculation to dex_ycb.py.
    # Seen subjects, camera views, grasped objects.
    if self._setup == 's0':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        sequence_ind = [i for i in range(100) if i % 5 != 4]
      if self._split == 'val':
        subject_ind = [0, 1]
        sequence_ind = [i for i in range(100) if i % 5 == 4]
      if self._split == 'test':
        subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
        sequence_ind = [i for i in range(100) if i % 5 == 4]

    # Unseen subjects.
    if self._setup == 's1':
      if self._split == 'train':
        raise NotImplementedError
      if self._split == 'val':
        raise NotImplementedError
      if self._split == 'test':
        raise NotImplementedError

    # Unseen handedness.
    if self._setup == 's2':
      if self._split == 'train':
        raise NotImplementedError
      if self._split == 'val':
        raise NotImplementedError
      if self._split == 'test':
        raise NotImplementedError

    # Unseen grasped objects.
    if self._setup == 's3':
      if self._split == 'train':
        raise NotImplementedError
      if self._split == 'val':
        raise NotImplementedError
      if self._split == 'test':
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

  def reset(self, idx=None, scene_id=None, hard_reset=False):
    if scene_id is None:
      scene_id = self._scene_ids[idx]
    else:
      assert scene_id in self._scene_ids

    observation = super().reset(scene_id, hard_reset=hard_reset)

    return observation
