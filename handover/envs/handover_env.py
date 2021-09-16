import gym
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data
import time
import numpy as np

from handover.envs.config import cfg
from handover.envs.dex_ycb import DexYCB
from handover.envs.table import Table
from handover.robots.panda import Panda
from handover.envs.ycb import YCB
from handover.envs.mano import MANO
from handover.utils.transform3d import get_t3d_from_qt


class HandoverEnv(gym.Env):

  def __init__(self, is_render=False):
    self._is_render = is_render

    self._time_step = cfg.ENV.TIME_STEP

    self._release_step_thresh = cfg.ENV.RELEASE_TIME_THRESH / self._time_step

    self._p = None
    self._last_frame_time = 0.0

    self._dex_ycb = DexYCB(is_preload_from_raw=False)
    self._cur_scene_id = None

  def reset(self, scene_id, hard_reset=False):
    if self._p is None:
      hard_reset = True
      if self._is_render:
        self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
      else:
        self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
      self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if self._is_render:
      self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

    if hard_reset:
      self._p.resetSimulation()
      self._p.setGravity(0, 0, -9.8)
      self._p.setPhysicsEngineParameter(fixedTimeStep=self._time_step)
      self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

      self._plane = self._p.loadURDF("plane_implicit.urdf")

      self._table = Table(self._p)

      self._panda = Panda(self._p)
      self._ycb = YCB(self._p, self._dex_ycb)
      self._mano = MANO(self._p, self._dex_ycb)

    if not hard_reset and scene_id != self._cur_scene_id:
      # Remove bodies in reverse added order to maintain deterministic body id
      # assignment for each scene.
      self._mano.clean()
      self._ycb.clean()

    self._cur_scene_id = scene_id

    self._panda.reset()
    self._ycb.reset(scene_id)
    self._mano.reset(scene_id)

    if self._is_render:
      self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

    self._reset_release()

    return None

  def step(self, action):
    if self._is_render:
      # Simulate real-time rendering with sleep if computation takes less than
      # real time.
      time_spent = time.time() - self._last_frame_time
      time_sleep = self._time_step - time_spent
      if time_sleep > 0:
        time.sleep(time_sleep)
      self._last_frame_time = time.time()

    self._panda.set_target_positions(action)
    self._ycb.step()
    self._mano.step()

    self._p.stepSimulation()

    if not self._ycb.released:
      if self._check_release():
        self._ycb.release()

    return None, None, False, {}

  def _reset_release(self):
    self._release_step_counter_passive = 0
    self._release_step_counter_active = 0

  def _check_release(self):
    pts = self._p.getContactPoints(
        bodyA=self._ycb.body_id[self._ycb.ycb_ids[self._ycb.ycb_grasp_ind]],
        bodyB=self._panda.body_id)

    pts = tuple([x for x in pts if x[9] > cfg.ENV.RELEASE_FORCE_THRESH])

    c = []
    for link_index in self._panda.LINK_IND_FINGERS:
      position = [x[6] for x in pts if x[4] == link_index]

      if len(position) == 0:
        c.append(False)
      else:
        pos, orn = self._p.getLinkState(self._panda.body_id, link_index)[4:6]
        t3d = get_t3d_from_qt(orn, pos)
        t3d = t3d.inverse()
        position = np.array(position, dtype=np.float32)
        position = t3d.transform_points(position)

        c_region = (
            (position[:, 0] > cfg.ENV.RELEASE_FINGER_CONTACT_X_RANGE[0]) &
            (position[:, 0] < cfg.ENV.RELEASE_FINGER_CONTACT_X_RANGE[1]) &
            (position[:, 1] > cfg.ENV.RELEASE_FINGER_CONTACT_Y_RANGE[0]) &
            (position[:, 1] < cfg.ENV.RELEASE_FINGER_CONTACT_Y_RANGE[1]) &
            (position[:, 2] > cfg.ENV.RELEASE_FINGER_CONTACT_Z_RANGE[0]) &
            (position[:, 2] < cfg.ENV.RELEASE_FINGER_CONTACT_Z_RANGE[1]))
        c.append(np.any(c_region))

    if not any(c) and len(pts) > 0:
      self._release_step_counter_passive += 1
    else:
      if self._release_step_counter_passive != 0:
        self._release_step_counter_passive = 0

    if all(c):
      self._release_step_counter_active += 1
    else:
      if self._release_step_counter_active != 0:
        self._release_step_counter_active = 0

    return (self._release_step_counter_passive >= self._release_step_thresh or
            self._release_step_counter_active >= self._release_step_thresh)
