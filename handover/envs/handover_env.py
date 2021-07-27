import gym
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data
import time

from handover.envs.dex_ycb import DexYCB
from handover.envs.table import Table
from handover.robots.panda import Panda
from handover.envs.ycb import YCB
from handover.envs.mano import MANO


class HandoverEnv(gym.Env):

  def __init__(self, is_render=False):
    self._is_render = is_render

    self._time_step = 0.001

    self._table_base_position = [0.61, 0.28, 0.0]
    self._table_base_orientation = [0, 0, 0, 1]
    self._panda_base_position = [0.61, -0.50, 0.875]
    self._panda_base_orientation = [0.0, 0.0, 0.7071068, 0.7071068]

    self._release_force_threshold = 0.0
    self._release_time_threshold = 0.2
    self._release_step_threshold = self._release_time_threshold / self._time_step

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

      self._table = Table(self._p,
                          base_position=self._table_base_position,
                          base_orientation=self._table_base_orientation)

      self._panda = Panda(self._p,
                          base_position=self._panda_base_position,
                          base_orientation=self._panda_base_orientation)
      self._ycb = YCB(self._p, self._dex_ycb, self._table.HEIGHT)
      self._mano = MANO(self._p, self._dex_ycb, self._table.HEIGHT)

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

    self._release_step_counter = 0

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
      pts = self._p.getContactPoints(
          bodyA=self._ycb.body_id[self._ycb.ycb_ids[self._ycb.ycb_grasp_ind]],
          bodyB=self._panda.body_id)
      if any([x[9] > self._release_force_threshold for x in pts]):
        self._release_step_counter += 1
      else:
        if self._release_step_counter != 0:
          self._release_step_counter = 0
      if self._release_step_counter >= self._release_step_threshold:
        self._ycb.release(self._mano.COLLISION_ID)

    return None, None, False, {}
