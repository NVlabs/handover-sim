import gym
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data
import time

from handover.robots.panda import Panda
from handover.envs.dex_ycb import DexYCB
from handover.envs.ycb import YCB
from handover.envs.mano import MANO


class HandoverEnv(gym.Env):

  def __init__(self,
               is_render=False,
               is_control_object=True,
               is_load_panda_mano=True):
    self._is_render = is_render
    self._is_control_object = is_control_object
    self._is_load_panda_mano = is_load_panda_mano

    self._time_step = 0.001

    self._table_base_position = [0.6, 0.3, 0.0]
    self._table_base_orientation = [0, 0, 0, 1]
    self._panda_base_position = [0.6, -0.5, 0.575]
    self._panda_base_orientation = [0.0, 0.0, 0.7071068, 0.7071068]

    self._p = None
    self._last_frame_time = 0.0

  @property
  def num_scenes(self):
    return self._ycb.num_scenes

  def reset(self, hard_reset=False, scene_id=None, pose=None):
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
      self._table = self._p.loadURDF(
          "table/table.urdf",
          basePosition=self._table_base_position,
          baseOrientation=self._table_base_orientation)
      self._p.changeVisualShape(self._table, -1, rgbaColor=[1, 1, 1, 1])
      self._table_height = 0.625

      if self._is_load_panda_mano:
        self._panda = Panda(self._p,
                            base_position=self._panda_base_position,
                            base_orientation=self._panda_base_orientation)

      self._dex_ycb = DexYCB(load_cache=True)
      self._ycb = YCB(self._p,
                      self._dex_ycb,
                      self._table_height,
                      is_control_object=self._is_control_object)
      if self._is_load_panda_mano:
        self._mano = MANO(self._p, self._dex_ycb, self._table_height)

    if self._is_load_panda_mano:
      self._panda.reset()
    self._ycb.reset(scene_id=scene_id, pose=pose)
    if self._is_load_panda_mano:
      self._mano.reset(scene_id=scene_id)

    if self._is_render:
      self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

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

    if self._is_load_panda_mano:
      self._panda.set_target_positions(action)
    self._ycb.step()
    if self._is_load_panda_mano:
      self._mano.step()

    self._p.stepSimulation()

    return None, None, None, None
