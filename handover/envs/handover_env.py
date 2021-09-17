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

    if hard_reset and cfg.ENV.IS_DRAW_RELEASE:
      self._release_draw_reset()

    if self._is_render:
      self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

    self._release_reset()

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

    if not self._ycb.released and self._release_check():
      self._ycb.release()

    if cfg.ENV.IS_DRAW_RELEASE:
      self._release_draw_step()

    return None, None, False, {}

  def _release_reset(self):
    self._release_step_counter_passive = 0
    self._release_step_counter_active = 0

  def _release_check(self):
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

  def _release_draw_reset(self):
    try:
      self._release_contact_center
      self._release_contact_half_extents
      self._release_contact_vertices
    except AttributeError:
      self._release_contact_center = np.array([[
          sum(cfg.ENV.RELEASE_FINGER_CONTACT_X_RANGE) / 2,
          sum(cfg.ENV.RELEASE_FINGER_CONTACT_Y_RANGE) / 2,
          sum(cfg.ENV.RELEASE_FINGER_CONTACT_Z_RANGE) / 2
      ]],
                                              dtype=np.float32)
      self._release_contact_half_extents = [
          (cfg.ENV.RELEASE_FINGER_CONTACT_X_RANGE[1] -
           cfg.ENV.RELEASE_FINGER_CONTACT_X_RANGE[0]) / 2,
          (cfg.ENV.RELEASE_FINGER_CONTACT_Y_RANGE[1] -
           cfg.ENV.RELEASE_FINGER_CONTACT_Y_RANGE[0]) / 2,
          (cfg.ENV.RELEASE_FINGER_CONTACT_Z_RANGE[1] -
           cfg.ENV.RELEASE_FINGER_CONTACT_Z_RANGE[0]) / 2
      ]
      vertices = []
      for x in cfg.ENV.RELEASE_FINGER_CONTACT_X_RANGE:
        for y in cfg.ENV.RELEASE_FINGER_CONTACT_Y_RANGE:
          for z in cfg.ENV.RELEASE_FINGER_CONTACT_Z_RANGE:
            vertices.append([x, y, z])
      self._release_contact_vertices = np.array(vertices, dtype=np.float32)

    self._body_id_release_contact_region = {
        k: None for k in self._panda.LINK_IND_FINGERS
    }
    self._body_id_release_contact_vertices = {
        k: [None for _ in range(len(self._release_contact_vertices))
           ] for k in self._panda.LINK_IND_FINGERS
    }

    for link_index in self._panda.LINK_IND_FINGERS:
      pos, orn = self._p.getLinkState(self._panda.body_id, link_index)[4:6]
      t3d = get_t3d_from_qt(orn, pos)
      center = t3d.transform_points(self._release_contact_center)[0]
      vertices = t3d.transform_points(self._release_contact_vertices)

      visual_id = self._p.createVisualShape(
          self._p.GEOM_BOX,
          halfExtents=self._release_contact_half_extents,
          rgbaColor=cfg.ENV.RELEASE_CONTACT_REGION_COLOR)
      self._body_id_release_contact_region[
          link_index] = self._p.createMultiBody(baseMass=0.0,
                                                baseVisualShapeIndex=visual_id,
                                                basePosition=center,
                                                baseOrientation=orn)
      visual_id = self._p.createVisualShape(
          self._p.GEOM_SPHERE,
          radius=cfg.ENV.RELEASE_CONTACT_VERTEX_RADIUS,
          rgbaColor=cfg.ENV.RELEASE_CONTACT_VERTEX_COLOR)
      for i in range(len(vertices)):
        self._body_id_release_contact_vertices[link_index][
            i] = self._p.createMultiBody(baseMass=0.0,
                                         baseVisualShapeIndex=visual_id,
                                         basePosition=vertices[i])

  def _release_draw_step(self):
    for link_index in self._panda.LINK_IND_FINGERS:
      pos, orn = self._p.getLinkState(self._panda.body_id, link_index)[4:6]
      t3d = get_t3d_from_qt(orn, pos)
      center = t3d.transform_points(self._release_contact_center)[0]
      vertices = t3d.transform_points(self._release_contact_vertices)

      self._p.resetBasePositionAndOrientation(
          self._body_id_release_contact_region[link_index], center, orn)
      for i in range(len(vertices)):
        self._p.resetBasePositionAndOrientation(
            self._body_id_release_contact_vertices[link_index][i], vertices[i],
            [0, 0, 0, 1])
