import numpy as np
import os

from handover.envs.dex_ycb import DexYCB

_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}


# TODO(ywchao): add ground-truth motions.
class YCB():

  def __init__(self, bullet_client):
    self._p = bullet_client

    xs = np.linspace(0.0, 1.2, 5)
    ys = np.linspace(0.0, 1.0, 4)
    offset = np.array([0.0, 2.0, 0.2], dtype=np.float32)
    self._base_position = {
        i: np.array([xs[o % len(xs)], ys[o // len(xs)], 0], dtype=np.float32) +
        offset for o, i in enumerate(_CLASSES)
    }
    self._base_orientation = [0, 0, 0, 1]

    # TODO(ywchao): use stable position for both q and t.
    self._table_height = 0.625
    self._float_offset_t = np.array([0.0, 0.0, self._table_height + 0.01],
                                    dtype=np.float32)

    self._translation_gain_p = [0.2] * 3
    self._translation_gain_d = [1.0] * 3
    self._rotation_gain_p = 1.0

    self._objects = None
    self._frame = 0
    self._num_frames = 0

    self._dex_ycb = DexYCB(load_cache=True)

  # TODO(ywchao): add default value for scene_id (random sampling).
  def reset(self, scene_id):
    if self._objects is None:
      self._objects = {}
      for i, name in _CLASSES.items():
        urdf_file = os.path.join(os.path.dirname(__file__), "../data/models",
                                 name, "model_normalized.urdf")
        uid = self._p.loadURDF(urdf_file,
                               basePosition=self._base_position[i],
                               baseOrientation=self._base_orientation,
                               useFixedBase=True)
        self._objects[i] = uid

    self._scene_id = scene_id
    scene_data = self._dex_ycb.get_scene_data(self._scene_id)
    self._ycb_ids = scene_data['ycb_ids']
    self._q = scene_data['pose'][:, :, :4].copy()
    self._t = scene_data['pose'][:, :, 4:].copy()

    # TODO(ywchao): use stable position for both q and t.
    self._t += self._float_offset_t

    self._frame = 0
    self._num_frames = len(scene_data['pose'])

    for i, uid in self._objects.items():
      if i in self._ycb_ids:
        q, t = self.get_target_positions(self._frame, self._ycb_ids.index(i))
      else:
        q = [0, 0, 0, 1]
        t = [0, 0, 0]

      # Reset joint states.
      self._p.resetJointState(self._objects[i], 0, t[0], targetVelocity=0)
      self._p.resetJointState(self._objects[i], 1, t[1], targetVelocity=0)
      self._p.resetJointState(self._objects[i], 2, t[2], targetVelocity=0)
      self._p.resetJointStateMultiDof(self._objects[i],
                                      3,
                                      q,
                                      targetVelocity=[0, 0, 0])

      # Reset controllers.
      self._p.setJointMotorControlArray(self._objects[i], [0, 1, 2],
                                        self._p.POSITION_CONTROL,
                                        forces=[0, 0, 0])
      self._p.setJointMotorControlMultiDof(self._objects[i],
                                           3,
                                           self._p.POSITION_CONTROL,
                                           targetPosition=[0, 0, 0, 1],
                                           targetVelocity=[0, 0, 0],
                                           force=[0, 0, 0])

      # Speed up simulation by disabling collision for unused objects.
      if i in self._ycb_ids:
        collision_id = -1
      else:
        collision_id = 0
      for j in range(self._p.getNumJoints(uid)):
        self._p.setCollisionFilterGroupMask(uid,
                                            j,
                                            collisionFilterGroup=collision_id,
                                            collisionFilterMask=collision_id)

  def step(self):
    self._frame += 1
    self._frame = min(self._frame, self._num_frames - 1)

    # TODO(ywchao): control grasped object only and use stable position for static ones.
    # TODO(ywchao): assert position is constant for static objects.
    # Set target position.
    for o, i in enumerate(self._ycb_ids):
      q, t = self.get_target_positions(self._frame, o)
      self._p.setJointMotorControlArray(self._objects[i], [0, 1, 2],
                                        self._p.POSITION_CONTROL,
                                        targetPositions=t,
                                        positionGains=self._translation_gain_p,
                                        velocityGains=self._translation_gain_d)
      # targetVelocity and velocityGain seem not to have any effect here.
      self._p.setJointMotorControlMultiDof(self._objects[i],
                                           3,
                                           self._p.POSITION_CONTROL,
                                           targetPosition=q,
                                           positionGain=self._rotation_gain_p)

  def get_target_positions(self, frame, obj_id):
    q = self._q[frame, obj_id]
    t = self._t[frame, obj_id] - self._base_position[self._ycb_ids[obj_id]]
    return q, t
