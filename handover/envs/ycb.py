import numpy as np
import os


# TODO(ywchao): add ground-truth motions.
class YCB():
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

  _COLLISION_ID = lambda self, i: 2**i

  def __init__(self, bullet_client, dex_ycb, table_height):
    self._p = bullet_client
    self._dex_ycb = dex_ycb
    self._table_height = table_height

    xs = np.linspace(0.0, 1.2, 5)
    ys = np.linspace(0.0, 1.0, 4)
    offset = np.array([0.0, 2.0, 0.2], dtype=np.float32)
    self._base_position = {
        i: np.array([xs[o % len(xs)], ys[o // len(xs)], 0], dtype=np.float32) +
        offset for o, i in enumerate(self._CLASSES)
    }
    self._base_orientation = [0, 0, 0, 1]

    self._translation_gain_p = [0.2] * 3
    self._translation_gain_d = [1.0] * 3
    self._rotation_gain_p = 1.0

    self._body_id = {}

  @property
  def body_id(self):
    return self._body_id

  @property
  def ycb_ids(self):
    return self._ycb_ids

  @property
  def ycb_grasp_ind(self):
    return self._ycb_grasp_ind

  @property
  def released(self):
    return self._released

  def reset(self, scene_id):
    scene_data = self._dex_ycb.get_scene_data(scene_id)
    self._ycb_ids = scene_data['ycb_ids']
    self._ycb_grasp_ind = scene_data['ycb_grasp_ind']
    self._pose = scene_data['pose_y']

    self._q = self._pose[:, :, :4].copy()
    self._t = self._pose[:, :, 4:].copy()

    self._t[:, :, 2] += self._table_height

    self._frame = 0
    self._num_frames = len(self._q)
    self._released = False

    if self._body_id == {}:
      for i in self._ycb_ids:
        urdf_file = os.path.join(os.path.dirname(__file__), "..", "data",
                                 "assets", self._CLASSES[i],
                                 "model_normalized.urdf")
        self._body_id[i] = self._p.loadURDF(
            urdf_file,
            basePosition=self._base_position[i],
            baseOrientation=self._base_orientation,
            useFixedBase=True)
    else:
      assert list(self._body_id.keys()) == self._ycb_ids

    for i in self._ycb_ids:
      q, t = self.get_target_position(self._frame, self._ycb_ids.index(i))

      # Reset joint states.
      self._p.resetJointState(self._body_id[i], 0, t[0])
      self._p.resetJointState(self._body_id[i], 1, t[1])
      self._p.resetJointState(self._body_id[i], 2, t[2])
      self._p.resetJointStateMultiDof(self._body_id[i], 3, q)

      # Reset controllers.
      self._p.setJointMotorControlArray(self._body_id[i], [0, 1, 2],
                                        self._p.POSITION_CONTROL,
                                        forces=[0, 0, 0])
      self._p.setJointMotorControlMultiDof(self._body_id[i],
                                           3,
                                           self._p.POSITION_CONTROL,
                                           targetPosition=[0, 0, 0, 1],
                                           force=[0, 0, 0])

      for j in range(self._p.getNumJoints(self._body_id[i])):
        self._p.setCollisionFilterGroupMask(
            self._body_id[i],
            j,
            collisionFilterGroup=self._COLLISION_ID(i),
            collisionFilterMask=self._COLLISION_ID(i))

  def clean(self):
    # Remove bodies in reverse added order to maintain deterministic body id
    # assignment for each scene.
    for i in list(self._body_id)[::-1]:
      self._p.removeBody(self._body_id[i])
      self._body_id.pop(i)

  def step(self):
    self._frame += 1
    self._frame = min(self._frame, self._num_frames - 1)

    # TODO(ywchao): control grasped object only and use stable position for static ones.
    # TODO(ywchao): assert position is constant for static objects.
    # Set target position.
    for o, i in enumerate(self._ycb_ids):
      if self._released and o == self._ycb_grasp_ind:
        continue
      q, t = self.get_target_position(self._frame, o)
      self._p.setJointMotorControlArray(self._body_id[i], [0, 1, 2],
                                        self._p.POSITION_CONTROL,
                                        targetPositions=t,
                                        positionGains=self._translation_gain_p,
                                        velocityGains=self._translation_gain_d)
      # targetVelocity and velocityGain seem not to have any effect here.
      self._p.setJointMotorControlMultiDof(self._body_id[i],
                                           3,
                                           self._p.POSITION_CONTROL,
                                           targetPosition=q,
                                           positionGain=self._rotation_gain_p)

  def release(self, mano_collision_id):
    self._p.setJointMotorControlArray(
        self._body_id[self._ycb_ids[self._ycb_grasp_ind]], [0, 1, 2],
        self._p.POSITION_CONTROL,
        forces=[0, 0, 0])
    self._p.setJointMotorControlMultiDof(
        self._body_id[self._ycb_ids[self._ycb_grasp_ind]],
        3,
        self._p.POSITION_CONTROL,
        targetPosition=[0, 0, 0, 1],
        force=[0, 0, 0])

    collision_id = -1 - mano_collision_id
    for j in range(
        self._p.getNumJoints(
            self._body_id[self._ycb_ids[self._ycb_grasp_ind]])):
      self._p.setCollisionFilterGroupMask(
          self._body_id[self._ycb_ids[self._ycb_grasp_ind]],
          j,
          collisionFilterGroup=collision_id,
          collisionFilterMask=collision_id)

    self._released = True

  def get_target_position(self, frame, obj_id):
    q = self._q[frame, obj_id]
    t = self._t[frame, obj_id] - self._base_position[self._ycb_ids[obj_id]]
    return q, t

  def get_base_state(self, ycb_id, is_table_frame=False):
    state_trans = self._p.getJointStates(self._body_id[ycb_id], [0, 1, 2])
    state_rot = self._p.getJointStateMultiDof(self._body_id[ycb_id], 3)
    pos_trans = [s[0] for s in state_trans]
    vel_trans = [s[1] for s in state_trans]
    pos_trans = [
        s + self._base_position[ycb_id][i] for i, s, in enumerate(pos_trans)
    ]
    if is_table_frame:
      pos_trans[2] -= self._table_height
    pos = state_rot[0] + tuple(pos_trans)
    vel = state_rot[1] + tuple(vel_trans)
    return pos, vel
