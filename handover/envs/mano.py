import os
import numpy as np

from mano_pybullet.hand_model import HandModel45
from mano_pybullet.hand_body_base_joint import HandBodyBaseJoint

_COLLISION_ID = 2**22


# TODO(ywchao): add ground-truth motions.
class MANO():

  def __init__(self, bullet_client, dex_ycb, table_height):
    self._p = bullet_client
    self._dex_ycb = dex_ycb
    self._table_height = table_height

    self._models_dir = os.path.join(os.path.dirname(__file__),
                                    "../data/mano_v1_2/models")

    self._body = None
    self._frame = 0
    self._num_frames = 0

  def reset(self, scene_id=None):
    assert self._body is None

    if scene_id is None:
      return
    else:
      scene_data = self._dex_ycb.get_scene_data(scene_id)
      self._mano_side = scene_data['mano_sides'][0]
      self._mano_betas = scene_data['mano_betas'][0]
      pose = scene_data['pose_m'][:, 0]

    self._sid = np.where(np.any(pose != 0, axis=1))[0][0]
    self._eid = np.where(np.any(pose != 0, axis=1))[0][-1]

    self._q = pose[:, 0:48].copy()
    self._t = pose[:, 48:51].copy()

    self._t[self._sid:self._eid + 1, 2] += self._table_height

    self._frame = 0
    self._num_frames = len(self._q)

    if self._frame == self._sid:
      self.make()

  def make(self):
    model = HandModel45(left_hand=self._mano_side == 'left',
                        models_dir=self._models_dir,
                        betas=self._mano_betas)
    self._body = HandBodyBaseJoint(self._p, model, shape_betas=self._mano_betas)

    self._body.reset_from_mano(self._t[self._frame], self._q[self._frame])

    for j in range(4, 50, 3):
      self._p.setCollisionFilterGroupMask(self._body.body_id,
                                          j,
                                          collisionFilterGroup=_COLLISION_ID,
                                          collisionFilterMask=_COLLISION_ID)

  def clean(self):
    if self._body is not None:
      self._p.removeBody(self._body.body_id)
      self._body = None

  def step(self):
    self._frame += 1
    self._frame = min(self._frame, self._num_frames - 1)

    if self._frame == self._sid:
      self.make()
    if self._frame > self._sid and self._frame <= self._eid:
      self._body.set_target_from_mano(self._t[self._frame],
                                      self._q[self._frame])
    if self._frame == self._eid + 1:
      self.clean()
