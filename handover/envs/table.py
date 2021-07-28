import os

from handover.envs.config import cfg


class Table():

  def __init__(self, bullet_client):
    self._p = bullet_client

    urdf_file = os.path.join(os.path.dirname(__file__), "..", "data", "assets",
                             "table", "table.urdf")
    self._body_id = self._p.loadURDF(
        urdf_file,
        basePosition=cfg.ENV.TABLE_BASE_POSITION,
        baseOrientation=cfg.ENV.TABLE_BASE_ORIENTATION)

    self._p.changeVisualShape(self._body_id, -1, rgbaColor=[1, 1, 1, 1])

    self._p.setCollisionFilterGroupMask(self._body_id, -1,
                                        cfg.ENV.COLLISION_ID_TABLE,
                                        cfg.ENV.COLLISION_ID_TABLE)

  @property
  def body_id(self):
    return self._body_id
