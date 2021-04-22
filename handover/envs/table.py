import os

_HEIGHT = 0.92

_COLLISION_ID = 2**0


class Table():
  height = _HEIGHT

  def __init__(self,
               bullet_client,
               base_position=[0, 0, 0],
               base_orientation=[0, 0, 0, 1],
               is_filter_collision=True):
    self._p = bullet_client
    self._base_position = base_position
    self._base_orientation = base_orientation
    self._is_filter_collision = is_filter_collision

    urdf_file = os.path.join(os.path.dirname(__file__),
                             "../data/assets/table/table.urdf")
    self._body_id = self._p.loadURDF(urdf_file,
                                     basePosition=self._base_position,
                                     baseOrientation=self._base_orientation)

    self._p.changeVisualShape(self._body_id, -1, rgbaColor=[1, 1, 1, 1])

    if self._is_filter_collision:
      self._p.setCollisionFilterGroupMask(self._body_id,
                                          -1,
                                          collisionFilterGroup=_COLLISION_ID,
                                          collisionFilterMask=_COLLISION_ID)
