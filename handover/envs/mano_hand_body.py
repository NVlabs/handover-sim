"""

Derived from:
https://github.com/ikalevatykh/mano_pybullet/blob/960c257cf465f8966e770562b66150beaa359230/mano_pybullet/hand_body.py
"""

import pybullet as pb

from mano_pybullet.hand_body_base_joint import HandBodyBaseJoint


class HandBodyBaseJointURDF(HandBodyBaseJoint):
  """Rigid multi-link hand body with base joints (URDF loader)."""

  def __init__(self, client, hand_model, urdf_file, **kwargs):
    """Constructor.

    Args:
      client: pybullet client.
      hand_model: Rigid hand model.
      urdf_file: Path to the URDF file.
      kwargs: Keyward arguments.
        flags: Configuration flags (default: FLAG_DEFAULT).
        shape_betas: A numpy array of shape [10] containing the MANO shape beta
          parameters (default: None).
    """
    self._urdf_file = urdf_file

    super().__init__(client, hand_model, **kwargs)

  def _make_body(self):
    joints = self._model.joints

    link_masses_counter = 5
    shape_indices = [link_masses_counter - 1]
    for i, j in self._model.kintree_table.T[1:]:
      for axis, limits in zip(joints[j].axes, joints[j].limits):
        link_masses_counter += 1
        self._joint_indices.append(link_masses_counter - 1)
        self._joint_limits.append(limits)
      shape_indices.append(link_masses_counter - 1)

    flags = pb.URDF_INITIALIZE_SAT_FEATURES
    if self.FLAG_USE_SELF_COLLISION & self._flags:
      flags |= pb.URDF_USE_SELF_COLLISION
      flags |= pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

    body_id = self._client.loadURDF(self._urdf_file,
                                    basePosition=[0.0, 0.0, 0.0],
                                    baseOrientation=[0.0, 0.0, 0.0, 1.0],
                                    useFixedBase=True,
                                    flags=flags)

    for j in shape_indices:
      self._client.changeVisualShape(body_id, j, rgbaColor=[0.0, 1.0, 0.0, 1.0])

    return body_id
