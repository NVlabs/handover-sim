"""

Derived from:
https://github.com/bryandlee/franka_pybullet/blob/7c66ad1a211a4c118bc58eb374063f7f55f60973/src/panda_gripper.py
https://github.com/liruiw/OMG-Planner/blob/dcbbb8279570cd62cf7388bf393c8b3e2d5686e5/bullet/panda_gripper.py
"""

import os


class Panda:

  def __init__(self,
               bullet_client,
               base_position=[0, 0, 0],
               base_orientation=[0, 0, 0, 1]):
    self._p = bullet_client
    self._base_position = base_position
    self._base_orientation = base_orientation

    self._init_pos = [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]

    self._position_control_gain_p = [
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
    ]
    self._position_control_gain_d = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ]
    self._max_torque = [250, 250, 250, 250, 250, 250, 250, 250, 250]

    self._body_id = None

  @property
  def body_id(self):
    return self._body_id

  def reset(self):
    if self._body_id is None:
      urdf_file = os.path.join(os.path.dirname(__file__), "..", "..",
                               "OMG-Planner", "bullet", "models", "panda",
                               "panda_gripper.urdf")
      self._body_id = self._p.loadURDF(urdf_file,
                                       basePosition=self._base_position,
                                       baseOrientation=self._base_orientation,
                                       useFixedBase=True,
                                       flags=self._p.URDF_USE_SELF_COLLISION)

      self._joint_indices = []
      for j in range(self._p.getNumJoints(self._body_id)):
        joint_info = self._p.getJointInfo(self._body_id, j)
        if joint_info[2] != self._p.JOINT_FIXED:
          self._joint_indices.append(j)

    # Reset joint states.
    for i, j in enumerate(self._joint_indices):
      self._p.resetJointState(self._body_id, j, self._init_pos[i])

    # Reset Controllers.
    self._p.setJointMotorControlArray(self._body_id,
                                      self._joint_indices,
                                      self._p.POSITION_CONTROL,
                                      forces=[0] * len(self._joint_indices))

  def set_target_positions(self, target_pos):
    self._p.setJointMotorControlArray(
        self._body_id,
        self._joint_indices,
        controlMode=self._p.POSITION_CONTROL,
        targetPositions=target_pos,
        forces=self._max_torque,
        positionGains=self._position_control_gain_p,
        velocityGains=self._position_control_gain_d)

  def get_joint_states(self):
    joint_states = self._p.getJointStates(self._body_id, self._joint_indices)
    joint_pos = [x[0] for x in joint_states]
    joint_vel = [x[1] for x in joint_states]
    return joint_pos, joint_vel
