"""

Derived from:
https://github.com/bryandlee/franka_pybullet/blob/7c66ad1a211a4c118bc58eb374063f7f55f60973/src/panda_gripper.py
https://github.com/liruiw/OMG-Planner/blob/dcbbb8279570cd62cf7388bf393c8b3e2d5686e5/bullet/panda_gripper.py
"""

import easysim
import os


class Panda:
    LINK_IND_HAND = 8
    LINK_IND_FINGERS = (9, 10)

    def __init__(self, cfg, scene):
        self._cfg = cfg
        self._scene = scene

        body = easysim.Body()
        body.name = "panda"
        body.urdf_file = os.path.join(
            os.path.dirname(__file__), "data", "assets", "franka_panda", "panda_gripper.urdf"
        )
        body.use_fixed_base = True
        body.use_self_collision = True
        body.initial_base_position = (
            self._cfg.ENV.PANDA_BASE_POSITION + self._cfg.ENV.PANDA_BASE_ORIENTATION
        )
        body.initial_dof_position = self._cfg.ENV.PANDA_INITIAL_POSITION
        body.initial_dof_velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        body.link_collision_filter = [self._cfg.ENV.COLLISION_FILTER_PANDA] * 11
        body.dof_control_mode = easysim.DoFControlMode.POSITION_CONTROL
        body.dof_position_gain = self._cfg.ENV.PANDA_POSITION_GAIN
        body.dof_velocity_gain = self._cfg.ENV.PANDA_VELOCITY_GAIN
        body.dof_max_force = self._cfg.ENV.PANDA_MAX_FORCE
        self._scene.add_body(body)
        self._body = body

    @property
    def body(self):
        return self._body

    def step(self, target):
        self.body.dof_target_position = target
