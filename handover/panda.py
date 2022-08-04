# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

"""

Derived from:
https://github.com/bryandlee/franka_pybullet/blob/7c66ad1a211a4c118bc58eb374063f7f55f60973/src/panda_gripper.py
https://github.com/liruiw/OMG-Planner/blob/dcbbb8279570cd62cf7388bf393c8b3e2d5686e5/bullet/panda_gripper.py
"""

import easysim
import os
import numpy as np
import torch

from scipy.spatial.transform import Rotation as Rot

from handover.transform3d import get_t3d_from_qt


class Panda:
    _URDF_FILE = os.path.join(
        os.path.dirname(__file__), "data", "assets", "franka_panda", "panda_gripper.urdf"
    )
    _RIGID_SHAPE_COUNT = 11

    LINK_IND_HAND = 8
    LINK_IND_FINGERS = (9, 10)

    def __init__(self, cfg, scene):
        self._cfg = cfg
        self._scene = scene

        body = easysim.Body()
        body.name = "panda"
        body.geometry_type = easysim.GeometryType.URDF
        body.urdf_file = self._URDF_FILE
        body.use_fixed_base = True
        body.use_self_collision = True
        body.initial_base_position = (
            self._cfg.ENV.PANDA_BASE_POSITION + self._cfg.ENV.PANDA_BASE_ORIENTATION
        )
        body.initial_dof_position = self._cfg.ENV.PANDA_INITIAL_POSITION
        body.initial_dof_velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        body.link_collision_filter = [
            self._cfg.ENV.COLLISION_FILTER_PANDA
        ] * self._RIGID_SHAPE_COUNT
        body.dof_control_mode = easysim.DoFControlMode.POSITION_CONTROL
        body.dof_position_gain = self._cfg.ENV.PANDA_POSITION_GAIN
        body.dof_velocity_gain = self._cfg.ENV.PANDA_VELOCITY_GAIN
        body.dof_max_force = self._cfg.ENV.PANDA_MAX_FORCE
        self._scene.add_body(body)
        self._body = body

    @property
    def body(self):
        return self._body

    def step(self, dof_target_position):
        self.body.dof_target_position = dof_target_position


class PandaHandCamera(Panda):
    _URDF_FILE = os.path.join(
        os.path.dirname(__file__),
        "data",
        "assets",
        "franka_panda",
        "panda_gripper_hand_camera.urdf",
    )
    _RIGID_SHAPE_COUNT = 12

    LINK_IND_CAMERA = 11

    def __init__(self, cfg, scene):
        super().__init__(cfg, scene)

        camera = easysim.Camera()
        camera.name = "panda_hand_camera"
        camera.width = 224
        camera.height = 224
        camera.vertical_fov = 90
        camera.near = 0.035
        camera.far = 2.0
        camera.position = [(0.0, 0.0, 0.0)]
        camera.orientation = [(0.0, 0.0, 0.0, 1.0)]
        self._scene.add_camera(camera)
        self._camera = camera

        # Get rotation from URDF to OpenGL view frame.
        orn = Rot.from_euler("XYZ", (-np.pi / 2, 0.0, -np.pi)).as_quat().astype(np.float32)
        self._quat_urdf_to_opengl = torch.from_numpy(orn)

        # Get deproject points before depth multiplication.
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = (
            self._camera.width
            / 2
            / np.tan(np.deg2rad(camera.vertical_fov) * self._camera.width / self._camera.height / 2)
        )
        K[1, 1] = self._camera.height / 2 / np.tan(np.deg2rad(camera.vertical_fov) / 2)
        K[0, 2] = self._camera.width / 2
        K[1, 2] = self._camera.height / 2
        K_inv = np.linalg.inv(K)
        x, y = np.meshgrid(np.arange(self._camera.width), np.arange(self._camera.height))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        ones = np.ones((self._camera.height, self._camera.width), dtype=np.float32)
        xy1s = np.stack((x, y, ones), axis=2).reshape(self._camera.width * self._camera.height, 3).T
        self._deproject_p = np.matmul(K_inv, xy1s).T

        # Get transform from hand to pinhole camera frame.
        pos = (+0.036, 0.0, +0.036)
        orn = Rot.from_euler("XYZ", (0.0, 0.0, +np.pi / 2)).as_quat().astype(np.float32)
        self._t3d_hand_to_camera = get_t3d_from_qt(orn, pos)

    def get_point_state(self, segmentation_id):
        # Get OpenGL view frame from URDF camera frame.
        pos = self.body.link_state[0, self.LINK_IND_CAMERA, 0:3]
        orn = self.body.link_state[0, self.LINK_IND_CAMERA, 3:7]
        orn = _quaternion_multiplication(orn, self._quat_urdf_to_opengl)

        # Set camera pose.
        self._camera.update_attr_array("position", torch.tensor([0]), pos)
        self._camera.update_attr_array("orientation", torch.tensor([0]), orn)

        # Render camera image.
        depth = self._camera.depth[0].numpy()
        segmentation = self._camera.segmentation[0].numpy()

        # Get point state in pinhole camera frame.
        segmentation = segmentation == segmentation_id
        point_state = (
            np.tile(depth[segmentation].reshape(-1, 1), (1, 3))
            * self._deproject_p[segmentation.ravel(), :]
        )

        # Transform point state to hand frame.
        point_state = self._t3d_hand_to_camera.transform_points(point_state)

        return point_state


def _quaternion_multiplication(q1, q2):
    q1x, q1y, q1z, q1w = torch.unbind(q1, axis=-1)
    q2x, q2y, q2z, q2w = torch.unbind(q2, axis=-1)
    return torch.stack(
        (
            q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y,
            q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x,
            q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w,
            q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z,
        ),
        axis=-1,
    )
