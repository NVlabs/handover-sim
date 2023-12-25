# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import numpy as np
import easysim
import os


class MANO:
    def __init__(self, cfg, scene, dex_ycb):
        self._cfg = cfg
        self._scene = scene
        self._dex_ycb = dex_ycb

        self._body = None
        self._cur_scene_id = None

    @property
    def body(self):
        return self._body

    def reset(self, scene_id):
        if scene_id != self._cur_scene_id:
            self._clean()

            scene_data = self._dex_ycb.get_scene_data(scene_id)

            self._name = "{}_{}".format(
                scene_data["name"].split("/")[0], scene_data["mano_sides"][0]
            )

            pose = scene_data["pose_m"][:, 0]
            self._sid = np.nonzero(np.any(pose != 0, axis=1))[0][0]
            self._eid = np.nonzero(np.any(pose != 0, axis=1))[0][-1]
            self._pose = pose.copy()
            self._pose[self._sid : self._eid + 1, 2] += self._cfg.ENV.TABLE_HEIGHT
            self._num_frames = len(self._pose)

            self._cur_scene_id = scene_id

        if self._cfg.ENV.YCB_MANO_START_FRAME == "first":
            self._frame = 0
        if self._cfg.ENV.YCB_MANO_START_FRAME == "last":
            self._frame = self._num_frames - 1
        if self._cfg.ENV.YCB_MANO_START_FRAME == "one_and_half_second":
            self._frame = int(np.round(1.5 / self._cfg.SIM.TIME_STEP))

        if self._frame >= self._sid and self._frame <= self._eid:
            self._make()
        else:
            self._clean()

    def _clean(self):
        if self.body is not None:
            self._scene.remove_body(self.body)
            self._body = None

    def _make(self):
        if self.body is None:
            body = easysim.Body()
            body.name = self._name
            body.geometry_type = easysim.GeometryType.URDF
            body.urdf_file = os.path.join(
                os.path.dirname(__file__),
                "data",
                "assets",
                self._name,
                "mano.urdf",
            )
            body.use_fixed_base = True
            body.initial_base_position = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            body.initial_dof_position = self._pose[self._frame]
            body.initial_dof_velocity = [0.0] * 51
            body.link_color = [(0.0, 1.0, 0.0, 1.0)] * 53
            body.link_collision_filter = [self._cfg.ENV.COLLISION_FILTER_MANO] * 53
            body.link_lateral_friction = [5.0] * 53
            body.link_spinning_friction = [5.0] * 53
            body.link_restitution = [0.5] * 53
            body.link_linear_damping = 10.0
            body.link_angular_damping = 10.0
            if self._cfg.ENV.MANO_SIMULATION_MODE == "default":
                body.dof_control_mode = easysim.DoFControlMode.POSITION_CONTROL
                body.dof_max_force = (
                    self._cfg.ENV.MANO_TRANSLATION_MAX_FORCE
                    + self._cfg.ENV.MANO_ROTATION_MAX_FORCE
                    + self._cfg.ENV.MANO_JOINT_MAX_FORCE
                )
                body.dof_position_gain = (
                    self._cfg.ENV.MANO_TRANSLATION_POSITION_GAIN
                    + self._cfg.ENV.MANO_ROTATION_POSITION_GAIN
                    + self._cfg.ENV.MANO_JOINT_POSITION_GAIN
                )
                body.dof_velocity_gain = (
                    self._cfg.ENV.MANO_TRANSLATION_VELOCITY_GAIN
                    + self._cfg.ENV.MANO_ROTATION_VELOCITY_GAIN
                    + self._cfg.ENV.MANO_JOINT_VELOCITY_GAIN
                )
            if self._cfg.ENV.MANO_SIMULATION_MODE == "disable_control_and_move_by_reset":
                body.dof_control_mode = easysim.DoFControlMode.TORQUE_CONTROL
                body.dof_force = [0.0] * 51
            self._scene.add_body(body)
            self._body = body

        if self._cfg.ENV.MANO_SIMULATION_MODE == "disable_control_and_move_by_reset":
            self._mass_zeroed = False

    def step(self, simulator):
        if self._cfg.ENV.MANO_SIMULATION_MODE == "disable_control_and_move_by_reset":
            skip_reset = self._frame == self._sid - 1 or self._frame == self._num_frames - 1

        self._frame += 1
        self._frame = min(self._frame, self._num_frames - 1)

        if self._frame == self._sid:
            self._make()
        if self._frame >= self._sid and self._frame <= self._eid:
            if self._cfg.ENV.MANO_SIMULATION_MODE == "default":
                self.body.dof_target_position = self._pose[self._frame]
            if self._cfg.ENV.MANO_SIMULATION_MODE == "disable_control_and_move_by_reset":
                if not self._mass_zeroed:
                    if self._frame == self._sid:
                        with simulator._disable_cov_rendering():
                            simulator._load_body(self._body)
                            simulator._cache_body_and_set_control_and_props(self._body)
                            simulator._set_callback_body(self._body)
                    for i in range(-1, simulator._num_links[self.body.name] - 1):
                        simulator._p.changeDynamics(
                            simulator._body_ids[self.body.name], i, mass=0.0
                        )
                    self._mass_zeroed = True
                if not skip_reset:
                    self.body.initial_dof_position = self._pose[self._frame]
                    self.body.env_ids_reset_dof_state = [0]
        if self._frame == self._eid + 1:
            self._clean()
