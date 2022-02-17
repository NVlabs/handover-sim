import os
import numpy as np
import easysim

from mano_pybullet.hand_model import HandModel45


# TODO(ywchao): add ground-truth motions.
class MANO:
    def __init__(self, cfg, scene, dex_ycb):
        self._cfg = cfg
        self._scene = scene
        self._dex_ycb = dex_ycb

        self._models_dir = os.path.join(os.path.dirname(__file__), "data", "mano_v1_2", "models")

        self._model = None
        self._origin = None
        self._body = None

    def reset(self, scene_id):
        scene_data = self._dex_ycb.get_scene_data(scene_id)
        self._subject = scene_data["name"].split("/")[0]
        self._mano_side = scene_data["mano_sides"][0]
        self._mano_betas = scene_data["mano_betas"][0]
        pose = scene_data["pose_m"][:, 0]

        self._sid = np.where(np.any(pose != 0, axis=1))[0][0]
        self._eid = np.where(np.any(pose != 0, axis=1))[0][-1]

        self._q = pose[:, 0:48].copy()
        self._t = pose[:, 48:51].copy()
        self._base_euler = pose[:, 51:54].copy()

        self._t[self._sid : self._eid + 1, 2] += self._cfg.ENV.TABLE_HEIGHT

        self._frame = 0
        self._num_frames = len(self._q)

        if self._frame == self._sid:
            self.make()
        else:
            self.clean()

    def make(self):
        if self._body is None:
            self._model = HandModel45(
                left_hand=self._mano_side == "left",
                models_dir=self._models_dir,
                betas=self._mano_betas,
            )
            self._origin = self._model.origins(self._mano_betas)[0]

            body = easysim.Body()
            body.name = "mano"
            body.urdf_file = os.path.join(
                os.path.dirname(__file__),
                "data",
                "assets",
                "{}_{}".format(self._subject, self._mano_side),
                "mano.urdf",
            )
            body.use_fixed_base = True
            body.initial_base_position = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            if self._cfg.SIM.SIMULATOR == "bullet":
                body.link_color = [(0.0, 1.0, 0.0, 1.0)] * 53
            body.link_collision_filter = [self._cfg.ENV.COLLISION_FILTER_MANO] * 53
            body.link_lateral_friction = [5.0] * 53
            body.link_spinning_friction = [5.0] * 53
            body.link_restitution = [0.5] * 53
            body.link_linear_damping = 10.0
            body.link_angular_damping = 10.0
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
            self._scene.add_body(body)

            self._body = body
        else:
            assert self._model.is_left_hand == (self._mano_side == "left")
            assert self._model._betas == self._mano_betas

        self._reset_from_mano(
            self._t[self._frame], self._q[self._frame], self._base_euler[self._frame]
        )

    @property
    def body(self):
        return self._body

    def _reset_from_mano(self, trans, mano_pose, base_euler):
        angles, basis = self._model.mano_to_angles(mano_pose)
        trans = trans + self._origin - basis @ self._origin
        self.body.initial_dof_position = trans.tolist() + base_euler.tolist() + angles

    def _set_target_from_mano(self, trans, mano_pose, base_euler):
        angles, basis = self._model.mano_to_angles(mano_pose)
        trans = trans + self._origin - basis @ self._origin
        self.body.dof_target_position = trans.tolist() + base_euler.tolist() + angles

    def clean(self):
        if self.body is not None:
            self._scene.remove_body(self.body)
            self._model = None
            self._origin = None
            self._body = None

    def step(self):
        self._frame += 1
        self._frame = min(self._frame, self._num_frames - 1)

        if self._frame == self._sid:
            self.make()
        if self._frame > self._sid and self._frame <= self._eid:
            self._set_target_from_mano(
                self._t[self._frame], self._q[self._frame], self._base_euler[self._frame]
            )
        if self._frame == self._eid + 1:
            self.clean()
