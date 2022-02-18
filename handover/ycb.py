import easysim
import os


# TODO(ywchao): add ground-truth motions.
class YCB:
    _CLASSES = {
        1: "002_master_chef_can",
        2: "003_cracker_box",
        3: "004_sugar_box",
        4: "005_tomato_soup_can",
        5: "006_mustard_bottle",
        6: "007_tuna_fish_can",
        7: "008_pudding_box",
        8: "009_gelatin_box",
        9: "010_potted_meat_can",
        10: "011_banana",
        11: "019_pitcher_base",
        12: "021_bleach_cleanser",
        13: "024_bowl",
        14: "025_mug",
        15: "035_power_drill",
        16: "036_wood_block",
        17: "037_scissors",
        18: "040_large_marker",
        20: "052_extra_large_clamp",
        21: "061_foam_brick",
    }

    def __init__(self, cfg, scene, dex_ycb):
        self._cfg = cfg
        self._scene = scene
        self._dex_ycb = dex_ycb

        self._bodies = {}

    def reset(self, scene_id):
        scene_data = self._dex_ycb.get_scene_data(scene_id)
        self._ycb_ids = scene_data["ycb_ids"]
        self._ycb_grasp_ind = scene_data["ycb_grasp_ind"]
        pose = scene_data["pose_y"]

        self._q = pose[:, :, 0:3].copy()
        self._t = pose[:, :, 3:6].copy()

        self._t[:, :, 2] += self._cfg.ENV.TABLE_HEIGHT

        self._frame = 0
        self._num_frames = len(self._q)
        self._released = False

        if self._bodies == {}:
            for i in self._ycb_ids:
                body = easysim.Body()
                body.name = f"ycb_{i:02d}"
                body.urdf_file = os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    "assets",
                    self._CLASSES[i],
                    "model_normalized.urdf",
                )
                body.use_fixed_base = True
                body.initial_base_position = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

                q = self._q[self._frame, self._ycb_ids.index(i)]
                t = self._t[self._frame, self._ycb_ids.index(i)]
                body.initial_dof_position = t.tolist() + q.tolist()
                body.initial_dof_velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                link_collision_filter = [
                    self._cfg.ENV.COLLISION_FILTER_YCB[[*self._CLASSES].index(i)]
                ] * 7
                if i == self._ycb_ids[self._ycb_grasp_ind]:
                    body.link_collision_filter = [link_collision_filter] * self._cfg.SIM.NUM_ENVS
                else:
                    body.link_collision_filter = link_collision_filter

                body.dof_control_mode = easysim.DoFControlMode.POSITION_CONTROL

                dof_max_force = (
                    self._cfg.ENV.YCB_TRANSLATION_MAX_FORCE + self._cfg.ENV.YCB_ROTATION_MAX_FORCE
                )
                if i == self._ycb_ids[self._ycb_grasp_ind]:
                    body.dof_max_force = [dof_max_force] * self._cfg.SIM.NUM_ENVS
                else:
                    body.dof_max_force = dof_max_force

                body.dof_position_gain = (
                    self._cfg.ENV.YCB_TRANSLATION_POSITION_GAIN
                    + self._cfg.ENV.YCB_ROTATION_POSITION_GAIN
                )
                body.dof_velocity_gain = (
                    self._cfg.ENV.YCB_TRANSLATION_VELOCITY_GAIN
                    + self._cfg.ENV.YCB_ROTATION_VELOCITY_GAIN
                )
                self._scene.add_body(body)

                self._bodies[i] = body
        else:
            assert [*self._bodies.keys()] == self._ycb_ids

            link_collision_filter = [
                self._cfg.ENV.COLLISION_FILTER_YCB[
                    [*self._CLASSES].index(self._ycb_ids[self._ycb_grasp_ind])
                ]
            ] * 7
            self.grasped_body.link_collision_filter = [
                link_collision_filter
            ] * self._cfg.SIM.NUM_ENVS

            dof_max_force = (
                self._cfg.ENV.YCB_TRANSLATION_MAX_FORCE + self._cfg.ENV.YCB_ROTATION_MAX_FORCE
            )
            self.grasped_body.dof_max_force = [dof_max_force] * self._cfg.SIM.NUM_ENVS

    @property
    def released(self):
        return self._released

    @property
    def grasped_body(self):
        return self._bodies[self._ycb_ids[self._ycb_grasp_ind]]

    @property
    def non_grasped_bodies(self):
        return [self._bodies[i] for i in self._ycb_ids if i != self._ycb_ids[self._ycb_grasp_ind]]

    def clean(self):
        for i in [*self._bodies][::-1]:
            self._scene.remove_body(self._bodies[i])
            del self._bodies[i]

    def step(self):
        self._frame += 1
        self._frame = min(self._frame, self._num_frames - 1)

        for i in self._ycb_ids:
            if self.released and i == self._ycb_ids[self._ycb_grasp_ind]:
                continue
            q = self._q[self._frame, self._ycb_ids.index(i)]
            t = self._t[self._frame, self._ycb_ids.index(i)]
            self._bodies[i].dof_target_position = t.tolist() + q.tolist()

    def release(self):
        dof_max_force = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.grasped_body.dof_max_force = [dof_max_force] * self._cfg.SIM.NUM_ENVS

        link_collision_filter = [self._cfg.ENV.COLLISION_FILTER_YCB_RELEASE] * 7
        self.grasped_body.link_collision_filter = [link_collision_filter] * self._cfg.SIM.NUM_ENVS

        self._released = True
