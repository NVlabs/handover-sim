import os
import pickle
import yaml
import numpy as np
import json

from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from scipy import interpolate


class DexYCB:
    _SUBJECTS = [
        "20200709-subject-01",
        "20200813-subject-02",
        "20200820-subject-03",
        "20200903-subject-04",
        "20200908-subject-05",
        "20200918-subject-06",
        "20200928-subject-07",
        "20201002-subject-08",
        "20201015-subject-09",
        "20201022-subject-10",
    ]
    _TIME_STEP_RAW = 0.04
    _TIME_STEP_CACHE = 0.001

    NUM_SEQUENCES = 1000

    def __init__(self, cfg, save_to_cache=False, preload_from_raw=False):
        self._cfg = cfg
        self._save_to_cache = save_to_cache
        self._preload_from_raw = preload_from_raw

        self._cache_dir = os.path.join(os.path.dirname(__file__), "data", "dex-ycb-cache")
        self._meta_file_str = os.path.join(self._cache_dir, "meta_{:03d}.json")
        self._pose_file_str = os.path.join(self._cache_dir, "pose_{:03d}.npz")

        if self._save_to_cache:
            self._scene_data = self.preload_from_raw(self._TIME_STEP_CACHE)
            self.save_to_cache()

        if self._preload_from_raw:
            self._scene_data = self.preload_from_raw(self._cfg.SIM.TIME_STEP)
        else:
            self._scene_data = {scene_id: None for scene_id in range(self.NUM_SEQUENCES)}

    def preload_from_raw(self, time_step_resample):
        print("Preloading DexYCB from raw dataset")

        # Get raw dataset dir.
        assert "DEX_YCB_DIR" in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
        raw_dir = os.environ["DEX_YCB_DIR"]

        # Load MANO model.
        mano = {}
        for k, name in zip(("right", "left"), ("RIGHT", "LEFT")):
            mano_file = os.path.join(
                os.path.dirname(__file__), "data", "mano_v1_2", "models", "MANO_{}.pkl".format(name)
            )
            with open(mano_file, "rb") as f:
                mano[k] = pickle.load(f, encoding="latin1")

        scene_data = {}
        scene_id = 0

        for n in self._SUBJECTS:
            print("{:02d}/{:02d}  {}".format(self._SUBJECTS.index(n) + 1, len(self._SUBJECTS), n))
            seq = sorted(os.listdir(os.path.join(raw_dir, n)))
            seq = [os.path.join(n, s) for s in seq]
            assert len(seq) == 100

            for name in seq:
                # Load meta.
                meta_file = os.path.join(raw_dir, name, "meta.yml")
                with open(meta_file, "r") as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)

                # Load extrinsics.
                extr_file = os.path.join(
                    raw_dir,
                    "calibration",
                    "extrinsics_{}".format(meta["extrinsics"]),
                    "extrinsics.yml",
                )
                with open(extr_file, "r") as f:
                    extr = yaml.load(f, Loader=yaml.FullLoader)
                tag_T = np.array(extr["extrinsics"]["apriltag"], dtype=np.float32).reshape(3, 4)
                tag_R = tag_T[:, :3]
                tag_t = tag_T[:, 3]
                tag_R_inv = tag_R.T
                tag_t_inv = np.matmul(tag_R_inv, -tag_t)

                # Load pose.
                pose_file = os.path.join(raw_dir, name, "pose.npz")
                pose = np.load(pose_file)

                # Process YCB pose.
                q = pose["pose_y"][:, :, :4]
                t = pose["pose_y"][:, :, 4:]

                q, t = self.transform(q, t, tag_R_inv, tag_t_inv)
                q, t = self.resample(q, t, time_step_resample)

                shape_q = q.shape
                q = q.reshape(-1, 4)
                q = Rot.from_quat(q).as_euler("XYZ").astype(np.float32)
                q = q.reshape(*shape_q[:2], 3)
                # https://math.stackexchange.com/questions/463748/getting-cumulative-euler-angle-from-a-single-quaternion
                q = np.unwrap(q, axis=0)

                pose_y = np.dstack((q, t))

                # Process MANO pose.
                mano_betas = []
                root_trans = []
                comp = []
                mean = []
                for s, c in zip(meta["mano_sides"], meta["mano_calib"]):
                    mano_calib_file = os.path.join(
                        raw_dir, "calibration", "mano_{}".format(c), "mano.yml"
                    )
                    with open(mano_calib_file, "r") as f:
                        mano_calib = yaml.load(f, Loader=yaml.FullLoader)
                    betas = mano_calib["betas"]
                    mano_betas.append(betas)
                    v = mano[s]["shapedirs"].dot(betas) + mano[s]["v_template"]
                    r = mano[s]["J_regressor"][0].dot(v)[0]
                    root_trans.append(r)
                    comp.append(mano[s]["hands_components"])
                    mean.append(mano[s]["hands_mean"])
                root_trans = np.array(root_trans, dtype=np.float32)
                comp = np.array(comp, dtype=np.float32)
                mean = np.array(mean, dtype=np.float32)

                i = np.any(pose["pose_m"] != 0.0, axis=2)

                q = pose["pose_m"][:, :, 0:3]
                t = pose["pose_m"][:, :, 48:51]

                t[i] += root_trans
                q, t = self.transform(q, t, tag_R_inv, tag_t_inv)
                t[i] -= root_trans

                p = pose["pose_m"][:, :, 3:48]
                p = np.einsum("abj,bjk->abk", p, comp) + mean
                p = p.reshape(-1, 3)
                p = Rot.from_rotvec(p).as_quat().astype(np.float32)
                p = p.reshape(-1, 1, 60)
                p[~i] = 0

                q = np.dstack((q, p))
                q, t = self.resample(q, t, time_step_resample)

                i = np.any(q != 0.0, axis=2)
                q_i = q[i]
                q_i_full = q_i.reshape(-1, 4)
                q_i_full = Rot.from_quat(q_i_full).as_rotvec().astype(np.float32)
                q_i_full = q_i_full.reshape(-1, 48)
                q = np.zeros((len(q), 1, 48), dtype=q.dtype)
                q[i] = q_i_full

                pose_m = np.dstack((q, t))

                q_i_base = q_i[:, 0:4]
                q_i_base = Rot.from_quat(q_i_base).as_euler("XYZ").astype(np.float32)
                # https://math.stackexchange.com/questions/463748/getting-cumulative-euler-angle-from-a-single-quaternion
                q_i_base = q_i_base.reshape(-1, 1, 3)
                q_i_base = np.unwrap(q_i_base, axis=0)
                q_i_base = q_i_base.reshape(-1, 3)
                base_euler = np.zeros((len(q), 1, 3), dtype=q.dtype)
                base_euler[i] = q_i_base

                pose_m = np.dstack((pose_m, base_euler))

                scene_data[scene_id] = {
                    "name": name,
                    "ycb_ids": meta["ycb_ids"],
                    "ycb_grasp_ind": meta["ycb_grasp_ind"],
                    "mano_sides": meta["mano_sides"],
                    "mano_betas": mano_betas,
                    "pose_y": pose_y,
                    "pose_m": pose_m,
                }

                scene_id += 1

        assert len(scene_data) == self.NUM_SEQUENCES

        return scene_data

    def transform(self, q, t, tag_R_inv, tag_t_inv):
        """Transforms 6D pose to tag coordinates."""
        q_trans = np.zeros((*q.shape[:2], 4), dtype=q.dtype)
        t_trans = np.zeros(t.shape, dtype=t.dtype)

        i = np.any(q != 0, axis=2) | np.any(t != 0, axis=2)
        q = q[i]
        t = t[i]

        if q.shape[1] == 4:
            R = Rot.from_quat(q).as_matrix().astype(np.float32)
        if q.shape[1] == 3:
            R = Rot.from_rotvec(q).as_matrix().astype(np.float32)
        R = np.matmul(tag_R_inv, R)
        t = np.matmul(tag_R_inv, t.T).T + tag_t_inv
        q = Rot.from_matrix(R).as_quat().astype(np.float32)

        q_trans[i] = q
        t_trans[i] = t

        return q_trans, t_trans

    # http://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Animating_Transformations.html
    # "The approach used for transformation interpolation ... by multiplying the three interpolated matrices together."
    def resample(self, q, t, time_step_resample):
        """Resample motion."""
        i = np.any(q != 0, axis=2) | np.any(t != 0, axis=2)

        q = q.reshape(*q.shape[:2], -1, 4)

        num_f, num_o, num_r, _ = q.shape

        assert time_step_resample != self._TIME_STEP_RAW
        times_key = np.arange(0, num_f * self._TIME_STEP_RAW, self._TIME_STEP_RAW)
        times_int = np.arange(0, num_f * self._TIME_STEP_RAW, time_step_resample)
        times_int = times_int[times_int <= times_key[-1]]

        q_int = np.zeros((len(times_int), num_o, num_r, 4), dtype=np.float32)
        t_int = np.zeros((len(times_int), num_o, 3), dtype=np.float32)

        for o in range(num_o):
            # TODO(ywchao): to be tested with different resample time steps; might need to cut down e_int.
            s_key = np.where(i[:, o])[0][0]
            e_key = np.where(i[:, o])[0][-1]
            s_int = int(np.round(s_key * self._TIME_STEP_RAW / time_step_resample))
            e_int = int(np.round(e_key * self._TIME_STEP_RAW / time_step_resample))

            for r in range(num_r):
                x = q[s_key : e_key + 1, o, r]
                x = Rot.from_quat(x)
                slerp = Slerp(times_key[s_key : e_key + 1], x)
                y = slerp(times_int[s_int : e_int + 1])
                y = y.as_quat().astype(np.float32)
                q_int[s_int : e_int + 1, o, r] = y

            for d in range(3):
                x = t[s_key : e_key + 1, o, d]
                f = interpolate.splrep(times_key[s_key : e_key + 1], x)
                y = interpolate.splev(times_int[s_int : e_int + 1], f)
                t_int[s_int : e_int + 1, o, d] = y

        q_int = q_int.reshape(-1, num_o, num_r * 4)

        return q_int, t_int

    def save_to_cache(self):
        print("Saving DexYCB to cache: {}".format(self._cache_dir))
        os.makedirs(self._cache_dir, exist_ok=True)

        for scene_id, data in self._scene_data.items():
            meta_file = self._meta_file_str.format(scene_id)
            if os.path.isfile(meta_file):
                print("Meta file already exists: {}".format(meta_file))
            else:
                meta = {
                    "name": data["name"],
                    "ycb_ids": data["ycb_ids"],
                    "ycb_grasp_ind": data["ycb_grasp_ind"],
                    "mano_sides": data["mano_sides"],
                    "mano_betas": data["mano_betas"],
                }
                with open(meta_file, "w") as f:
                    json.dump(meta, f)

            pose_file = self._pose_file_str.format(scene_id)
            if os.path.isfile(pose_file):
                print("Pose file already exists: {}".format(pose_file))
            else:
                pose = {
                    "pose_y": data["pose_y"],
                    "pose_m": data["pose_m"],
                }
                np.savez_compressed(pose_file, **pose)

    def get_scene_data(self, scene_id):
        if self._scene_data[scene_id] is None:
            self._scene_data[scene_id] = self.load_from_cache(scene_id)

        return self._scene_data[scene_id]

    def load_from_cache(self, scene_id):
        meta = self.load_meta_from_cache(scene_id)

        pose_file = self._pose_file_str.format(scene_id)
        pose = np.load(pose_file)

        # Resample from cache.
        pose_y = pose["pose_y"]
        pose_m = pose["pose_m"]
        if self._cfg.SIM.TIME_STEP != self._TIME_STEP_CACHE:
            assert self._cfg.SIM.TIME_STEP > self._TIME_STEP_CACHE
            ind_int = np.arange(0, len(pose_y) * self._TIME_STEP_CACHE, self._cfg.SIM.TIME_STEP)
            ind_int = np.round(ind_int / self._TIME_STEP_CACHE).astype(np.int64)
            ind_int = ind_int[ind_int < len(pose_y)]
            pose_y = pose_y[ind_int]
            pose_m = pose_m[ind_int]

        scene_data = {
            "name": meta["name"],
            "ycb_ids": meta["ycb_ids"],
            "ycb_grasp_ind": meta["ycb_grasp_ind"],
            "mano_sides": meta["mano_sides"],
            "mano_betas": meta["mano_betas"],
            "pose_y": pose_y,
            "pose_m": pose_m,
        }

        return scene_data

    def load_meta_from_cache(self, scene_id):
        meta_file = self._meta_file_str.format(scene_id)
        with open(meta_file, "r") as f:
            meta = json.load(f)
        return meta
