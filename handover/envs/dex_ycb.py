import os
import pickle
import yaml
import numpy as np
import json

from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from scipy import interpolate

_SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]
_NUM_SEQUENCES = 1000


class DexYCB():

  def __init__(self, load_cache=False):
    self._load_cache = load_cache

    assert 'DEX_YCB_DIR' in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
    self._raw_dir = os.environ['DEX_YCB_DIR']

    # TODO(ywchao): set time_step_resample from input and modify pose_file accordingly.
    self._time_step_raw = 0.04
    self._time_step_resample = 0.001

    self._cache_dir = os.path.join(os.path.dirname(__file__),
                                   "../data/dex-ycb-cache")
    self._meta_file = os.path.join(self._cache_dir, "meta.json")
    self._pose_file = os.path.join(self._cache_dir, "pose.npz")

    if self._load_cache:
      self.load_cache()
    else:
      self.load_raw()

    assert len(self._sequences) == _NUM_SEQUENCES
    self._num_scenes = len(self._sequences)

  @property
  def num_scenes(self):
    return self._num_scenes

  def load_raw(self):
    print('Loading DexYCB from raw dataset')

    # Load MANO model.
    mano = {}
    for k, name in zip(('right', 'left'), ('RIGHT', 'LEFT')):
      mano_file = os.path.join(
          os.path.dirname(__file__),
          "../data/mano_v1_2/models/MANO_{}.pkl".format(name))
      with open(mano_file, 'rb') as f:
        mano[k] = pickle.load(f, encoding='latin1')

    self._sequences = []
    self._ycb_ids = []
    self._ycb_grasp_ind = []
    self._mano_sides = []
    self._mano_betas = []
    self._pose_y = []
    self._pose_m = []

    for n in _SUBJECTS:
      print('{:02d}/{:02d}  {}'.format(
          _SUBJECTS.index(n) + 1, len(_SUBJECTS), n))
      seq = sorted(os.listdir(os.path.join(self._raw_dir, n)))
      seq = [os.path.join(n, s) for s in seq]
      assert len(seq) == 100
      self._sequences += seq

      for i, q in enumerate(seq):
        # Load meta.
        meta_file = os.path.join(self._raw_dir, q, "meta.yml")
        with open(meta_file, 'r') as f:
          meta = yaml.load(f, Loader=yaml.FullLoader)
        self._ycb_ids.append(meta['ycb_ids'])
        self._ycb_grasp_ind.append(meta['ycb_grasp_ind'])
        self._mano_sides.append(meta['mano_sides'])

        # Load extrinsics.
        extr_file = os.path.join(self._raw_dir, "calibration",
                                 "extrinsics_" + meta['extrinsics'],
                                 "extrinsics.yml")
        with open(extr_file, 'r') as f:
          extr = yaml.load(f, Loader=yaml.FullLoader)
        tag_T = np.array(extr['extrinsics']['apriltag'],
                         dtype=np.float32).reshape(3, 4)
        tag_R = tag_T[:, :3]
        tag_t = tag_T[:, 3]
        tag_R_inv = tag_R.T
        tag_t_inv = np.matmul(tag_R_inv, -tag_t)

        # Load pose.
        pose_file = os.path.join(self._raw_dir, q, "pose.npz")
        pose = np.load(pose_file)

        # Process YCB pose.
        q = pose['pose_y'][:, :, :4]
        t = pose['pose_y'][:, :, 4:]

        q, t = self.transform(q, t, tag_R_inv, tag_t_inv)
        q, t = self.resample(q, t)

        pose_y = np.dstack((q, t))
        self._pose_y.append(pose_y)

        # Process MANO pose.
        mano_betas = []
        root_trans = []
        for s, c in zip(meta['mano_sides'], meta['mano_calib']):
          mano_calib_file = os.path.join(self._raw_dir, "calibration",
                                         "mano_{}".format(c), "mano.yml")
          with open(mano_calib_file, 'r') as f:
            mano_calib = yaml.load(f, Loader=yaml.FullLoader)
          betas = mano_calib['betas']
          mano_betas.append(betas)
          v = mano[s]['shapedirs'].dot(betas) + mano[s]['v_template']
          r = mano[s]['J_regressor'][0].dot(v)[0]
          root_trans.append(r)
        root_trans = np.array(root_trans, dtype=np.float32)

        i = np.any(pose['pose_m'] != 0.0, axis=2)

        q = pose['pose_m'][:, :, 0:3]
        t = pose['pose_m'][:, :, 48:51]

        t[i] += root_trans
        q, t = self.transform(q, t, tag_R_inv, tag_t_inv)
        t[i] -= root_trans

        p = pose['pose_m'][:, :, 3:48]
        p = np.matmul(p, mano[s]['hands_components']) + mano[s]['hands_mean']
        p = p.reshape(-1, 3)
        p = Rot.from_rotvec(p).as_quat().astype(np.float32)
        p = p.reshape(-1, 1, 60)
        p[~i] = 0

        q = np.dstack((q, p))
        q, t = self.resample(q, t)

        i = np.any(q != 0.0, axis=2)
        q_i = q[i]
        q_i = q_i.reshape(-1, 4)
        q_i = Rot.from_quat(q_i).as_rotvec().astype(np.float32)
        q_i = q_i.reshape(-1, 48)
        q = np.zeros((len(q), 1, 48), dtype=q.dtype)
        q[i] = q_i

        pose_m = np.dstack((q, t))
        self._pose_m.append(pose_m)
        self._mano_betas.append(mano_betas)

  def transform(self, q, t, tag_R_inv, tag_t_inv):
    """Transforms 6D pose to tag coordinates.
    """
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
  def resample(self, q, t):
    """Resample motion.
    """
    i = np.any(q != 0, axis=2) | np.any(t != 0, axis=2)

    q = q.reshape(*q.shape[:2], -1, 4)

    num_f, num_o, num_r, _ = q.shape

    assert self._time_step_resample != self._time_step_raw
    times_key = np.arange(0, num_f * self._time_step_raw, self._time_step_raw)
    times_int = np.arange(0, num_f * self._time_step_raw,
                          self._time_step_resample)
    while times_int[-1] > times_key[-1]:
      times_int = times_int[:-1]

    q_int = np.zeros((len(times_int), num_o, num_r, 4), dtype=np.float32)
    t_int = np.zeros((len(times_int), num_o, 3), dtype=np.float32)

    for o in range(num_o):
      # TODO(ywchao): to be tested with different resample time steps; might need to cut down e_int.
      s_key = np.where(i[:, o])[0][0]
      e_key = np.where(i[:, o])[0][-1]
      s_int = int(
          np.round(s_key * self._time_step_raw / self._time_step_resample))
      e_int = int(
          np.round(e_key * self._time_step_raw / self._time_step_resample))

      for r in range(num_r):
        x = q[s_key:e_key + 1, o, r]
        x = Rot.from_quat(x)
        slerp = Slerp(times_key[s_key:e_key + 1], x)
        y = slerp(times_int[s_int:e_int + 1])
        y = y.as_quat().astype(np.float32)
        q_int[s_int:e_int + 1, o, r] = y

      for d in range(3):
        x = t[s_key:e_key + 1, o, d]
        f = interpolate.splrep(times_key[s_key:e_key + 1], x)
        y = interpolate.splev(times_int[s_int:e_int + 1], f)
        t_int[s_int:e_int + 1, o, d] = y

    q_int = q_int.reshape(-1, num_o, num_r * 4)

    return q_int, t_int

  def save_cache(self):
    print('Saving DexYCB to cache: {}'.format(self._cache_dir))
    os.makedirs(self._cache_dir, exist_ok=True)

    if os.path.isfile(self._meta_file):
      print('Meta file already exists: {}'.format(self._meta_file))
    else:
      meta = {
          'sequences': self._sequences,
          'ycb_ids': self._ycb_ids,
          'ycb_grasp_ind': self._ycb_grasp_ind,
          'mano_sides': self._mano_sides,
          'mano_betas': self._mano_betas,
      }
      with open(self._meta_file, 'w') as f:
        json.dump(meta, f)

    if os.path.isfile(self._pose_file):
      print('Pose file already exists: {}'.format(self._pose_file))
    else:
      pose = {}
      for s, y, m in zip(self._sequences, self._pose_y, self._pose_m):
        pose[s + '/pose_y'] = y
        pose[s + '/pose_m'] = m
      np.savez_compressed(self._pose_file, **pose)

  def load_cache(self):
    print('Loading DexYCB from cache:{}'.format(self._cache_dir))

    assert os.path.isfile(self._meta_file)
    with open(self._meta_file, 'r') as f:
      meta = json.load(f)
    self._sequences = meta['sequences']
    self._ycb_ids = meta['ycb_ids']
    self._ycb_grasp_ind = meta['ycb_grasp_ind']
    self._mano_sides = meta['mano_sides']
    self._mano_betas = meta['mano_betas']

    assert os.path.isfile(self._pose_file)
    pose = np.load(self._pose_file)
    self._pose_y = [pose[s + '/pose_y'] for s in self._sequences]
    self._pose_m = [pose[s + '/pose_m'] for s in self._sequences]

  def get_scene_data(self, scene_id):
    data = {
        'sequence': self._sequences[scene_id],
        'ycb_ids': self._ycb_ids[scene_id],
        'ycb_grasp_ind': self._ycb_grasp_ind[scene_id],
        'mano_sides': self._mano_sides[scene_id],
        'mano_betas': self._mano_betas[scene_id],
        'pose_y': self._pose_y[scene_id],
        'pose_m': self._pose_m[scene_id],
    }
    return data
