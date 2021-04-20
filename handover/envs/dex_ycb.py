import os
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

  def load_raw(self):
    print('Loading DexYCB from raw dataset')
    self._sequences = []
    self._ycb_ids = []
    self._ycb_grasp_ind = []
    self._pose = []

    for n in _SUBJECTS:
      print('{:02d}/{:02d}  {}'.format(
          _SUBJECTS.index(n) + 1, len(_SUBJECTS), n))
      seq = sorted(os.listdir(os.path.join(self._raw_dir, n)))
      seq = [os.path.join(n, s) for s in seq]
      assert len(seq) == 100
      self._sequences += seq

      for i, q in enumerate(seq):
        meta_file = os.path.join(self._raw_dir, q, "meta.yml")
        with open(meta_file, 'r') as f:
          meta = yaml.load(f, Loader=yaml.FullLoader)
        self._ycb_ids.append(meta['ycb_ids'])
        self._ycb_grasp_ind.append(meta['ycb_grasp_ind'])

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

        pose_file = os.path.join(self._raw_dir, q, "pose.npz")
        pose = np.load(pose_file)
        q_raw = pose['pose_y'][:, :, :4]
        t_raw = pose['pose_y'][:, :, 4:]
        num_f = pose['pose_y'].shape[0]
        num_y = pose['pose_y'].shape[1]

        # Transform to tag coordinates.
        q = q_raw.reshape(-1, 4)
        t = t_raw.reshape(-1, 3)
        R = Rot.from_quat(q).as_matrix().astype(np.float32)
        R = np.matmul(tag_R_inv, R)
        t = np.matmul(tag_R_inv, t.T).T + tag_t_inv
        q = Rot.from_matrix(R).as_quat().astype(np.float32)
        q_trans = q.reshape(-1, num_y, 4)
        t_trans = t.reshape(-1, num_y, 3)

        # Resample motion.
        # http://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Animating_Transformations.html
        # "The approach used for transformation interpolation ... by multiplying the three interpolated matrices together."
        assert self._time_step_resample != self._time_step_raw
        times_key = np.arange(0, num_f * self._time_step_raw,
                              self._time_step_raw)
        times_int = np.arange(0, num_f * self._time_step_raw,
                              self._time_step_resample)
        while times_int[-1] > times_key[-1]:
          times_int = times_int[:-1]

        q_int = np.zeros((len(times_int), num_y, 4), dtype=np.float32)
        t_int = np.zeros((len(times_int), num_y, 3), dtype=np.float32)

        for o in range(num_y):
          q = q_trans[:, o]
          q = Rot.from_quat(q)
          slerp = Slerp(times_key, q)
          q = slerp(times_int)
          q = q.as_quat().astype(np.float32)
          q_int[:, o] = q

        for o in range(num_y):
          for d in range(3):
            x = t_trans[:, o, d]
            f = interpolate.splrep(times_key, x)
            y = interpolate.splev(times_int, f)
            t_int[:, o, d] = y

        pose = np.dstack((q_int, t_int))
        self._pose.append(pose)

  @property
  def num_scenes(self):
    return self._num_scenes

  def save_cache(self):
    print('Saving DexYCB to cache: {}'.format(self._cache_dir))
    os.makedirs(self._cache_dir, exist_ok=True)

    if os.path.isfile(self._meta_file):
      print('Meta file already exists: {}'.format(self._meta_file))
    else:
      meta = {
          'sequences': self._sequences,
          'ycb_ids': self._ycb_ids,
          'ycb_grasp_ind': self._ycb_grasp_ind
      }
      with open(self._meta_file, 'w') as f:
        json.dump(meta, f)

    if os.path.isfile(self._pose_file):
      print('Pose file already exists: {}'.format(self._pose_file))
    else:
      pose = {s: d for s, d in zip(self._sequences, self._pose)}
      np.savez_compressed(self._pose_file, **pose)

  def load_cache(self):
    print('Loading DexYCB from cache:{}'.format(self._cache_dir))

    assert os.path.isfile(self._meta_file)
    with open(self._meta_file, 'r') as f:
      meta = json.load(f)
    self._sequences = meta['sequences']
    self._ycb_ids = meta['ycb_ids']
    self._ycb_grasp_ind = meta['ycb_grasp_ind']

    assert os.path.isfile(self._pose_file)
    pose = np.load(self._pose_file)
    self._pose = [pose[s] for s in self._sequences]

  def get_scene_data(self, scene_id):
    data = {
        'sequence': self._sequences[scene_id],
        'ycb_ids': self._ycb_ids[scene_id],
        'pose': self._pose[scene_id],
        'ycb_grasp_ind': self._ycb_grasp_ind[scene_id],
    }
    return data
