"""

Derived from:
https://github.com/facebookresearch/pytorch3d/blob/bcee361d048f14b3d1fbfa2c3e498d64c06a7612/pytorch3d/transforms/transform3d.py
"""

import numpy as np

from scipy.spatial.transform import Rotation as Rot


class Transform3D:
    def __init__(self, dtype=np.float32, matrix=None):
        if matrix is None:
            self._matrix = np.eye(4, dtype=dtype).reshape(1, 4, 4)
        else:
            if matrix.ndim not in (2, 3):
                raise ValueError('"matrix" has to be a 2- or a 3-dimensional array.')
            if matrix.shape[-2] != 4 or matrix.shape[-1] != 4:
                raise ValueError('"matrix" has to be a array of shape (minibatch, 4, 4)')
            # Set dtype from matrix.
            dtype = matrix.dtype
            self._matrix = matrix.reshape(-1, 4, 4)

        self.dtype = dtype

    def get_matrix(self):
        return np.copy(self._matrix)

    def _get_matrix_inverse(self):
        return np.linalg.inv(self._matrix)

    def inverse(self):
        tinv = Transform3D(dtype=self.dtype)

        # self._get_matrix_inverse() implements efficient inverse of self._matrix.
        tinv._matrix = self._get_matrix_inverse()

        return tinv

    def transform_points(self, points):
        points_batch = np.copy(points)
        if points_batch.ndim == 2:
            points_batch = points_batch[None]
        if points_batch.ndim != 3:
            msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % repr(points.shape))

        N, P, _ = points_batch.shape
        ones = np.ones((N, P, 1), dtype=points.dtype)
        points_batch = np.dstack([points_batch, ones])

        composed_matrix = self.get_matrix()
        points_out = _broadcast_bmm(points_batch, composed_matrix)
        denom = points_out[..., 3:]
        points_out = points_out[..., :3] / denom

        # When transform is (1, 4, 4) and points is (P, 3) return points_out of shape (P, 3).
        if points_out.shape[0] == 1 and points.ndim == 2:
            points_out = points_out.reshape(points.shape)

        return points_out


def _broadcast_bmm(a, b):
    if len(a) != len(b):
        if not ((len(a) == 1) or (len(b) == 1)):
            msg = "Expected batch dim for bmm to be equal to 1; got %r, %r"
            raise ValueError(msg % (a.shape, b.shape))
        if len(a) == 1:
            a = np.broadcast_to(a, (len(b), *a.shape[1:]))
        if len(b) == 1:
            b = np.broadcast_to(b, (len(a), *b.shape[1:]))
    return np.einsum("bij,bjk->bik", a, b)


def get_t3d_from_qt(q, t, dtype=np.float32):
    R = Rot.from_quat(q).as_matrix().astype(dtype)
    t = np.asanyarray(t, dtype=dtype)
    mat = np.eye(4, dtype=dtype)
    if R.ndim == 2:
        mat[:3, :3] = R
        mat[:3, 3] = t
        mat = mat.T
    if R.ndim == 3:
        N = len(R)
        mat = np.tile(mat.reshape(1, 4, 4), (N, 1, 1))
        mat[:, :3, :3] = R
        mat[:, :3, 3] = t
        mat = np.transpose(mat, (0, 2, 1))
    return Transform3D(matrix=mat)
