import numpy as np
import time
import pybullet


class Stabilizer():

  def __init__(self):
    self._step_size = 0.00001

    self._init_time = 1.0
    self._static_th_trans = 0.001
    self._static_th_rot = 0.1
    self._stable_time = 1.0
    self._buffer_time = 1.0

    self._max_trials = 5000
    self._max_time = 20.0

    self._test_time = 1.0
    self._test_th_trans = 0.001
    self._test_th_rot = 1.0

  def run(self, env, verbose=False):
    """ Runs stabilizer.

    Args:
      env: An environment with the following attributes and methods:
          is_render: Whether created with GUI.
          time_step: Time step.
          obj_ids: A list of (int) object ids.
          obj_pose: A numpy array of shape [num_frames, len(obj_ids), 7]
            containing the default pose of objects. Each pose is a 7-d vector
            representing rotation in quaternion (x, y, z, w) and translation
            (x, y, z).
          obj_names: A dictionary mapping object id to name.
          obj_order: A list containing the running order of all object ids.
          reset(obj_pose): Reset function.
          step(): Step function.
          get_base_state(obj_id): A function to get an object's base state.
          get_contact_points(obj_id): A function to get an object's contact
            points.
          set_collision_filter(obj_id, collision_id): A Function to set an
            object's collision filter group and mask.
      verbose: Whether to print the run details.

    Returns:
      pose_stable: A numpy array of the same size as env.pose containing the
        stable pose of objects.
    """
    init_steps = int((1 / env.time_step) * self._init_time)
    stable_steps = int((1 / env.time_step) * self._stable_time)
    buffer_steps = int((1 / env.time_step) * self._buffer_time)
    max_steps = int((1 / env.time_step) * self._max_time)

    pose = env.obj_pose[0].copy()
    delta = np.zeros_like(pose)

    ind = [env.obj_ids.index(i) for i in env.obj_order if i in env.obj_ids]

    for i, o in enumerate(ind):
      t_start = time.time()
      obj_id = env.obj_ids[o]
      print('-' * 7)
      print('Running object {}: {}'.format(i + 1, env.obj_names[obj_id]))

      print('Finding contact free position')
      if verbose:
        print('{:>5s}  {:^32s}  {:^32s}  {:>4s}'.format('trial', ' delta_start',
                                                        'delta_step', 'done'))
        print('-' * 79)

      trial = 1
      delta_start = np.zeros((3,), dtype=np.float32)

      while True:
        env.reset(pose)

        # Disable collision for unsolved objects.
        for j, p in enumerate(ind):
          if j <= i and np.any(pose[p] != 0):
            collision_id = -1
          else:
            collision_id = 0
          env.set_collision_filter(env.obj_ids[p], collision_id)

        env.step()

        cp = env.get_contact_points(obj_id)
        done = cp is () or max([x[9] for x in cp]) == 0
        if trial == 1 and done:
          print('Succeeded with the default pose')
        if verbose:
          delta_step = env.get_base_state(obj_id)[0][4:] - pose[o, 4:]
          print(
              '{:05d}  {:+10.7f} {:+10.7f} {:+10.7f}  {:+10.7f} {:+10.7f} {:+10.7f}  {:4d}'
              .format(trial, *delta_start, *delta_step, done))
        if done or trial == self._max_trials:
          break

        # Find the most penetrated contact point and move in the contact normal
        # direction.
        target_sd = [c[8] for c in cp]
        c = target_sd.index(min(target_sd))
        contact_n = cp[c][7]
        delta_trial = [x * self._step_size for x in contact_n]
        pose[o, 4:] += delta_trial
        delta_start += delta_trial

        trial += 1

      if trial == self._max_trials and not done:
        print('Failed: reached maximum number of trials')
        pose[o] = 0
        continue

      print('Finding stable position after free fall')
      if verbose:
        print('{:>6s}  {:^32s}  {:>4s} {:>4s} {:>4s} {:>4s}'.format(
            'step', ' delta_step', 'cont', 'stat', 'stab', 'done'),
              end='')
        for j in range(i + 1):
          print(' | {:>8s} {:>8s} {:>8s} {:>8s}'.format('v_trans', 'a_trans',
                                                        'v_rot', 'a_rot'),
                end='')
        print('')
        print('-' * (61 + 38 * (i + 1)))

      step = 2
      step_static = 0
      step_stable = 0
      is_static = False
      is_stable = False

      v_trans_pre = [(0, 0, 0) for _ in range(i + 1)]
      v_rot_pre = [(0, 0, 0) for _ in range(i + 1)]

      while True:
        env.step()

        pose_cur = []
        a_trans = []
        a_rot = []
        for j, p in enumerate(ind[:(i + 1)]):
          pos, vel = env.get_base_state(env.obj_ids[p])

          pose_cur.append(pos)
          v_trans = vel[3:]
          v_rot = vel[:3]
          a_trans.append([x - y for x, y in zip(v_trans, v_trans_pre[j])])
          a_rot.append([x - y for x, y in zip(v_rot, v_rot_pre[j])])
          v_trans_pre[j] = v_trans
          v_rot_pre[j] = v_rot

        if is_static:
          if self.static_test(a_trans, a_rot):
            step_static += 1
          else:
            is_static = False
            step_static = 0
        else:
          if step >= init_steps and self.static_test(a_trans, a_rot):
            is_static = True

        if is_stable:
          step_stable += 1
        else:
          if step_static == stable_steps:
            is_stable = True

        done = step_stable == buffer_steps
        if verbose:
          delta_step = env.get_base_state(obj_id)[0][4:] - pose[o, 4:]
          is_contact = env.get_contact_points(obj_id) is not ()
          print(
              '{:06d}  {:+10.7f} {:+10.7f} {:+10.7f}  {:4d} {:4d} {:4d} {:4d}'.
              format(step, *delta_step, is_contact, is_static, is_stable, done),
              end='')
          for j in range(i + 1):
            print(' | {:8.5f} {:8.5f} {:8.5f} {:8.5f}'.format(
                np.linalg.norm(v_trans_pre[j]), np.linalg.norm(a_trans[j]),
                np.linalg.norm(v_rot_pre[j]), np.linalg.norm(a_rot[j])),
                  end='')
          print('')
        if done or step == max_steps:
          break
        step += 1
      print('Done')

      if step == max_steps and not done:
        print('Stopped: reached maximum number of steps')

      delta_trans = []
      delta_rot = []
      for j, p in enumerate(ind[:(i + 1)]):
        if j == i or np.any(pose[p] != 0):
          pose[p] = pose_cur[j]
          d_trans = pose_cur[j][4:] - env.obj_pose[0, p, 4:]
          d_rot = pybullet.multiplyTransforms(
              (0, 0, 0),
              pybullet.invertTransform((0, 0, 0), env.obj_pose[0, p, :4])[1],
              (0, 0, 0), pose_cur[j][:4])[1]
          delta[p] = np.hstack((d_rot, d_trans))
        else:
          d_trans = (0, 0, 0)
          d_tor = (0, 0, 0, 0)
        delta_trans.append(d_trans)
        delta_rot.append(d_rot)

      t_spent = time.time() - t_start
      delta_step = pose_cur[-1][4:] - pose[o, 4:]
      print('time:   {:6.2f}'.format(t_spent))
      print('trials:  {:5d}'.format(trial))
      print('steps:  {:6d}'.format(step))
      print('delta_start: {:+10.7f} {:+10.7f} {:+10.7f}'.format(*delta_start))
      print('delta_step:  {:+10.7f} {:+10.7f} {:+10.7f}'.format(*delta_step))

      norm_trans = np.linalg.norm(delta_trans[-1])
      norm_rot = np.arccos(
          np.clip(2 * np.dot(env.obj_pose[0, o, :4], pose_cur[-1][:4])**2 - 1,
                  -1, +1)) * 180 / np.pi
      print('pose delta')
      print('  trans: {:10.7f} {:10.7f} {:10.7f}             {:10.7f} m'.format(
          *delta_trans[-1], norm_trans))
      print('  rot:   {:10.7f} {:10.7f} {:10.7f} {:10.7f}  {:10.5f} deg'.format(
          *delta_rot[-1], norm_rot))

    return pose, delta

  def static_test(self, a_trans, a_rot):
    return all(
        np.linalg.norm(x) < self._static_th_trans for x in a_trans) and all(
            np.linalg.norm(x) < self._static_th_rot for x in a_rot)

  def test(self, env, pose, verbose=False):
    """ Tests stable pose.

    Args:
      env: An environment with the following attributes and methods:
          is_render: Whether created with GUI.
          time_step: Time step.
          obj_ids: A list of (int) object ids.
          obj_pose: A numpy array of shape [num_frames, len(obj_ids), 7]
            containing the default pose of objects. Each pose is a 7-d vector
            representing rotation in quaternion (x, y, z, w) and translation
            (x, y, z).
          obj_names: A dictionary mapping object id to name.
          obj_order: A list containing the running order of all object ids.
          reset(obj_pose): Reset function.
          step(): Step function.
          get_base_state(obj_id): A function to get an object's base state.
          get_contact_points(obj_id): A function to get an object's contact
            points.
          set_collision_filter(obj_id, collision_id): A Function to set an
            object's collision filter group and mask.
      pose: A numpy array of shape [len(obj_ids), 7] containing the stable pose
        to test. Each pose is a 7-d vector representing rotation in quaternion
        (x, y, z, w) and translation (x, y, z).
      verbose: Whether to print the run details.
    """
    test_step = int((1 / env.time_step) * self._test_time)

    print('-' * 7)
    print('Testing')

    env.reset(pose)

    v_trans_pre = [(0, 0, 0) for _ in env.obj_ids]
    v_rot_pre = [(0, 0, 0) for _ in env.obj_ids]

    if verbose:
      print('{:>5s}'.format('step'), end='')
      for i in env.obj_ids:
        print(' | {:^35s}'.format(env.obj_names[i]), end='')
      print('')
      print(' ' * 5, end='')
      for i in env.obj_ids:
        print(' | {:>8s} {:>8s} {:>8s} {:>8s}'.format('v_trans', 'a_trans',
                                                      'v_rot', 'a_rot'),
              end='')
      print('')
      print('-' * (5 + 38 * len(env.obj_ids)))

    if env.is_render:
      t_start = time.time()

    for step in range(test_step):
      if env.is_render:
        t_spent = time.time() - t_start
        t_sleep = max(env.time_step - t_spent, 0)
        time.sleep(t_sleep)
        t_start = time.time()

      env.step()

      if verbose:
        print('{:05d}'.format(step), end='')

      for o, i in enumerate(env.obj_ids):
        _, vel = env.get_base_state(i)
        v_trans = vel[3:]
        v_rot = vel[:3]
        a_trans = [x - y for x, y in zip(v_trans, v_trans_pre[o])]
        a_rot = [x - y for x, y in zip(v_rot, v_rot_pre[o])]
        v_trans_pre[o] = v_trans
        v_rot_pre[o] = v_rot

        if verbose:
          print(' | {:8.5f} {:8.5f} {:8.5f} {:8.5f}'.format(
              np.linalg.norm(v_trans), np.linalg.norm(a_trans),
              np.linalg.norm(v_rot), np.linalg.norm(a_rot)),
                end='')
      if verbose:
        print('')
    print('Done')

    print('pose delta')
    for o, i in enumerate(env.obj_ids):
      print('{}'.format(env.obj_names[i]))
      if np.all(pose[o] == 0):
        print('  invalid input pose')
        continue
      pos, _ = env.get_base_state(i)
      delta_trans = pos[4:] - pose[o, 4:]
      delta_rot = pybullet.multiplyTransforms((0, 0, 0),
                                              pybullet.invertTransform(
                                                  (0, 0, 0), pose[o, :4])[1],
                                              (0, 0, 0), pos[:4])[1]
      norm_trans = np.linalg.norm(delta_trans)
      norm_rot = np.arccos(
          np.clip(2 * np.dot(pose[o, :4], pos[:4])**2 - 1, -1,
                  +1)) * 180 / np.pi
      print('  trans: {:10.7f} {:10.7f} {:10.7f}             {:10.7f} m'.format(
          *delta_trans, norm_trans))
      print('  rot:   {:10.7f} {:10.7f} {:10.7f} {:10.7f}  {:10.5f} deg'.format(
          *delta_rot, norm_rot))

      if norm_trans > self._test_th_trans:
        print("Failed: norm_trans > {}".format(self._test_th_trans))
      if norm_rot > self._test_th_rot:
        print("Failed: norm_rot > {}".format(self._test_th_rot))
