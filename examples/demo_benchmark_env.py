import pybullet
import numpy as np

from handover.envs.benchmark_wrapper import HandoverBenchmarkEnv

from demo_trajectory import start_conf, traj, num_action_repeat

setup = 's0'
split = 'train'

scene_id = 105


class Policy():

  def __init__(self, env):
    self._env = env

  def forward(self, frame):
    if frame < 3000:
      action = start_conf
    elif frame < 3000 + len(traj) * num_action_repeat:
      i = (frame - 3000) // num_action_repeat
      action = traj[i]
    elif frame < 3000 + len(traj) * num_action_repeat + 200:
      action = traj[-1].copy()
      action[-2:] = 0.0
    else:
      if frame == 3000 + len(traj) * num_action_repeat + 200:
        self._back = self.back()
      if frame < 3000 + len(traj) * num_action_repeat + 200 + len(
          self._back) * num_action_repeat:
        i = (frame - 3000 - len(traj) * num_action_repeat -
             200) // num_action_repeat
        action = self._back[i]
      else:
        action = self._back[-1]

    return action

  def back(self):
    back = []
    pos = pybullet.getLinkState(self._env._panda.body_id,
                                self._env._panda.LINK_ID_HAND)[4]
    pos = np.array(pos, dtype=np.float32)
    vec = self._env.goal_center - pos
    step = (vec / np.linalg.norm(vec)) * 0.03
    num_steps = int(np.ceil(np.linalg.norm(vec) / 0.03))
    for i in range(num_steps):
      pos += step
      conf = np.array(pybullet.calculateInverseKinematics(
          self._env._panda.body_id, self._env._panda.LINK_ID_HAND, pos),
                      dtype=np.float32)
      conf[-2:] = 0.0
      back.append(conf)

    return back


def main():
  env = HandoverBenchmarkEnv(setup, split, is_render=True, is_draw_goal=True)

  pi = Policy(env)

  while True:
    env.reset(scene_id=scene_id)
    frame = 0

    while True:
      action = pi.forward(frame)
      _, _, done, info = env.step(action)

      frame += 1
      if done:
        print('Done. Status: {:d}'.format(info['status']))
        break


if __name__ == '__main__':
  main()
