import numpy as np

from handover.cmd import set_config_from_args
from handover.handover_env import HandoverEnv

scene_id = 105


def main():
  set_config_from_args()

  env = HandoverEnv(is_render=True)

  while True:
    env.reset(scene_id)
    for _ in range(3000):
      action = np.array(env._panda._init_pos, dtype=np.float32)
      action += np.random.uniform(low=-1.0, high=+1.0, size=len(action))
      env.step(action)


if __name__ == '__main__':
  main()
