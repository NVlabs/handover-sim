import numpy as np

from handover.config import get_config_from_args
from handover.handover_env import HandoverEnv

scene_id = 105


def main():
  cfg = get_config_from_args()

  cfg.SIM.RENDER = True

  env = HandoverEnv(cfg)

  while True:
    env.reset(scene_id=scene_id)
    for _ in range(3000):
      action = np.array(cfg.ENV.PANDA_INITIAL_POSITION, dtype=np.float32)
      action += np.random.uniform(low=-1.0, high=+1.0, size=len(action))
      env.step(action)


if __name__ == '__main__':
  main()
