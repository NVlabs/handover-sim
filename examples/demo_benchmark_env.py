import numpy as np
import pybullet

from handover.config import get_config_from_args
from handover.benchmark_wrapper import HandoverBenchmarkEnv

from demo_trajectory import start_conf, traj, num_action_repeat

setup = "s0"
split = "train"

scene_id = 105


class Policy:
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
            if (
                frame
                < 3000 + len(traj) * num_action_repeat + 200 + len(self._back) * num_action_repeat
            ):
                i = (frame - 3000 - len(traj) * num_action_repeat - 200) // num_action_repeat
                action = self._back[i]
            else:
                action = self._back[-1]

        return action

    def back(self):
        back = []
        pos = self._env.panda.body.link_state[0][self._env.panda.LINK_IND_HAND][0:3]
        pos = np.array(pos, dtype=np.float32)
        vec = self._env.goal_center - pos
        step = (vec / np.linalg.norm(vec)) * 0.03
        num_steps = int(np.ceil(np.linalg.norm(vec) / 0.03))
        for _ in range(num_steps):
            pos += step
            conf = pybullet.calculateInverseKinematics(
                self._env.panda.body.contact_id[0], self._env.panda.LINK_IND_HAND - 1, pos
            )
            conf = np.asanyarray(conf, dtype=np.float32)
            conf[-2:] = 0.0
            back.append(conf)

        return back


def main():
    cfg = get_config_from_args()

    cfg.SIM.RENDER = True
    cfg.BENCHMARK.SETUP = setup
    cfg.BENCHMARK.SPLIT = split
    # TODO(ywchao):
    # cfg.BENCHMARK.IS_DRAW_GOAL = True

    env = HandoverBenchmarkEnv(cfg)

    pi = Policy(env)

    while True:
        env.reset(scene_id=scene_id)
        frame = 0

        while True:
            action = pi.forward(frame)
            _, _, done, info = env.step(action)

            frame += 1
            if done:
                print("Done. Status: {:d}".format(info["status"]))
                break


if __name__ == "__main__":
    main()
