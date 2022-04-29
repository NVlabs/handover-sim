import numpy as np
import pybullet

from handover.config import get_config_from_args
from handover.benchmark_wrapper import HandoverBenchmarkEnv

from demo_trajectory import (
    start_conf,
    traj,
    time_wait,
    time_action_repeat,
    time_close_gripper,
    back_step_size,
)

setup = "s0"
split = "train"

scene_id = 105


class Policy:
    def __init__(self, env):
        self._env = env

        self._steps_wait = int(time_wait / self._env.cfg.SIM.TIME_STEP)
        self._steps_action_repeat = int(time_action_repeat / self._env.cfg.SIM.TIME_STEP)
        self._steps_close_gripper = int(time_close_gripper / self._env.cfg.SIM.TIME_STEP)

    def forward(self, frame):
        if frame < self._steps_wait:
            action = start_conf
        elif frame < self._steps_wait + len(traj) * self._steps_action_repeat:
            i = (frame - self._steps_wait) // self._steps_action_repeat
            action = traj[i]
        elif (
            frame
            < self._steps_wait + len(traj) * self._steps_action_repeat + self._steps_close_gripper
        ):
            action = traj[-1].copy()
            action[7:9] = 0.0
        else:
            if (
                frame
                == self._steps_wait
                + len(traj) * self._steps_action_repeat
                + self._steps_close_gripper
            ):
                self._back = self.back()
            if (
                frame
                < self._steps_wait
                + len(traj) * self._steps_action_repeat
                + self._steps_close_gripper
                + len(self._back) * self._steps_action_repeat
            ):
                i = (
                    frame
                    - self._steps_wait
                    - len(traj) * self._steps_action_repeat
                    - self._steps_close_gripper
                ) // self._steps_action_repeat
                action = self._back[i]
            else:
                action = self._back[-1]

        return action

    def back(self):
        back = []
        pos = self._env.panda.body.link_state[0][self._env.panda.LINK_IND_HAND][0:3].numpy()
        vec_goal = self._env.goal_center - pos
        vec_step = vec_goal / np.linalg.norm(vec_goal) * back_step_size
        num_steps = int(np.ceil(np.linalg.norm(vec_goal) / back_step_size))
        for _ in range(num_steps):
            pos += vec_step
            conf = pybullet.calculateInverseKinematics(
                self._env.panda.body.contact_id[0], self._env.panda.LINK_IND_HAND - 1, pos
            )
            conf = np.array(conf, dtype=np.float32)
            conf[7:9] = 0.0
            back.append(conf)

        return back


def main():
    cfg = get_config_from_args()

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
