import abc
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


class SimplePolicy(abc.ABC):
    def __init__(
        self,
        env,
        time_wait=time_wait,
        time_action_repeat=time_action_repeat,
        time_close_gripper=time_close_gripper,
        back_step_size=back_step_size,
    ):
        self._env = env
        self._steps_wait = int(time_wait / self._env.cfg.SIM.TIME_STEP)
        self._steps_action_repeat = int(time_action_repeat / self._env.cfg.SIM.TIME_STEP)
        self._steps_close_gripper = int(time_close_gripper / self._env.cfg.SIM.TIME_STEP)
        self._back_step_size = back_step_size

    def reset(self):
        self._done = False
        self._done_frame = None
        self._done_action = None
        self._back = None

    def forward(self, frame):
        if frame < self._steps_wait:
            # Wait.
            action = start_conf
        elif not self._done:
            # Approach object until reaching grasp pose.
            action, done = self.plan(frame)
            if done:
                self._done = True
                self._done_frame = frame + 1
                self._done_action = action
        else:
            # Close gripper and back out.
            if frame < self._done_frame + self._steps_close_gripper:
                action = self._done_action.copy()
                action[7:9] = 0.0
            else:
                if self._back is None:
                    self._back = []
                    pos = self._env.panda.body.link_state[0][self._env.panda.LINK_IND_HAND][
                        0:3
                    ].numpy()
                    dpos_goal = self._env.goal_center - pos
                    dpos_step = dpos_goal / np.linalg.norm(dpos_goal) * self._back_step_size
                    num_steps = int(np.ceil(np.linalg.norm(dpos_goal) / self._back_step_size))
                    for _ in range(num_steps):
                        pos += dpos_step
                        conf = pybullet.calculateInverseKinematics(
                            self._env.panda.body.contact_id[0],
                            self._env.panda.LINK_IND_HAND - 1,
                            pos,
                        )
                        conf = np.array(conf, dtype=np.float32)
                        conf[7:9] = 0.0
                        self._back.append(conf)

                i = (
                    frame - self._done_frame - self._steps_close_gripper
                ) // self._steps_action_repeat
                i = min(i, len(self._back) - 1)
                action = self._back[i]

        return action

    @abc.abstractmethod
    def plan(self, frame):
        """ """


class DemoPolicy(SimplePolicy):
    def plan(self, frame):
        i = (frame - self._steps_wait) // self._steps_action_repeat
        action = traj[i]
        done = frame == self._steps_wait + len(traj) * self._steps_action_repeat - 1
        return action, done


def main():
    cfg = get_config_from_args()

    cfg.BENCHMARK.SETUP = setup
    cfg.BENCHMARK.SPLIT = split
    # TODO(ywchao):
    # cfg.BENCHMARK.IS_DRAW_GOAL = True

    env = HandoverBenchmarkEnv(cfg)

    pi = DemoPolicy(env)

    while True:
        env.reset(scene_id=scene_id)
        pi.reset()
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
