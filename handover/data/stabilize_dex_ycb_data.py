import argparse
import os
import numpy as np

from handover.utils.stabilizer import Stabilizer
from handover.envs.handover_env import HandoverEnv


def parse_args():
  parser = argparse.ArgumentParser(description='Stabilize DexYCB data.')
  parser.add_argument('--num_batches',
                      help='Number of batches',
                      default=1,
                      type=int)
  parser.add_argument('--batch_index', help='Batch index', default=1, type=int)
  args = parser.parse_args()
  return args


class HandoverEnvStabilizerWrapper(HandoverEnv):

  def __init__(self, is_render=False):
    super().__init__(is_render=is_render,
                     is_control_object=False,
                     is_load_panda=False)
    self._is_render = is_render

    # Set the order using a volume-based heuristic.
    super().reset()
    obj_vol = {i: self._ycb.get_aabb_volume(i) for i in self._ycb.CLASSES}
    obj_order = sorted(obj_vol.keys(), key=lambda x: -obj_vol[x])

    # Make manual adjustments for corner cases.
    # Move 002_master_chef_can before 021_bleach_cleanser (scene 50)
    obj_order.pop(5)
    obj_order.insert(4, 1)
    # Move 052_extra_large_clamp before 024_bowl (scene 69)
    obj_order.pop(7)
    obj_order.insert(6, 20)
    # Move 008_pudding_box before 011_banana (scene 146)
    obj_order.pop(12)
    obj_order.insert(11, 7)
    # Move 010_potted_meat_can before 024_bowl (scene 161)
    obj_order.pop(13)
    obj_order.insert(7, 9)
    # Move 007_tuna_fish_can before 037_scissors (scene 183)
    obj_order.pop(17)
    obj_order.insert(15, 6)
    # Move 004_sugar_box before 024_bowl (scene 663)
    obj_order.pop(11)
    obj_order.insert(8, 3)
    # Move 009_gelatin_box before 037_scissors (scene 780)
    obj_order.pop(17)
    obj_order.insert(16, 8)

    self._obj_order = obj_order

  @property
  def is_render(self):
    return self._is_render

  @property
  def time_step(self):
    return self._time_step

  @property
  def obj_ids(self):
    return self._ycb.ycb_ids

  @property
  def obj_pose(self):
    return self._ycb.pose

  @property
  def obj_names(self):
    return self._ycb.CLASSES

  @property
  def obj_order(self):
    return self._obj_order

  def reset(self, obj_pose=None):
    super().reset(scene_id=self._scene_id, pose=obj_pose)

  def step(self):
    super().step(None)

  def get_base_state(self, obj_id):
    return self._ycb.get_base_state(obj_id, is_table_frame=True)

  def get_contact_points(self, obj_id):
    return self._ycb.get_contact_points(obj_id)

  def set_collision_filter(self, obj_id, collision_id):
    return self._ycb.set_collision_filter(obj_id, collision_id)

  def set_scene(self, scene_id):
    self._scene_id = scene_id
    super().reset(hard_reset=True, scene_id=self._scene_id)


def main():
  args = parse_args()

  env = HandoverEnvStabilizerWrapper()

  stabilizer = Stabilizer()

  stable_dir = os.path.join(os.path.dirname(__file__), "dex-ycb-stabilized")
  os.makedirs(stable_dir, exist_ok=True)

  sid = int(env.num_scenes / args.num_batches) * (args.batch_index - 1)
  eid = int(env.num_scenes / args.num_batches) * args.batch_index

  for i, scene_id in enumerate(range(sid, eid), 1):
    print('=' * 20)
    print('{:04d}/{:04d}: scene {}'.format(i, eid - sid, scene_id))
    env.set_scene(scene_id)

    pose, delta = stabilizer.run(env)

    stabilizer.test(env, pose)

    stable_file = os.path.join(stable_dir, '{:04d}.npz'.format(scene_id))
    np.savez_compressed(stable_file, pose=pose, delta=delta)


if __name__ == '__main__':
  main()
