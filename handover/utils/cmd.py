import argparse

from handover.envs.config import merge_cfg_from_file, merge_cfg_from_list


def parse_config_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', help='Path to config file', dest='cfg_file')
  parser.add_argument('cfg',
                      nargs=argparse.REMAINDER,
                      help='See handover/envs/config.py for all options')
  args = parser.parse_args()
  return args


def set_config_from_args():
  args = parse_config_args()
  if args.cfg_file is not None:
    merge_cfg_from_file(args.cfg_file)
  if args.cfg is not None:
    merge_cfg_from_list(args.cfg)
