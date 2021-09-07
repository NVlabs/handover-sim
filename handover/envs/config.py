"""

Derived from:
https://github.com/rbgirshick/fast-rcnn/blob/388cc7c337f0ea74af56303606e5fb522ea3ec77/lib/fast_rcnn/config.py
https://github.com/facebookresearch/Detectron/blob/6efa991ae6d3f87abd3b1f475e3b98cbd37d25ca/detectron/core/config.py
"""

import yaml

from easydict import EasyDict as edict
from ast import literal_eval

__C = edict()

cfg = __C

# Environment parameters.
__C.ENV = edict()

__C.ENV.TIME_STEP = 0.001

__C.ENV.RELEASE_FORCE_THRESH = 0.0
__C.ENV.RELEASE_TIME_THRESH = 0.2

__C.ENV.TABLE_BASE_POSITION = (0.61, 0.28, 0.0)
__C.ENV.TABLE_BASE_ORIENTATION = (0, 0, 0, 1)
__C.ENV.TABLE_HEIGHT = 0.92

__C.ENV.PANDA_BASE_POSITION = (0.61, -0.50, 0.875)
__C.ENV.PANDA_BASE_ORIENTATION = (0.0, 0.0, 0.7071068, 0.7071068)

__C.ENV.YCB_TRANSLATION_GAIN_P = (0.2,) * 3
__C.ENV.YCB_TRANSLATION_GAIN_D = (1.0,) * 3
__C.ENV.YCB_TRANSLATION_FORCE = (50.0,) * 3
__C.ENV.YCB_ROTATION_GAIN_P = 1.0
__C.ENV.YCB_ROTATION_GAIN_D = 0.0
__C.ENV.YCB_ROTATION_FORCE = (5.0,) * 3

__C.ENV.COLLISION_ID_TABLE = 2**0
__C.ENV.COLLISION_ID_YCB = lambda i: 2**i
__C.ENV.COLLISION_ID_MANO = 2**22

# Benchmark parameters.
__C.BENCHMARK = edict()

__C.BENCHMARK.CONTACT_FORCE_THRESH = 0.0

__C.BENCHMARK.GOAL_CENTER = (0.61, -0.20, 1.25)
__C.BENCHMARK.GOAL_RADIUS = 0.15
__C.BENCHMARK.GOAL_COLOR = (0.85, 0.19, 0.21, 0.5)

__C.BENCHMARK.SUCCESS_TIME_THRESH = 0.1

__C.BENCHMARK.MAX_EPISODE_TIME = 13.0


def merge_cfg_from_file(filename):
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))
  _merge_a_into_b(yaml_cfg, __C)


def merge_cfg_from_list(cfg_list):
  assert len(cfg_list) % 2 == 0
  for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = full_key.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d, 'Non-existent config key: {}'.format(full_key)
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d, 'Non-existent config key: {}'.format(full_key)
    value = _decode_cfg_value(v)
    value = _check_and_coerce_cfg_value_type(value, d[subkey], subkey, full_key)
    d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
  assert isinstance(
      a, edict), '`a` (cur type {}) must be an instance of EasyDict'.format(
          type(a))

  for k, v in a.items():
    full_key = '.'.join(stack) + '.' + k if stack is not None else k
    # a must specify keys that are in b.
    if k not in b:
      raise KeyError('Non-existent config key: {}'.format(full_key))

    v = _decode_cfg_value(v)
    v = _check_and_coerce_cfg_value_type(v, b[k], k, k)

    # Recursively merge dicts.
    if type(v) is edict:
      try:
        stack_push = [k] if stack is None else stack + [k]
        _merge_a_into_b(v, b[k], stack=stack_push)
      except:
        raise
    else:
      b[k] = v


def _decode_cfg_value(v):
  if isinstance(v, edict):
    return v
  try:
    v = literal_eval(v)
  except:
    # Handle the case when v is a string literal.
    pass
  return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
  # The types must match (with some exceptions).
  type_b = type(value_b)
  type_a = type(value_a)
  if type_a is type_b:
    return value_a

  # Exceptions.
  if isinstance(value_a, tuple) and isinstance(value_b, list):
    # EasyDict converts tuple to list.
    value_a = list(value_a)
  else:
    raise ValueError(
        'Type mismatch ({} vs. {}) with values ({} vs. {}) for config key: {}'.
        format(type_b, type_a, value_b, value_a, full_key))
  return value_a
