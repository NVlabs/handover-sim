#!/bin/bash

if [ -z $1 ]; then
  echo "Result dir is required."
  exit
else
  res_dir=$1
fi

for x in $res_dir/*.npz; do
  index=$(basename $x)
  index=${index%.*}
  python examples/render_benchmark.py \
    --res_dir $res_dir \
    --index $index \
    SIM.BULLET.USE_EGL True \
    ENV.RENDER_OFFSCREEN True \
    BENCHMARK.SAVE_OFFSCREEN_RENDER True
done
