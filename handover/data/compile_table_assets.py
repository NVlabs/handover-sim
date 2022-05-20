# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import os
import pybullet_data

trg_root = os.path.join(os.path.dirname(__file__), "assets")

replace = [
    ['="0 0 0.6"', '="0 0 0.9"'],
    ['="1.5 1 0.05"', '="1.77 0.73 0.04"'],
    ['="0.1 0.1 0.58"', '="0.1 0.1 0.88"'],
    ['="-0.65 -0.4 0.29"', '="-0.785 -0.265 0.44"'],
    ['="-0.65 0.4 0.29"', '="-0.785 0.265 0.44"'],
    ['="0.65 -0.4 0.29"', '="0.785 -0.265 0.44"'],
    ['="0.65 0.4 0.29"', '="0.785 0.265 0.44"'],
]


def main():
    print("Compiling table assets")

    src_dir = os.path.join(pybullet_data.getDataPath(), "table")
    trg_dir = os.path.join(trg_root, "table")
    os.makedirs(trg_dir, exist_ok=True)

    symlink_files = ("table.obj", "table.mtl", "table.png")
    for x in symlink_files:
        src_obj = os.path.join(src_dir, x)
        trg_obj = os.path.join(trg_dir, x)
        if not os.path.isfile(trg_obj):
            os.symlink(src_obj, trg_obj)
        else:
            assert os.readlink(trg_obj) == src_obj

    src_urdf = os.path.join(src_dir, "table.urdf")
    trg_urdf = os.path.join(trg_dir, "table.urdf")
    with open(src_urdf, "r") as f:
        src_lines = [line.rstrip("\n") for line in f]
    trg_lines = []
    for l in src_lines:
        for r in replace:
            if r[0] in l:
                l = l.replace(r[0], r[1])
        trg_lines.append(l)
    if not os.path.isfile(trg_urdf):
        with open(trg_urdf, "w") as f:
            for l in trg_lines:
                f.write(l + "\n")
    else:
        with open(trg_urdf, "r") as f:
            lines = [line.rstrip("\n") for line in f]
        assert lines == trg_lines

    print("Done.")


if __name__ == "__main__":
    main()
