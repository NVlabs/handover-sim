# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

from handover.config import get_config_from_args
from handover.dex_ycb import DexYCB


def main():
    print("Caching DexYCB data")

    cfg = get_config_from_args()

    DexYCB(cfg, flags=DexYCB.FLAG_SAVE_TO_CACHE)

    print("Done.")


if __name__ == "__main__":
    main()
