# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import argparse

from handover.benchmark_evaluator import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate benchmark.")
    parser.add_argument("--res_dir", help="result directory produced by benchmark runner")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    evaluate(args.res_dir)


if __name__ == "__main__":
    main()
