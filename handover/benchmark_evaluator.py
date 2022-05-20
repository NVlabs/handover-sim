# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import os
import gym
import logging
import sys
import glob
import numpy as np
import itertools

from tabulate import tabulate

from handover.config import get_cfg
from handover.benchmark_wrapper import EpisodeStatus, HandoverBenchmarkWrapper


def get_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def evaluate(res_dir):
    cfg_file = os.path.join(res_dir, "config.yaml")
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)

    env = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))

    log_file = os.path.join(res_dir, "evaluate.log")
    logger = get_logger(log_file)

    logger.info("Running evaluation for {}".format(res_dir))

    files = glob.glob(os.path.join(res_dir, "*.npz"))
    if len(files) != env.num_scenes:
        raise ValueError(
            "Number of .npz files ({}}) does not match the number of scenes ({})".format(
                len(files), env.num_scenes
            )
        )

    result = []
    elapsed_frame = []
    elapsed_time = []

    for idx in range(env.num_scenes):
        res_file = os.path.join(res_dir, "{:03d}.npz".format(idx))
        res = np.load(res_file)

        result.append(res["result"])
        elapsed_frame.append(res["elapsed_frame"])
        elapsed_time.append(np.sum(res["elapsed_time"]))

    result = np.array(result)
    elapsed_frame = np.array(elapsed_frame)
    elapsed_time = np.array(elapsed_time)

    assert np.all(result != 0), "Result label must not be 0"
    assert np.all(
        (result == EpisodeStatus.SUCCESS) | (result & EpisodeStatus.SUCCESS == 0)
    ), "Result label must be either SUCCESS or FAILURE"

    mask_succ = result == EpisodeStatus.SUCCESS
    mask_fail_1 = (
        result & EpisodeStatus.FAILURE_HUMAN_CONTACT == EpisodeStatus.FAILURE_HUMAN_CONTACT
    )
    mask_fail_2 = result & EpisodeStatus.FAILURE_OBJECT_DROP == EpisodeStatus.FAILURE_OBJECT_DROP
    mask_fail_3 = result & EpisodeStatus.FAILURE_TIMEOUT == EpisodeStatus.FAILURE_TIMEOUT

    result = {}
    result["num_scenes"] = env.num_scenes
    result["num_success"] = np.sum(mask_succ)
    result["success_rate"] = np.mean(mask_succ)
    result["time_exec"] = np.mean(elapsed_frame[mask_succ]) * cfg.SIM.TIME_STEP
    result["time_plan"] = np.mean(elapsed_time[mask_succ])
    result["time_total"] = result["time_exec"] + result["time_plan"]
    result["num_failure_human_contact"] = np.sum(mask_fail_1)
    result["num_failure_object_drop"] = np.sum(mask_fail_2)
    result["num_failure_timeout"] = np.sum(mask_fail_3)
    result["failure_human_contact"] = np.mean(mask_fail_1)
    result["failure_object_drop"] = np.mean(mask_fail_2)
    result["failure_timeout"] = np.mean(mask_fail_3)
    result["scene_ids_success"] = np.nonzero(mask_succ)[0]
    result["scene_ids_failure_human_contact"] = np.nonzero(mask_fail_1)[0]
    result["scene_ids_failure_object_drop"] = np.nonzero(mask_fail_2)[0]
    result["scene_ids_failure_timeout"] = np.nonzero(mask_fail_3)[0]

    tabular_data = []
    tabular_data += [
        "{:.2f} ({:3d}/{:3d})".format(
            100 * result["success_rate"], result["num_success"], result["num_scenes"]
        )
    ]
    tabular_data += [result["time_exec"]]
    tabular_data += [result["time_plan"]]
    tabular_data += [result["time_total"]]
    tabular_data += [
        "{:.2f} ({:3d}/{:3d})".format(
            100 * result["failure_human_contact"],
            result["num_failure_human_contact"],
            result["num_scenes"],
        )
    ]
    tabular_data += [
        "{:.2f} ({:3d}/{:3d})".format(
            100 * result["failure_object_drop"],
            result["num_failure_object_drop"],
            result["num_scenes"],
        )
    ]
    tabular_data += [
        "{:.2f} ({:3d}/{:3d})".format(
            100 * result["failure_timeout"], result["num_failure_timeout"], result["num_scenes"]
        )
    ]
    tabular_data = [tabular_data]
    metrics = ["(%)", "exec", "plan", "total", "hand contact", "object drop", "timeout"]
    metrics[0] = "{:^{}s}".format(metrics[0], max(len(metrics[0]), 12))
    table = tabulate(
        tabular_data,
        headers=metrics,
        tablefmt="pipe",
        floatfmt=".3f",
        numalign="center",
        stralign="center",
    )
    idx = [i for i, x in enumerate(table) if x == "|"]
    metrics = [
        "{:^{}s}".format("success rate", idx[1] - idx[0] - 5),
        "{:^{}s}".format("mean accum time (s)", idx[4] - idx[1] - 5),
        "{:^{}s}".format("failure (%)", idx[7] - idx[4] - 5),
    ]
    header = tabulate(None, headers=metrics, tablefmt="pipe", stralign="center")
    header = header.split("\n")[0]
    table = header + "\n" + table

    logger.info("Evaluation results: \n" + table)

    logger.info("Printing scene ids")
    n_cols = 20
    tabular_data = ["{:3d}".format(x) for x in result["scene_ids_success"]]
    tabular_data = itertools.zip_longest(*[tabular_data[i::n_cols] for i in range(n_cols)])
    table = tabulate(tabular_data)
    logger.info("Success ({} scenes): \n".format(result["num_success"]) + table)
    tabular_data = ["{:3d}".format(x) for x in result["scene_ids_failure_human_contact"]]
    tabular_data = itertools.zip_longest(*[tabular_data[i::n_cols] for i in range(n_cols)])
    table = tabulate(tabular_data)
    logger.info(
        "Failure - hand contact ({} scenes): \n".format(result["num_failure_human_contact"]) + table
    )
    tabular_data = ["{:3d}".format(x) for x in result["scene_ids_failure_object_drop"]]
    tabular_data = itertools.zip_longest(*[tabular_data[i::n_cols] for i in range(n_cols)])
    table = tabulate(tabular_data)
    logger.info(
        "Failure - object drop ({} scenes): \n".format(result["num_failure_object_drop"]) + table
    )
    tabular_data = ["{:3d}".format(x) for x in result["scene_ids_failure_timeout"]]
    tabular_data = itertools.zip_longest(*[tabular_data[i::n_cols] for i in range(n_cols)])
    table = tabulate(tabular_data)
    logger.info("Failure - timeout ({} scenes): \n".format(result["num_failure_timeout"]) + table)

    logger.info("Evaluation complete.")

    return result
