#!/bin/bash

# OMGPlanner
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-27_21-34-28_omg-planner_s0_test
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-27_22-16-20_omg-planner_s1_test
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-27_23-06-42_omg-planner_s2_test
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_00-53-14_omg-planner_s3_test

# Yang et al. ICRA 2021
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_08-57-34_yang-icra2021_s0_test
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_09-36-00_yang-icra2021_s1_test
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_10-17-43_yang-icra2021_s2_test
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_12-02-41_yang-icra2021_s3_test

# GA-DDPG hold
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_01-24-37_ga-ddpg-hold_s0_test
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_02-11-05_ga-ddpg-hold_s1_test
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_03-07-40_ga-ddpg-hold_s2_test
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_04-58-58_ga-ddpg-hold_s3_test

# # GA-DDPG w/o hold
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_05-31-34_ga-ddpg-wo-hold_s0_test
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_06-10-03_ga-ddpg-wo-hold_s1_test
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_06-58-17_ga-ddpg-wo-hold_s2_test
python examples/evaluate_benchmark.py \
  --res_dir=results/icra2022_results/2022-02-28_08-31-50_ga-ddpg-wo-hold_s3_test
