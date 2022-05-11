import argparse

from handover.benchmark_evaluator import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate benchmark.")
    parser.add_argument("--res_dir", help="Result directory produced by benchmark runner")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    evaluate(args.res_dir)


if __name__ == "__main__":
    main()
