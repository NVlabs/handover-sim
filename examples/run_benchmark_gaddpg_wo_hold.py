from handover.config import get_config_from_args
from handover.benchmark_runner import BenchmarkRunner

from run_benchmark_gaddpg_hold import GADDPGPolicy


class GADDPGwoHoldPolicy(GADDPGPolicy):
    def __init__(self, cfg):
        super().__init__(cfg, time_wait=0.0)

    @property
    def name(self):
        return "ga-ddpg-wo-hold"


def main():
    cfg = get_config_from_args()

    policy = GADDPGwoHoldPolicy(cfg)

    benchmark_runner = BenchmarkRunner(cfg)
    benchmark_runner.run(policy)


if __name__ == "__main__":
    main()
