"""
run_all_experiments.py
----------------------
Master script to run all experiments sequentially.

Usage:
    python run_all_experiments.py            # Run all experiments
    python run_all_experiments.py --exp 1    # Run only experiment 1
    python run_all_experiments.py --exp 1 3  # Run experiments 1 and 3

Author: Self Project — Policy Gradient Methods for Continuous Decision Processes
Date: July 2025
"""

import argparse
import sys
import time


def run_experiment(n: int):
    t0 = time.time()
    print(f"\n{'#'*70}")
    print(f"# RUNNING EXPERIMENT {n}")
    print(f"{'#'*70}\n")

    if n == 1:
        import experiments.exp1_lqr_baseline as exp
    elif n == 2:
        import experiments.exp2_continuous_control as exp
    elif n == 3:
        import experiments.exp3_variance_reduction as exp
    elif n == 4:
        import experiments.exp4_hjb_connection as exp
    else:
        print(f"Unknown experiment: {n}")
        return

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Experiment {n} finished in {elapsed:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='*', type=int, default=[1, 2, 3, 4],
                        help='Which experiments to run (default: all)')
    args = parser.parse_args()

    print("Policy Gradient Methods for Continuous Decision Processes")
    print("Self Project — July 2025")
    print("=" * 60)

    for exp_n in sorted(args.exp):
        run_experiment(exp_n)

    print("\nAll experiments complete. Results saved in results/")
