"""Phase 2 orchestrator: how the hedging error shrinks as we rebalance more often.

We hedge the SAME 1000 underlying paths at four frequencies and check that the
hedging-error std scales like sqrt(rebalance interval) — a slope near 0.5 on a
log-log plot.

Run with:  python -m src.main_phase2
"""

import os

import numpy as np
import pandas as pd

from .black_scholes import bs_call_price
from .config import Params
from .hedge import delta_hedge
from .paths import simulate_gbm
from .plotting import plot_error_scaling, plot_pnl_histograms

# project root is one level up from src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")

# Finest resolution we simulate: twice-daily over a 252-day year.
N_FINE = 504
STEPS_PER_DAY = 2  # so one fine step = 0.5 trading days

# (label, stride in fine steps, nominal interval in trading days). Ordered coarse -> fine.
FREQUENCIES = [
    ("monthly", 42, 21.0),
    ("weekly", 10, 5.0),
    ("daily", 2, 1.0),
    ("twice-daily", 1, 0.5),
]


def make_hedge_steps(n_fine, stride):
    """Column indices to rebalance on: every `stride`-th step, always including expiry.

    For strides that don't divide n_fine evenly (weekly: 504/10), the last interval is a
    short stub. We force the terminal step in so liquidation lands on the true S_T.
    """
    steps = list(range(0, n_fine, stride))
    if steps[-1] != n_fine:
        steps.append(n_fine)
    return np.array(steps)


def main(params: Params = None):
    params = params or Params()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # the call premium — same contract as Phase 1, priced once
    premium = float(
        bs_call_price(params.S0, params.K, params.r, params.sigma, params.T)
    )
    print(f"Black-Scholes call premium at t=0: {premium:.6f} (USD per EUR)")

    # Simulate the fine paths ONCE. Every frequency hedges a subset of these same
    # columns, so they all see an identical underlying — frequency is the only variable.
    paths = simulate_gbm(
        params.S0, params.mu, params.sigma, params.T,
        N_FINE, params.n_paths, params.seed,
    )

    results = []
    for label, stride, interval_days in FREQUENCIES:
        hedge_steps = make_hedge_steps(N_FINE, stride)
        pnl = delta_hedge(
            paths, params.K, params.r, params.sigma, params.T, premium,
            hedge_steps=hedge_steps,
        )
        results.append({
            "frequency": label,
            "label": label,        # used by the overlaid-histogram plot helper
            "interval_days": interval_days,
            "n_intervals": len(hedge_steps) - 1,
            "mean_pnl": float(np.mean(pnl)),
            "std_pnl": float(np.std(pnl, ddof=1)),
            "std_pct_of_premium": 100.0 * float(np.std(pnl, ddof=1)) / premium,
            "pnl": pnl,
        })

    summary = pd.DataFrame(results)[
        ["frequency", "interval_days", "n_intervals",
         "mean_pnl", "std_pnl", "std_pct_of_premium"]
    ]
    print("\nHedging error by rebalancing frequency "
          f"({params.n_paths} paths, same underlying):")
    print(summary.to_string(index=False))

    # The headline: fit log(std) = a + b * log(interval). The sqrt-law predicts b ~ 0.5.
    intervals = np.array([r["interval_days"] for r in results])
    stds = np.array([r["std_pnl"] for r in results])
    slope, intercept = np.polyfit(np.log(intervals), np.log(stds), 1)
    print(f"\nLog-log fit of std vs interval: slope = {slope:.3f} "
          f"(sqrt scaling -> 0.5)")

    # save artifacts
    hist_path = os.path.join(FIGURES_DIR, "pnl_histograms_by_frequency.png")
    scaling_path = os.path.join(FIGURES_DIR, "hedging_error_scaling.png")
    plot_pnl_histograms(results, premium, hist_path)
    plot_error_scaling(intervals, stds, slope, intercept, scaling_path)

    summary_path = os.path.join(RESULTS_DIR, "phase2_frequency_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved figures -> {hist_path}, {scaling_path}")
    print(f"Saved results -> {summary_path}")
    return summary, slope


if __name__ == "__main__":
    main()
