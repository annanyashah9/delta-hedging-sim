"""Phase 1 orchestrator: price, simulate, hedge, summarize, and save.

Run with:  python -m src.main
"""

import os

import numpy as np
import pandas as pd

from .black_scholes import bs_call_price
from .config import Params
from .hedge import delta_hedge
from .paths import simulate_gbm
from .plotting import plot_pnl_histogram

# project root is one level up from src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")


def summarize(pnl, premium):
    """Build a one-row summary DataFrame of the P&L distribution."""
    mean_pnl = float(np.mean(pnl))
    return pd.DataFrame(
        [{
            "premium": premium,
            "mean_pnl": mean_pnl,
            "std_pnl": float(np.std(pnl, ddof=1)),
            "min_pnl": float(np.min(pnl)),
            "max_pnl": float(np.max(pnl)),
            "mean_pct_of_premium": 100.0 * mean_pnl / premium,
        }]
    )


def main(params: Params = None):
    params = params or Params()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # the premium we'd charge (and receive) for the call today
    premium = float(
        bs_call_price(params.S0, params.K, params.r, params.sigma, params.T)
    )
    print(f"Black-Scholes call premium at t=0: {premium:.6f} (USD per EUR)")

    # roll out the spot paths, then hedge each one and see what we're left with
    paths = simulate_gbm(
        params.S0, params.mu, params.sigma, params.T,
        params.n_steps, params.n_paths, params.seed,
    )
    pnl = delta_hedge(paths, params.K, params.r, params.sigma, params.T, premium)

    # the number that matters: mean P&L as a fraction of the premium, ~0 if BS holds
    summary = summarize(pnl, premium)
    row = summary.iloc[0]
    print("\nHedging P&L summary over "
          f"{params.n_paths} paths ({params.n_steps} daily steps):")
    print(f"  mean : {row['mean_pnl']:+.6e}")
    print(f"  std  : {row['std_pnl']:.6e}")
    print(f"  min  : {row['min_pnl']:+.6e}")
    print(f"  max  : {row['max_pnl']:+.6e}")
    print(f"  mean as % of premium: {row['mean_pct_of_premium']:+.4f}%")

    # write out the picture and the numbers
    fig_path = os.path.join(FIGURES_DIR, "pnl_histogram.png")
    plot_pnl_histogram(pnl, premium, fig_path)

    summary.to_csv(os.path.join(RESULTS_DIR, "pnl_summary.csv"), index=False)
    pd.DataFrame({"path": np.arange(len(pnl)), "pnl": pnl}).to_csv(
        os.path.join(RESULTS_DIR, "pnl_per_path.csv"), index=False
    )

    print(f"\nSaved figure  -> {fig_path}")
    print(f"Saved results -> {os.path.join(RESULTS_DIR, 'pnl_summary.csv')}, "
          f"{os.path.join(RESULTS_DIR, 'pnl_per_path.csv')}")
    return pnl, summary


if __name__ == "__main__":
    main()
