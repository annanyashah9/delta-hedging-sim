"""Phase 3 orchestrator: implied vs realized volatility and the gamma P&L.

We price and hedge the short call at a fixed implied vol, but let the underlying
actually move at a different realized vol. The delta-hedged P&L then shifts off zero,
and that shift is the gamma P&L. We check the mean simulated P&L against the
theoretical gamma P&L scenario by scenario.

The volatility split is the whole point:
  - sigma_implied sets the premium AND every delta we hedge with,
  - sigma_realized enters only through the simulated GBM paths.

Run with:  python -m src.main_phase3
"""

import os

import numpy as np
import pandas as pd

from .black_scholes import bs_call_price
from .config import Params
from .hedge import delta_hedge, gamma_pnl
from .paths import simulate_gbm
from .plotting import plot_pnl_histograms, plot_pnl_vs_gamma

# project root is one level up from src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")

# (label, sigma_realized). Implied vol is fixed at params.sigma below.
SCENARIOS = [
    ("realized < implied", 0.07),   # calmer than charged -> expect profit
    ("realized = implied", 0.10),   # matched -> expect ~zero (Phase 1 sanity check)
    ("realized > implied", 0.13),   # wilder than charged -> expect loss
]


def main(params: Params = None):
    params = params or Params()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sigma_implied = params.sigma
    premium = float(
        bs_call_price(params.S0, params.K, params.r, sigma_implied, params.T)
    )
    print(f"Implied vol = {sigma_implied:.2f}, "
          f"BS premium at t=0 = {premium:.6f} (USD per EUR)\n")

    results = []
    for label, sigma_realized in SCENARIOS:
        # Same seed every scenario, so the underlying shocks are comparable; only the
        # realized vol that scales them changes. sigma_realized lives ONLY here.
        paths = simulate_gbm(
            params.S0, params.mu, sigma_realized, params.T,
            params.n_steps, params.n_paths, params.seed,
        )
        # Hedge at the implied vol — the desk only knows the vol it quoted.
        sim_pnl = delta_hedge(
            paths, params.K, params.r, sigma_implied, params.T, premium
        )
        # Theoretical gamma P&L for the same paths (gamma evaluated at implied vol).
        gpnl = gamma_pnl(
            paths, params.K, params.r, sigma_implied, params.T, sigma_realized
        )
        results.append({
            "scenario": label,
            "label": label,
            "sigma_implied": sigma_implied,
            "sigma_realized": sigma_realized,
            "mean_sim_pnl": float(np.mean(sim_pnl)),
            "mean_gamma_pnl": float(np.mean(gpnl)),
            "std_sim_pnl": float(np.std(sim_pnl, ddof=1)),
            "pnl": sim_pnl,
            "sim_pnl": sim_pnl,
            "gamma_pnl": gpnl,
        })

    summary = pd.DataFrame(results)[
        ["scenario", "sigma_implied", "sigma_realized",
         "mean_sim_pnl", "mean_gamma_pnl", "std_sim_pnl"]
    ]
    print("Mean simulated hedging P&L vs theoretical gamma P&L:")
    print(summary.to_string(index=False))

    # save artifacts
    hist_path = os.path.join(FIGURES_DIR, "pnl_histograms_by_vol_scenario.png")
    scatter_path = os.path.join(FIGURES_DIR, "pnl_vs_gamma_scatter.png")
    plot_pnl_histograms(
        results, premium, hist_path,
        title=("Delta-hedging P&L by realized-vol scenario\n"
               f"short EUR/USD call hedged at implied vol = {sigma_implied:.2f}, "
               f"{params.n_paths} paths"),
    )
    plot_pnl_vs_gamma(results, scatter_path)

    summary_path = os.path.join(RESULTS_DIR, "phase3_vol_scenario_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved figures -> {hist_path}, {scatter_path}")
    print(f"Saved results -> {summary_path}")
    return summary


if __name__ == "__main__":
    main()
