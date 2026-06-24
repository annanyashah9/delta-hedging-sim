"""Phase 4 orchestrator: model risk from hedging stochastic vol with a constant-vol delta.

The underlying now follows Heston (stochastic variance), but the desk keeps hedging with
the constant-vol Black-Scholes delta at a single fixed sigma_hedge. The hedge is reused
unchanged from Phase 1 — only the path-generation model changes. Parameters are set so
the AVERAGE vol matches the Phase 1 baseline, isolating the effect of vol being random:
the hedging error gets a wider spread and fatter tails.

Run with:  python -m src.main_phase4
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .black_scholes import bs_call_price
from .config import Params
from .hedge import delta_hedge
from .paths import simulate_gbm, simulate_heston
from .plotting import plot_pnl_histograms

# project root is one level up from src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")

# Heston knobs (theta, v0, sigma_hedge are derived from the Phase 1 sigma below).
KAPPA = 3.0    # mean-reversion speed
XI = 0.3       # vol-of-vol (the source of the fat tails)
RHOS = [0.0, -0.7]   # symmetric, then the realistic leverage case


def realized_vol(paths, T):
    """Annualized realized vol of the log-returns, averaged across paths (a sanity check)."""
    log_ret = np.diff(np.log(paths), axis=1)
    dt = T / (paths.shape[1] - 1)
    return float(np.mean(np.std(log_ret, axis=1, ddof=1)) / np.sqrt(dt))


def tail_stats(pnl):
    """Spread and tail metrics for a P&L distribution."""
    return {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)),
        "p5": float(np.percentile(pnl, 5)),
        "p95": float(np.percentile(pnl, 95)),
        "skew": float(skew(pnl)),
        "excess_kurtosis": float(kurtosis(pnl, fisher=True)),
        "worst_1pct": float(np.percentile(pnl, 1)),   # 1st-percentile (worst-case) loss
    }


def main(params: Params = None):
    params = params or Params()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Fair comparison: long-run vol = Phase 1 vol, start at the long-run level, and price /
    # hedge at that same level. So the premium equals Phase 1's and only randomness is new.
    sigma_hedge = params.sigma
    theta = sigma_hedge**2
    v0 = theta
    premium = float(
        bs_call_price(params.S0, params.K, params.r, sigma_hedge, params.T)
    )
    print(f"sigma_hedge = {sigma_hedge:.2f}, theta = {theta:.4f}, v0 = {v0:.4f}")
    print(f"BS premium at t=0 = {premium:.6f} (USD per EUR)")
    print(f"Heston: kappa = {KAPPA}, xi = {XI}; Feller 2*kappa*theta = "
          f"{2*KAPPA*theta:.3f} vs xi^2 = {XI**2:.3f}\n")

    results = []

    # --- Phase 1 baseline: constant-vol GBM, hedged at sigma_hedge ---
    base_paths = simulate_gbm(
        params.S0, params.mu, sigma_hedge, params.T,
        params.n_steps, params.n_paths, params.seed,
    )
    base_pnl = delta_hedge(
        base_paths, params.K, params.r, sigma_hedge, params.T, premium
    )
    results.append({"label": "baseline (constant vol)", "pnl": base_pnl,
                    "realized_vol": realized_vol(base_paths, params.T), **tail_stats(base_pnl)})

    # --- Heston, hedged with the SAME constant-vol delta ---
    for rho in RHOS:
        paths = simulate_heston(
            params.S0, params.mu, v0, KAPPA, theta, XI, rho, params.T,
            params.n_steps, params.n_paths, params.seed,
        )
        pnl = delta_hedge(paths, params.K, params.r, sigma_hedge, params.T, premium)
        results.append({"label": f"Heston (rho={rho:+.1f})", "pnl": pnl,
                        "realized_vol": realized_vol(paths, params.T), **tail_stats(pnl)})

    # ratios vs the baseline (first row)
    base = results[0]
    for d in results:
        d["std_ratio_vs_baseline"] = d["std_pnl"] / base["std_pnl"]
        d["worst1pct_ratio_vs_baseline"] = d["worst_1pct"] / base["worst_1pct"]

    cols = ["label", "realized_vol", "mean_pnl", "std_pnl", "p5", "p95",
            "skew", "excess_kurtosis", "worst_1pct", "std_ratio_vs_baseline",
            "worst1pct_ratio_vs_baseline"]
    summary = pd.DataFrame(results)[cols]
    print("Phase 1 baseline vs Phase 4 Heston (hedged with constant-vol delta):")
    print(summary.to_string(index=False))

    # one-line tail-thickening summary (use the leverage case as the headline)
    lev = results[-1]
    print(f"\nTail thickening (Heston rho={RHOS[-1]:+.1f} vs baseline): "
          f"std x{lev['std_ratio_vs_baseline']:.2f}, "
          f"worst-1% loss x{lev['worst1pct_ratio_vs_baseline']:.2f}")

    # save artifacts
    hist_path = os.path.join(FIGURES_DIR, "pnl_heston_vs_baseline.png")
    plot_pnl_histograms(
        results, premium, hist_path,
        title=("Model risk: stochastic-vol paths hedged with a constant-vol delta\n"
               f"short EUR/USD call, hedged at sigma = {sigma_hedge:.2f}, "
               f"{params.n_paths} paths"),
    )
    summary_path = os.path.join(RESULTS_DIR, "phase4_model_risk_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved figure  -> {hist_path}")
    print(f"Saved results -> {summary_path}")
    return summary


if __name__ == "__main__":
    main()
