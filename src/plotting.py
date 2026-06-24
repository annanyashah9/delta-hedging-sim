"""Plotting helpers for the hedging experiment."""

import matplotlib

matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
import numpy as np


def plot_pnl_histogram(pnl, premium, path):
    """Histogram of per-path hedging P&L, saved to `path`, with zero marked."""
    mean_pnl = float(np.mean(pnl))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(pnl, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)

    # zero is perfect replication; the mean line is the average error we actually got
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5,
               label="zero (perfect replication)")
    ax.axvline(mean_pnl, color="#C44E52", linestyle="-", linewidth=1.5,
               label=f"mean P&L = {mean_pnl:.2e}")

    ax.set_title(
        f"Daily delta-hedging P&L of a short EUR/USD call\n"
        f"BS premium = {premium:.6f} (USD per EUR), {len(pnl)} paths"
    )
    ax.set_xlabel("Hedging P&L at expiry (USD per EUR)")
    ax.set_ylabel("Number of paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pnl_histograms(results, premium, path):
    """Overlay the P&L histograms for several rebalancing frequencies on one axis.

    `results` is a list of dicts with keys "frequency" and "pnl" (ordered coarse -> fine).
    Outline (step) histograms on shared axes make the narrowing easy to compare directly.
    """
    # shared bins from the widest distribution (the coarsest hedge) so every histogram
    # is drawn on the same scale
    widest = max(results, key=lambda d: np.std(d["pnl"]))
    lo, hi = np.min(widest["pnl"]), np.max(widest["pnl"])
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for d in results:
        ax.hist(d["pnl"], bins=bins, histtype="step", linewidth=1.8,
                label=f"{d['frequency']} (std={np.std(d['pnl']):.2e})")

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="zero")
    ax.set_title(
        "Delta-hedging P&L by rebalancing frequency\n"
        f"short EUR/USD call, BS premium = {premium:.6f}, "
        f"{len(results[0]['pnl'])} paths"
    )
    ax.set_xlabel("Hedging P&L at expiry (USD per EUR)")
    ax.set_ylabel("Number of paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_scaling(intervals_days, stds, slope, intercept, path):
    """Log-log plot of hedging-error std vs rebalance interval, with the fitted line.

    A slope near 0.5 is the headline result: hedging error scales like sqrt(interval).
    `slope`/`intercept` are the fit of log(std) on log(interval) (natural logs).
    """
    intervals_days = np.asarray(intervals_days, dtype=float)
    stds = np.asarray(stds, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(intervals_days, stds, "o", markersize=9, color="#4C72B0", label="simulated std")

    # fitted line evaluated across the x-range (fit was done in natural-log space)
    x_line = np.linspace(intervals_days.min(), intervals_days.max(), 100)
    y_line = np.exp(intercept) * x_line**slope
    ax.loglog(x_line, y_line, "-", color="#C44E52",
              label=f"fit: slope = {slope:.3f} (sqrt-law -> 0.5)")

    ax.set_title("Hedging-error scaling with rebalance interval")
    ax.set_xlabel("Rebalance interval (trading days)")
    ax.set_ylabel("Std of hedging P&L (USD per EUR)")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
