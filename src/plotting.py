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
