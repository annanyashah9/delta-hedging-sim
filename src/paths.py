"""Spot-path simulation: constant-vol GBM (Phase 1) and Heston stochastic vol (Phase 4)."""

import numpy as np


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed):
    """Simulate GBM spot paths via the exact log-Euler step

        S_{t+dt} = S_t * exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z).

    Drift `mu` is left free (it defaults to r elsewhere) so a later phase can show
    the hedged P&L doesn't care what it is. Returns an (n_paths, n_steps + 1) array
    with S0 in column 0 and the expiry spot in the last column.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # one shock per path per day
    Z = rng.standard_normal((n_paths, n_steps))

    # accumulate the log returns, then exponentiate back to price levels
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_returns, axis=1)

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


def simulate_heston(S0, mu, v0, kappa, theta, xi, rho, T, n_steps, n_paths, seed):
    """Simulate Heston spot paths: stochastic variance via a mean-reverting CIR process.

        dS = mu S dt + sqrt(v) S dW1
        dv = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho

    Returns a price array shaped exactly like `simulate_gbm`'s — (n_paths, n_steps + 1),
    S0 in column 0 — so the existing constant-vol `delta_hedge` consumes it unchanged. The
    variance path is internal: it shapes the prices but is never handed to the hedge.

    Variance is stepped with the FULL-TRUNCATION Euler scheme (Lord et al. 2010): the
    variance state may go slightly negative, but every use of it takes the positive part
    v_pos = max(v, 0) — under both square roots and in the mean-reversion term — so the
    sqrt never sees a negative number. This is the right scheme here because our parameters
    can violate the Feller condition, i.e. variance legitimately reaches zero.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Z1 drives the price, Z2 drives the variance, correlated by rho. Drawing two
    # independent matrices and mixing keeps the seed discipline simple and comparable
    # across rho settings (same Z1, Z3 -> only the correlation changes).
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z3 = rng.standard_normal((n_paths, n_steps))
    Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * Z3

    prices = np.empty((n_paths, n_steps + 1), dtype=float)
    prices[:, 0] = S0
    S = np.full(n_paths, float(S0))
    v = np.full(n_paths, float(v0))

    for t in range(n_steps):
        v_pos = np.maximum(v, 0.0)                       # full truncation: use v+ everywhere
        # price step (log-Euler) using the current variance
        S = S * np.exp((mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * Z1[:, t])
        prices[:, t + 1] = S
        # variance step: raw v carries forward, but v_pos feeds the drift and diffusion
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqrt_dt * Z2[:, t]

    return prices
