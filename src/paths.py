"""Vectorized geometric Brownian motion path simulation."""

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
