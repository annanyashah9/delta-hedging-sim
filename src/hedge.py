"""Daily self-financing delta hedge of a short European call.

Sell one call at t=0, pocket the BS premium, then try to replicate the payoff we
now owe by holding delta units of spot and parking the rest in cash. Whatever's
left at expiry is the hedging P&L. If BS is right it sits near zero on average.

The whole point is that the strategy is self-financing: once we're set up at t=0,
the cash balance only ever moves because of interest or because we traded shares.
We never top it up or skim from it.
"""

import numpy as np

from .black_scholes import bs_delta


def delta_hedge(paths, K, r, sigma, T, premium):
    """Hedge the short call daily across every path and return per-path P&L.

    `paths` is (n_paths, n_steps + 1): column 0 is S0, the last column is S_T.
    `premium` is what we received for the call at t=0.
    """
    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    growth = np.exp(r * dt)   # one day of continuously-compounded interest

    # t=0: buy delta_0 units of spot. The premium covers part of it; the rest is
    # borrowed (or lent) through cash. That's our opening self-financing balance.
    S0 = paths[:, 0]
    delta_prev = bs_delta(S0, K, r, sigma, T)
    cash = premium - delta_prev * S0

    # Walk forward one day at a time, rebalancing to the new delta. We stop one
    # step short of expiry (the last delta is held through to T).
    for i in range(1, n_steps):
        cash *= growth                              # yesterday's cash earns a day of interest
        tau = T - i * dt                            # time left to expiry, still positive here
        S_i = paths[:, i]
        delta_new = bs_delta(S_i, K, r, sigma, tau)
        cash -= (delta_new - delta_prev) * S_i      # buy/sell shares, paid out of cash
        delta_prev = delta_new

    # Expiry: collect the last day's interest, dump the stock at S_T, and pay out
    # whatever the call is worth. The remainder is the hedging error.
    S_T = paths[:, n_steps]
    cash *= growth
    cash += delta_prev * S_T                        # liquidate the hedge
    payoff = np.maximum(S_T - K, 0.0)               # what we owe the call holder
    return cash - payoff
