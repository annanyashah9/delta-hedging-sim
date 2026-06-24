"""Self-financing delta hedge of a short European call.

Sell one call at t=0, pocket the BS premium, then try to replicate the payoff we
now owe by holding delta units of spot and parking the rest in cash. Whatever's
left at expiry is the hedging P&L. If BS is right it sits near zero on average.

The whole point is that the strategy is self-financing: once we're set up at t=0,
the cash balance only ever moves because of interest or because we traded shares.
We never top it up or skim from it.

By default we rebalance at every step of `paths` (the Phase 1 daily hedge). Phase 2
passes `hedge_steps` to rebalance on a coarser subset of the same fine-grained paths,
which lets us compare hedging frequencies on identical underlying realizations.
"""

import numpy as np

from .black_scholes import bs_delta, bs_gamma


def delta_hedge(paths, K, r, sigma, T, premium, hedge_steps=None):
    """Hedge the short call across every path and return per-path P&L.

    `paths` is (n_paths, n_steps + 1): column 0 is S0, the last column is S_T.
    `premium` is what we received for the call at t=0.

    `hedge_steps` is an optional 1-D array of column indices at which we rebalance.
    It must start at 0 (inception) and end at the last column (expiry). Left as None
    it defaults to every column, i.e. the original daily hedge — same numbers as before.

    Only the spots on the hedge dates matter: between rebalances we hold a fixed delta,
    so the cash just earns interest and the stock leg's P&L depends only on the interval
    endpoints. That's why subsampling the same paths is exact, not an approximation.
    """
    n_steps = paths.shape[1] - 1
    if hedge_steps is None:
        hedge_steps = np.arange(n_steps + 1)

    # Calendar time of each hedge date. Intervals can be uneven (e.g. a short stub at
    # the end of a weekly schedule), so we accrue interest over the actual elapsed time.
    times = hedge_steps * (T / n_steps)

    # t=0: buy delta_0 units of spot. The premium covers part of it; the rest is
    # borrowed (or lent) through cash. That's our opening self-financing balance.
    S0 = paths[:, hedge_steps[0]]
    delta_prev = bs_delta(S0, K, r, sigma, T)
    cash = premium - delta_prev * S0

    # Step through the rebalance dates, stopping one short of expiry: at each one,
    # accrue interest since the previous date, then trade to the new delta.
    for j in range(1, len(hedge_steps) - 1):
        cash *= np.exp(r * (times[j] - times[j - 1]))   # interest over the elapsed interval
        tau = T - times[j]                              # time left to expiry, still positive
        S_i = paths[:, hedge_steps[j]]
        delta_new = bs_delta(S_i, K, r, sigma, tau)
        cash -= (delta_new - delta_prev) * S_i          # buy/sell shares, paid out of cash
        delta_prev = delta_new

    # Expiry: collect the final interval's interest, dump the stock at S_T, and pay out
    # whatever the call is worth. The remainder is the hedging error.
    S_T = paths[:, hedge_steps[-1]]
    cash *= np.exp(r * (times[-1] - times[-2]))
    cash += delta_prev * S_T                            # liquidate the hedge
    payoff = np.maximum(S_T - K, 0.0)                   # what we owe the call holder
    return cash - payoff


def gamma_pnl(paths, K, r, sigma_implied, T, sigma_realized, hedge_steps=None):
    """Theoretical gamma P&L of the SHORT call when realized vol != implied vol.

    Per path, sum over the rebalance intervals:

        sum_i  0.5 * Gamma(S_i, tau_i; sigma_implied) * S_i^2
                   * (sigma_implied^2 - sigma_realized^2) * dt_i

    The gamma is the BS gamma at the vol we hedged with (sigma_implied); the only
    thing realized vol does here is set the (sigma_implied^2 - sigma_realized^2) gap.

    Sign matches `delta_hedge` (short option): calmer-than-charged realized vol
    (sigma_realized < sigma_implied) gives a positive gap -> profit; wilder -> loss.
    When the two vols match, every term is zero.

    `hedge_steps` mirrors `delta_hedge`: the interval-start dates we evaluate gamma on,
    defaulting to every daily step. Fully vectorized across paths.
    """
    n_steps = paths.shape[1] - 1
    if hedge_steps is None:
        hedge_steps = np.arange(n_steps + 1)

    times = hedge_steps * (T / n_steps)
    starts = hedge_steps[:-1]               # interval-start columns (exclude expiry)
    S = paths[:, starts]                    # (n_paths, n_intervals) spots at interval starts
    tau = T - times[:-1]                    # time-to-expiry at each start, all > 0
    dt = np.diff(times)                     # interval lengths (handles uneven grids)

    gamma = bs_gamma(S, K, r, sigma_implied, tau)   # broadcast over the spot matrix
    vol_gap = sigma_implied**2 - sigma_realized**2
    contributions = 0.5 * gamma * S**2 * vol_gap * dt
    return contributions.sum(axis=1)
