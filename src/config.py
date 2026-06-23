"""All the knobs for the Phase 1 experiment in one place.

Keeping them here means the rest of the code stays uncluttered and later phases
can spin up their own parameter sets without touching the core logic.
"""

from dataclasses import dataclass


@dataclass
class Params:
    """One near-the-money European call on EUR/USD.

    Everything is quoted in USD per EUR, so S0 = 1.10 means 1.10 USD buys 1 EUR.
    """

    S0: float = 1.10          # spot at t=0
    K: float = 1.10           # strike — same as spot, so we start at-the-money
    r: float = 0.03           # risk-free rate (continuously compounded), annual
    sigma: float = 0.10       # volatility, annual
    T: float = 1.0            # time to expiry, in years
    mu: float = 0.03          # path drift. Defaults to r, but it's its own knob so
    #                           a later phase can check the hedge ignores it.
    n_steps: int = 252        # trading days, i.e. one rebalance per day
    n_paths: int = 1000       # Monte Carlo paths
    seed: int = 42            # fixed so runs are reproducible
