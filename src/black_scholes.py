"""Closed-form Black-Scholes call price and delta.

We use plain Black-Scholes with one rate `r`, i.e. EUR/USD is treated like a stock
that pays no dividend. The proper FX version (Garman-Kohlhagen, with the EUR rate
playing the role of a dividend yield) can wait for a later phase.

Everything here takes scalar or array `S`/`T`, so the hedge loop can price delta
for all paths in a single call.
"""

import numpy as np
from scipy.stats import norm


def bs_d1_d2(S, K, r, sigma, T):
    """The usual d1, d2 from the Black-Scholes formula."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_call_price(S, K, r, sigma, T):
    """European call price: C = S N(d1) - K e^{-rT} N(d2)."""
    d1, d2 = bs_d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, r, sigma, T):
    """Call delta, N(d1) — units of spot to hold against one short call."""
    d1, _ = bs_d1_d2(S, K, r, sigma, T)
    return norm.cdf(d1)


def bs_gamma(S, K, r, sigma, T):
    """Call gamma, phi(d1) / (S sigma sqrt(T)) — same for a put.

    The curvature of the option value in spot; it's what drives the gamma P&L
    when realized vol differs from the vol we hedged with.
    """
    d1, _ = bs_d1_d2(S, K, r, sigma, T)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))
