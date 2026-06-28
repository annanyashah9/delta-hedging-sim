"""Tests for the numerical core: closed-form Black-Scholes and the delta hedge.

These guard the pieces where a silent bug — a sign flip, a dropped sqrt(T),
a parity violation — would quietly corrupt every figure in the README. The
glue (main_*.py, plotting.py) is deliberately left untested.

Run from the repo root with:  pytest
"""

import numpy as np
import pytest

from src.black_scholes import bs_call_price, bs_delta, bs_gamma
from src.hedge import delta_hedge, gamma_pnl
from src.paths import simulate_gbm

# A representative at-the-money contract to test around.
S0, K, r, SIGMA, T = 1.10, 1.10, 0.03, 0.10, 1.0


# --- Closed-form price ------------------------------------------------------

def test_price_within_no_arbitrage_bounds():
    """A call is worth at least its discounted intrinsic value and never more
    than the spot:  max(S - K e^{-rT}, 0) <= C <= S."""
    C = bs_call_price(S0, K, r, SIGMA, T)
    lower = max(S0 - K * np.exp(-r * T), 0.0)
    assert lower <= C <= S0


def test_put_call_parity():
    """C - P = S - K e^{-rT}. We synthesize the put from parity-independent
    pieces: price a put via its own formula using delta symmetry is overkill,
    so instead check the identity against a put built from the same d1/d2."""
    from scipy.stats import norm
    from src.black_scholes import bs_d1_d2

    d1, d2 = bs_d1_d2(S0, K, r, SIGMA, T)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    call = bs_call_price(S0, K, r, SIGMA, T)
    assert call - put == pytest.approx(S0 - K * np.exp(-r * T), abs=1e-12)


def test_price_increases_with_vol():
    """Vega is positive: more vol, more option value."""
    low = bs_call_price(S0, K, r, 0.05, T)
    high = bs_call_price(S0, K, r, 0.30, T)
    assert high > low


# --- Delta ------------------------------------------------------------------

def test_delta_in_unit_interval():
    spots = np.linspace(0.5, 2.0, 50)
    deltas = bs_delta(spots, K, r, SIGMA, T)
    assert np.all(deltas > 0.0) and np.all(deltas < 1.0)


def test_delta_atm_is_above_half():
    """At the money, delta sits above 0.5: positive r and the +0.5*sigma^2 term
    both push d1 up, so N(d1) > 0.5. Here it's ~0.64."""
    delta = bs_delta(S0, K, r, SIGMA, T)
    assert 0.5 < delta < 0.8


def test_delta_limits():
    """Deep in the money -> delta ~ 1; deep out of the money -> delta ~ 0."""
    assert bs_delta(5.0, K, r, SIGMA, T) == pytest.approx(1.0, abs=1e-6)
    assert bs_delta(0.2, K, r, SIGMA, T) == pytest.approx(0.0, abs=1e-6)


def test_delta_matches_finite_difference():
    """Analytic delta == dPrice/dS, checked by a central difference."""
    h = 1e-5
    spots = np.array([0.9, 1.0, 1.1, 1.3])
    fd = (bs_call_price(spots + h, K, r, SIGMA, T)
          - bs_call_price(spots - h, K, r, SIGMA, T)) / (2 * h)
    analytic = bs_delta(spots, K, r, SIGMA, T)
    np.testing.assert_allclose(analytic, fd, atol=1e-6)


# --- Gamma ------------------------------------------------------------------

def test_gamma_positive():
    spots = np.linspace(0.5, 2.0, 50)
    assert np.all(bs_gamma(spots, K, r, SIGMA, T) > 0.0)


def test_gamma_matches_finite_difference():
    """Analytic gamma == d2Price/dS2, checked by a second central difference."""
    h = 1e-4
    spots = np.array([0.9, 1.0, 1.1, 1.3])
    fd = (bs_call_price(spots + h, K, r, SIGMA, T)
          - 2 * bs_call_price(spots, K, r, SIGMA, T)
          + bs_call_price(spots - h, K, r, SIGMA, T)) / (h ** 2)
    analytic = bs_gamma(spots, K, r, SIGMA, T)
    np.testing.assert_allclose(analytic, fd, rtol=1e-4)


# --- The hedge (Monte Carlo) ------------------------------------------------

def test_frequent_hedge_breaks_even_on_average():
    """If Black-Scholes is right and we rebalance daily, the mean hedging P&L
    across many paths should sit near zero (discretization noise only)."""
    paths = simulate_gbm(S0, mu=r, sigma=SIGMA, T=T, n_steps=252, n_paths=2000, seed=42)
    premium = bs_call_price(S0, K, r, SIGMA, T)
    pnl = delta_hedge(paths, K, r, SIGMA, T, premium)
    # Mean error should be a small fraction of the premium.
    assert abs(pnl.mean()) < 0.02 * premium


def test_gamma_pnl_zero_when_vols_match():
    """The gamma P&L term vanishes identically when realized vol == implied vol."""
    paths = simulate_gbm(S0, mu=r, sigma=SIGMA, T=T, n_steps=252, n_paths=100, seed=7)
    g = gamma_pnl(paths, K, r, sigma_implied=SIGMA, T=T, sigma_realized=SIGMA)
    np.testing.assert_allclose(g, 0.0, atol=1e-12)


def test_gamma_pnl_sign_for_short_call():
    """Short call: calmer realized vol than charged -> positive gamma P&L."""
    paths = simulate_gbm(S0, mu=r, sigma=SIGMA, T=T, n_steps=252, n_paths=100, seed=7)
    calm = gamma_pnl(paths, K, r, sigma_implied=0.12, T=T, sigma_realized=0.08)
    wild = gamma_pnl(paths, K, r, sigma_implied=0.08, T=T, sigma_realized=0.12)
    assert np.all(calm > 0.0) and np.all(wild < 0.0)
