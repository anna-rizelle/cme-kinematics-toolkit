import numpy as np
import pandas as pd
import pytest

from cme_toolkit.synthetic import model_position_rsun
from cme_toolkit.uncertainty import (
    format_monte_carlo_report,
    monte_carlo_fits,
    summarize_monte_carlo,
)

T10 = np.arange(10) * 720.0
SIGMA = 0.03  # R_sun, illustrative test value


def _noisy(a_ms2, seed):
    rng = np.random.default_rng(seed)
    return model_position_rsun(T10, 3.0, 300.0, a_ms2) + rng.normal(0, SIGMA, T10.shape)


@pytest.fixture(scope="module")
def mc_main():
    return monte_carlo_fits(T10, _noisy(25.0, 2), SIGMA, n_trials=4000, seed=1)


def test_mc_spread_matches_analytic_covariance(mc_main):
    s = summarize_monte_carlo(mc_main)
    assert len(s) == 5  # 2 linear + 3 quadratic parameters
    assert np.allclose(s["mc_std"] / s["analytic_sigma"], 1.0, atol=0.05)


def test_mc_mean_matches_baseline_fit(mc_main):
    s = summarize_monte_carlo(mc_main)
    sem = s["analytic_sigma"] / np.sqrt(mc_main.n_trials)
    assert np.all(np.abs(s["mc_mean"] - s["fit_value"]) < 4 * sem)


def test_intervals_are_ordered_and_match_gaussian_width(mc_main):
    s = summarize_monte_carlo(mc_main)
    assert (s["ci95_low"] < s["ci68_low"]).all()
    assert (s["ci68_low"] < s["fit_value"]).all()
    assert (s["fit_value"] < s["ci68_high"]).all()
    assert (s["ci68_high"] < s["ci95_high"]).all()
    half68 = (s["ci68_high"] - s["ci68_low"]) / 2
    assert np.allclose(half68 / s["analytic_sigma"], 1.0, atol=0.08)


def test_model_selection_counts_for_strongly_accelerating_data():
    mc = monte_carlo_fits(T10, _noisy(50.0, 3), SIGMA, n_trials=300, seed=5)
    assert sum(mc.favored_counts.values()) == 300
    assert mc.favored_counts["quadratic"] / 300 > 0.99


def test_same_seed_reproduces_and_different_seed_differs():
    r = _noisy(25.0, 2)
    a = monte_carlo_fits(T10, r, SIGMA, n_trials=150, seed=7)
    b = monte_carlo_fits(T10, r, SIGMA, n_trials=150, seed=7)
    c = monte_carlo_fits(T10, r, SIGMA, n_trials=150, seed=8)
    pd.testing.assert_frame_equal(a.samples["quadratic"], b.samples["quadratic"])
    assert not a.samples["quadratic"].equals(c.samples["quadratic"])


def test_scalar_and_array_sigma_agree():
    r = _noisy(25.0, 2)
    a = monte_carlo_fits(T10, r, SIGMA, n_trials=120, seed=1)
    b = monte_carlo_fits(T10, r, np.full(10, SIGMA), n_trials=120, seed=1)
    pd.testing.assert_frame_equal(a.samples["linear"], b.samples["linear"])


def test_too_few_trials_raises():
    with pytest.raises(ValueError, match="n_trials"):
        monte_carlo_fits(T10, _noisy(25.0, 2), SIGMA, n_trials=50)


def test_bad_sigma_raises():
    r = _noisy(25.0, 2)
    with pytest.raises(ValueError, match="sigma"):
        monte_carlo_fits(T10, r, -0.03, n_trials=120)
    with pytest.raises(ValueError, match="sigma"):
        monte_carlo_fits(T10, r, np.full(5, SIGMA), n_trials=120)


def test_report_labels_assumed_sigma_and_records_seed():
    r = _noisy(25.0, 2)
    assumed = monte_carlo_fits(T10, r, 0.05, n_trials=120, seed=11, sigma_is_assumption=True)
    text = format_monte_carlo_report("SYN_X", assumed)
    assert "ILLUSTRATIVE ASSUMPTION" in text and "seed = 11" in text
    supplied = monte_carlo_fits(T10, r, SIGMA, n_trials=120, seed=11)
    assert "ILLUSTRATIVE ASSUMPTION" not in format_monte_carlo_report("SYN_X", supplied)