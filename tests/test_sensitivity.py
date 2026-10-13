import numpy as np
import pytest

from cme_toolkit.sensitivity import acceleration_study
from cme_toolkit.synthetic import model_position_rsun
from cme_toolkit.uncertainty import monte_carlo_fits
from cme_toolkit.uncertainty_plots import plot_acceleration_study, plot_monte_carlo


def test_analytic_acceleration_sigma_scales_exactly_with_noise_and_cadence():
    df = acceleration_study([0.02, 0.04], [10], [12, 24], n_trials=120, seed=0)

    def a_sigma(sigma, cadence):
        row = df[(df["sigma_rsun"] == sigma) & (df["cadence_min"] == cadence)]
        return row["analytic_std_a_ms2"].iloc[0]

    assert a_sigma(0.04, 12) / a_sigma(0.02, 12) == pytest.approx(2.0, rel=1e-9)  # linear in sigma
    assert a_sigma(0.02, 12) / a_sigma(0.02, 24) == pytest.approx(4.0, rel=1e-9)  # 1 / window^2


def test_monte_carlo_matches_analytic_across_the_grid():
    df = acceleration_study([0.03], [5, 10, 20], [6, 12, 24], n_trials=500, seed=4)
    assert len(df) == 9
    assert np.allclose(df["mc_std_a_ms2"] / df["analytic_std_a_ms2"], 1.0, atol=0.15)


def test_detection_rate_is_high_for_large_acceleration_and_low_for_none():
    strong = acceleration_study([0.03], [10], [12], a_ms2=50.0, n_trials=300, seed=1)
    none = acceleration_study([0.03], [10], [12], a_ms2=0.0, n_trials=300, seed=2)
    assert strong["quadratic_favored_fraction"].iloc[0] > 0.95
    assert none["quadratic_favored_fraction"].iloc[0] < 0.06  # false-claim rate ~1%


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="at least 5"):
        acceleration_study([0.03], [4], [12])


def test_study_has_one_row_per_combination():
    df = acceleration_study([0.01, 0.03], [5, 10], [6, 12, 24], n_trials=100, seed=0)
    assert len(df) == 12
    assert {"duration_h", "a_over_sigma_a", "quadratic_favored_fraction"} <= set(df.columns)


def test_plots_are_written(tmp_path):
    t = np.arange(10) * 720.0
    mc = monte_carlo_fits(t, model_position_rsun(t, 3.0, 300.0, 25.0), 0.03,
                          n_trials=120, seed=1)
    p1 = plot_monte_carlo("SYN X/1", mc, tmp_path, synthetic=True)
    df = acceleration_study([0.03], [5, 10], [12], n_trials=120, seed=1)
    p2 = plot_acceleration_study(df, tmp_path / "study.png")
    assert p1.stat().st_size > 0 and p2.stat().st_size > 0