import numpy as np
import pytest
from scipy.optimize import curve_fit

from cme_toolkit import config
from cme_toolkit.synthetic import model_position_rsun
from cme_toolkit.trajectory_fitting import compare_models, fit_linear, fit_quadratic

R = config.SOLAR_RADIUS_KM
T_IRREG = np.array([0, 600, 1500, 2100, 3600, 4000, 5700], dtype=float)
T10 = np.arange(10) * 720.0
SIGMA = 0.03  # R_sun, illustrative test value


def test_linear_fit_recovers_constant_velocity_exactly():
    r = model_position_rsun(T_IRREG, 3.0, 450.0, 0.0)
    fit = fit_linear(T_IRREG, r)
    assert fit.params["r0_rsun"] == pytest.approx(3.0, abs=1e-9)
    assert fit.params["velocity_km_s"] == pytest.approx(450.0, rel=1e-9)
    assert fit.rmse_rsun < 1e-9
    assert np.allclose(fit.predict_position(T_IRREG), r)


def test_quadratic_fit_recovers_constant_acceleration_exactly():
    r = model_position_rsun(T_IRREG, 3.0, 300.0, 20.0)
    fit = fit_quadratic(T_IRREG, r)
    assert fit.params["r0_rsun"] == pytest.approx(3.0, abs=1e-9)
    assert fit.params["velocity_km_s"] == pytest.approx(300.0, rel=1e-8)
    assert fit.params["acceleration_m_s2"] == pytest.approx(20.0, rel=1e-6)
    assert fit.rmse_rsun < 1e-9


def test_linear_model_leaves_curvature_in_residuals_for_accelerating_data():
    r = model_position_rsun(T_IRREG, 3.0, 300.0, 50.0)
    assert fit_linear(T_IRREG, r).rmse_rsun > 0.05
    assert fit_quadratic(T_IRREG, r).rmse_rsun < 1e-9


@pytest.mark.parametrize("weighted", [True, False])
def test_quadratic_matches_scipy_curve_fit(weighted):
    rng = np.random.default_rng(7)
    r = model_position_rsun(T10, 3.0, 300.0, 25.0) + rng.normal(0, SIGMA, T10.shape)
    sigma = np.full_like(r, SIGMA) if weighted else None

    def model(t, r0, v0, a):
        return r0 + (v0 * t + 0.5 * (a / 1000.0) * t ** 2) / R

    popt, pcov = curve_fit(model, T10, r, p0=[3.0, 300.0, 20.0],
                           sigma=sigma, absolute_sigma=weighted)
    fit = fit_quadratic(T10, r, sigma)
    keys = ("r0_rsun", "velocity_km_s", "acceleration_m_s2")
    assert np.allclose([fit.params[k] for k in keys], popt, rtol=1e-4)
    assert np.allclose([fit.param_sigma[k] for k in keys],
                       np.sqrt(np.diag(pcov)), rtol=1e-2)


def test_reported_uncertainties_are_calibrated_for_linear_fit():
    rng = np.random.default_rng(12345)
    truth = model_position_rsun(T10, 3.0, 450.0, 0.0)
    sigma = np.full_like(truth, SIGMA)
    trials, hits = 500, 0
    for _ in range(trials):
        fit = fit_linear(T10, truth + rng.normal(0, SIGMA, truth.shape), sigma)
        hits += abs(fit.params["velocity_km_s"] - 450.0) <= fit.param_sigma["velocity_km_s"]
    assert 0.60 <= hits / trials <= 0.77  # expect ~0.68


def test_reported_uncertainties_are_calibrated_for_quadratic_fit():
    rng = np.random.default_rng(54321)
    truth = model_position_rsun(T10, 3.0, 300.0, 25.0)
    sigma = np.full_like(truth, SIGMA)
    trials, hits_v, hits_a = 500, 0, 0
    for _ in range(trials):
        fit = fit_quadratic(T10, truth + rng.normal(0, SIGMA, truth.shape), sigma)
        hits_v += abs(fit.params["velocity_km_s"] - 300.0) <= fit.param_sigma["velocity_km_s"]
        hits_a += abs(fit.params["acceleration_m_s2"] - 25.0) <= fit.param_sigma["acceleration_m_s2"]
    assert 0.60 <= hits_v / trials <= 0.77
    assert 0.60 <= hits_a / trials <= 0.77


def test_compare_models_favors_quadratic_for_strongly_accelerating_data():
    rng = np.random.default_rng(3)
    r = model_position_rsun(T10, 3.0, 300.0, 50.0) + rng.normal(0, SIGMA, T10.shape)
    s = np.full_like(r, SIGMA)
    cmp = compare_models(fit_linear(T10, r, s), fit_quadratic(T10, r, s))
    assert cmp["favored_model"] == "quadratic"


def test_compare_models_rarely_claims_acceleration_for_constant_velocity_data():
    rng = np.random.default_rng(99)
    truth = model_position_rsun(T10, 3.0, 450.0, 0.0)
    s = np.full_like(truth, SIGMA)
    trials, false_claims = 300, 0
    for _ in range(trials):
        r = truth + rng.normal(0, SIGMA, truth.shape)
        cmp = compare_models(fit_linear(T10, r, s), fit_quadratic(T10, r, s))
        false_claims += cmp["favored_model"] == "quadratic"
    assert false_claims / trials < 0.05  # expect ~1%


def test_compare_models_refuses_to_judge_with_too_few_points():
    t = np.array([0, 600, 1200, 1800], dtype=float)
    r = model_position_rsun(t, 3.0, 400.0, 10.0)
    cmp = compare_models(fit_linear(t, r), fit_quadratic(t, r))
    assert cmp["favored_model"] == "not assessable"


def test_zero_degrees_of_freedom_gives_nan_uncertainties_not_fake_ones():
    fit = fit_quadratic([0, 600, 1200], [3.0, 3.4, 3.9])
    assert fit.dof == 0
    assert all(np.isnan(v) for v in fit.param_sigma.values())


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="at least 3"):
        fit_quadratic([0, 600], [3.0, 3.4])


def test_non_increasing_times_raise():
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_linear([0, 600, 600], [3.0, 3.4, 3.8])


def test_bad_uncertainties_raise():
    with pytest.raises(ValueError, match="sigma_rsun"):
        fit_linear([0, 600, 1200], [3.0, 3.4, 3.8], [0.03, -0.03, 0.03])