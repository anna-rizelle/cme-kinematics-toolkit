import numpy as np
import pytest

from cme_toolkit import config
from cme_toolkit.kinematics import finite_differences
from cme_toolkit.synthetic import model_position_rsun

T = np.array([0, 600, 1500, 2100, 3600, 4000, 5700], dtype=float)  # irregular


def test_constant_velocity_recovered_with_irregular_spacing():
    r = model_position_rsun(T, 3.0, 450.0, 0.0)
    res = finite_differences(T, r)
    assert np.allclose(res.velocity["velocity_km_s"], 450.0, rtol=1e-9)
    assert np.allclose(res.acceleration["acceleration_m_s2"], 0.0, atol=1e-6)


def test_constant_acceleration_recovered_with_irregular_spacing():
    a_true = 20.0  # m/s^2
    r = model_position_rsun(T, 3.0, 300.0, a_true)
    res = finite_differences(T, r)
    # Interval-average speed of a quadratic equals the speed at the midpoint.
    t_mid = res.velocity["time_s"].to_numpy()
    assert np.allclose(res.velocity["velocity_km_s"], 300.0 + (a_true / 1000.0) * t_mid, rtol=1e-9)
    assert np.allclose(res.acceleration["acceleration_m_s2"], a_true, rtol=1e-6)


def test_uncertainty_propagation_uniform_spacing():
    dt = 720.0
    t = np.arange(0, 5 * dt, dt)
    r = np.linspace(3.0, 5.0, len(t))
    res = finite_differences(t, r, np.full_like(r, 0.05))
    R = config.SOLAR_RADIUS_KM
    assert np.allclose(res.velocity["velocity_sigma_km_s"], R * 0.05 * np.sqrt(2) / dt, rtol=1e-9)
    assert np.allclose(res.acceleration["acceleration_sigma_m_s2"],
                       R * 0.05 * np.sqrt(6) / dt ** 2 * 1000.0, rtol=1e-9)


@pytest.mark.parametrize("times", [[0, 100, 100, 200], [0, 100, 50, 200]])
def test_nonpositive_interval_raises(times):
    with pytest.raises(ValueError, match="strictly increasing"):
        finite_differences(times, [3.0, 3.1, 3.2, 3.3])


def test_two_points_give_velocity_but_no_acceleration():
    res = finite_differences([0, 600], [3.0, 3.4])
    assert len(res.velocity) == 1
    assert len(res.acceleration) == 0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        finite_differences([0, 1, 2], [3.0, 3.1])