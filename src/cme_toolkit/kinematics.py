"""Finite-difference velocity and acceleration from position-time data.

Math
----
For consecutive points i and i+1 (positions r in R_sun, times t in s):

    v_i = (r_{i+1} - r_i) * R_SUN_KM / (t_{i+1} - t_i)        [km/s]

This is the AVERAGE speed over the interval, so it is assigned to the
interval midpoint. Acceleration is the same idea applied to the velocities:

    a_j = (v_{j+1} - v_j) / (tmid_{j+1} - tmid_j) * 1000      [m/s^2]

Irregular spacing is handled because every difference is divided by its own
actual time step.

Limitations
-----------
* Positions are PROJECTED (plane-of-sky) distances, so speeds are lower
  bounds on the true 3D speed.
* Differentiation amplifies noise: a_j depends on three positions, and
  its uncertainty scales as sigma_r / dt^2. Point-to-point acceleration is
  often noise-dominated. Treat it as diagnostic, not as a physical result.
* Uncertainty propagation assumes independent, Gaussian-like position errors
  (systematic errors, such as a consistent leading-edge bias, are NOT included).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

_R = config.SOLAR_RADIUS_KM


@dataclass(frozen=True)
class FiniteDifferenceResult:
    velocity: pd.DataFrame      # time_s, velocity_km_s, velocity_sigma_km_s
    acceleration: pd.DataFrame  # time_s, acceleration_m_s2, acceleration_sigma_m_s2


def elapsed_seconds(timestamps: pd.Series) -> np.ndarray:
    """Seconds since the first timestamp (timestamps must be datetime-typed)."""
    return (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(dtype=float)


def finite_differences(t_s, r_rsun, sigma_rsun=None) -> FiniteDifferenceResult:
    """Interval velocity and acceleration with optional 1-sigma propagation.

    t_s must be strictly increasing. sigma_rsun, if given, must be finite
    and positive for every point.
    """
    t = np.asarray(t_s, dtype=float)
    r = np.asarray(r_rsun, dtype=float)
    if t.ndim != 1 or t.shape != r.shape:
        raise ValueError("t_s and r_rsun must be 1-D arrays of equal length.")
    if len(t) < 2:
        raise ValueError("At least 2 points are needed to compute a velocity.")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Times must be strictly increasing (found a zero or negative interval).")

    sigma = None
    if sigma_rsun is not None:
        sigma = np.asarray(sigma_rsun, dtype=float)
        if sigma.shape != r.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("sigma_rsun must match r_rsun and be finite and positive; "
                             "pass None if uncertainties are unavailable.")

    # velocity at interval midpoints
    v = np.diff(r) * _R / dt
    t_v = 0.5 * (t[:-1] + t[1:])
    sigma_v = (np.full_like(v, np.nan) if sigma is None
               else _R * np.sqrt(sigma[:-1] ** 2 + sigma[1:] ** 2) / dt)
    velocity = pd.DataFrame({"time_s": t_v, "velocity_km_s": v,
                             "velocity_sigma_km_s": sigma_v})

    # acceleration between successive velocity midpoints
    if len(t) >= 3:
        dt_v = np.diff(t_v)
        a = np.diff(v) / dt_v * 1000.0  # km/s^2 -> m/s^2
        t_a = 0.5 * (t_v[:-1] + t_v[1:])
        if sigma is None:
            sigma_a = np.full_like(a, np.nan)
        else:
            # dv = R*(c0*r_j + c1*r_{j+1} + c2*r_{j+2}); neighbouring velocities
            # share a point, so propagate from the positions directly.
            dt1, dt2 = dt[:-1], dt[1:]
            c0, c1, c2 = 1.0 / dt1, -(1.0 / dt1 + 1.0 / dt2), 1.0 / dt2
            var = (c0 * sigma[:-2]) ** 2 + (c1 * sigma[1:-1]) ** 2 + (c2 * sigma[2:]) ** 2
            sigma_a = _R * np.sqrt(var) / dt_v * 1000.0
    else:
        t_a = a = sigma_a = np.array([], dtype=float)
    acceleration = pd.DataFrame({"time_s": t_a, "acceleration_m_s2": a,
                                 "acceleration_sigma_m_s2": sigma_a})
    return FiniteDifferenceResult(velocity, acceleration)