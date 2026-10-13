"""How measurement noise, point count and cadence affect fitted acceleration.

Everything here uses SYNTHETIC trajectories with a known true acceleration.
The sigma, cadence and acceleration values are illustrative assumptions chosen
to explore the trade-offs; they are not LASCO calibration numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic import model_position_rsun
from .uncertainty import monte_carlo_fits

MIN_STUDY_POINTS = 5  # below this the model comparison refuses to judge


def acceleration_study(sigmas_rsun, n_points_list, cadences_min, *,
                       r0_rsun: float = 3.0, v0_kms: float = 300.0,
                       a_ms2: float = 25.0, n_trials: int = 500,
                       seed: int = 0) -> pd.DataFrame:
    """One row per (sigma, n_points, cadence) combination.

    Columns include the analytic and Monte Carlo standard deviation of the
    fitted acceleration, a_over_sigma_a (true acceleration divided by the
    analytic uncertainty), and quadratic_favored_fraction (share of trials in
    which AICc prefers the quadratic model).
    """
    if any(n < MIN_STUDY_POINTS for n in n_points_list):
        raise ValueError(f"n_points must be at least {MIN_STUDY_POINTS} for every case.")

    rows: list[dict] = []
    for sigma in sigmas_rsun:
        for n in n_points_list:
            for cadence in cadences_min:
                t = np.arange(n) * cadence * 60.0
                truth = model_position_rsun(t, r0_rsun, v0_kms, a_ms2)
                mc = monte_carlo_fits(t, truth, sigma, n_trials=n_trials,
                                      seed=seed + len(rows))
                quad = mc.samples["quadratic"]
                lin = mc.samples["linear"]
                analytic = mc.baseline["quadratic"].param_sigma["acceleration_m_s2"]
                rows.append({
                    "sigma_rsun": sigma,
                    "n_points": n,
                    "cadence_min": cadence,
                    "duration_h": (n - 1) * cadence / 60.0,
                    "true_a_ms2": a_ms2,
                    "analytic_std_a_ms2": analytic,
                    "mc_std_a_ms2": float(quad["acceleration_m_s2"].std(ddof=1)),
                    "mc_std_v0_kms": float(quad["velocity_km_s"].std(ddof=1)),
                    "mc_std_linear_v_kms": float(lin["velocity_km_s"].std(ddof=1)),
                    "a_over_sigma_a": abs(a_ms2) / analytic,
                    "quadratic_favored_fraction": mc.favored_counts["quadratic"] / n_trials,
                    "n_trials": n_trials,
                })
    return pd.DataFrame(rows)