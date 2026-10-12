"""Synthetic CME trajectories with KNOWN parameters, for testing only.

Everything produced here is fake and is labelled data_origin='synthetic'.
It must never be presented as a real observation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def model_position_rsun(times_s, r0_rsun: float, v0_kms: float, a_ms2: float = 0.0) -> np.ndarray:
    """r(t) = r0 + (v0 t + a t^2 / 2) / R_sun, with a converted from m/s^2 to km/s^2."""
    t = np.asarray(times_s, dtype=float)
    return r0_rsun + (v0_kms * t + 0.5 * (a_ms2 / 1000.0) * t ** 2) / config.SOLAR_RADIUS_KM


def make_trajectory(event_id: str, t0: pd.Timestamp, times_s, r0_rsun: float,
                    v0_kms: float, a_ms2: float = 0.0, sigma_rsun: float = 0.0,
                    rng: np.random.Generator | None = None,
                    instrument: str = "SYNTHETIC") -> pd.DataFrame:
    """Return a DataFrame in the project CSV schema. times_s should be whole seconds."""
    times_s = np.asarray(times_s, dtype=float)
    r = model_position_rsun(times_s, r0_rsun, v0_kms, a_ms2)
    if sigma_rsun > 0:
        rng = rng if rng is not None else np.random.default_rng()
        r = r + rng.normal(0.0, sigma_rsun, size=r.shape)
    stamps = (t0 + pd.to_timedelta(times_s, unit="s")).strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame({
        "event_id": event_id,
        "timestamp": list(stamps),
        "position_solar_radii": r,
        "position_uncertainty_solar_radii": sigma_rsun if sigma_rsun > 0 else np.nan,
        "instrument": instrument,
        "source_image": "",
        "data_origin": "synthetic",
    })