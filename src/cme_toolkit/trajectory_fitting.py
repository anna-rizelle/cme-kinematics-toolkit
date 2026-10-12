"""Least-squares trajectory fits for CME leading-edge positions.

Models (t = seconds since the event's first observation, r in R_sun):
    linear     r(t) = r0 + v t                 constant projected speed
    quadratic  r(t) = r0 + v0 t + 0.5 a t^2    constant projected acceleration

Both are linear in their parameters, so we solve weighted least squares
directly (no iterative optimiser). Internally time is in HOURS for numerical
conditioning; results are converted to km/s and m/s^2.

Parameter names (same keys for both models):
    r0_rsun            position at t = 0
    velocity_km_s      linear: the constant speed; quadratic: speed AT t = 0
    acceleration_m_s2  quadratic only

Uncertainty conventions:
    sigma given    -> covariance = (X^T W X)^-1  (trusts your uncertainties)
    sigma omitted  -> covariance scaled by RSS/(n-k) (needs n > k, else NaN)
Only measurement noise is represented; model error and systematic error
(projection, consistent leading-edge bias) are NOT included.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config

_HOUR = 3600.0
_F_V = config.SOLAR_RADIUS_KM / _HOUR                # R_sun/h   -> km/s
_F_A = config.SOLAR_RADIUS_KM * 1000.0 / _HOUR ** 2  # R_sun/h^2 -> m/s^2

CAUTION = (
    "A preferred quadratic model means a curved path describes these points "
    "better; it does not show that the CME physically accelerates. Projection "
    "effects, leading-edge mis-identification, correlated errors and "
    "underestimated uncertainties can all produce apparent curvature."
)


@dataclass(frozen=True)
class FitResult:
    model: str
    params: dict
    param_sigma: dict
    n_points: int
    n_params: int
    dof: int
    rss_rsun2: float
    rmse_rsun: float
    chi2: float            # NaN when no uncertainties were supplied
    reduced_chi2: float    # NaN when no uncertainties or dof == 0
    aic: float
    aicc: float
    bic: float
    criterion_basis: str
    residuals_rsun: np.ndarray = field(repr=False)
    coef_hours: np.ndarray = field(repr=False)

    def predict_position(self, t_s) -> np.ndarray:
        tau = np.atleast_1d(np.asarray(t_s, dtype=float)) / _HOUR
        return np.vander(tau, len(self.coef_hours), increasing=True) @ self.coef_hours

    def velocity_km_s(self, t_s) -> np.ndarray:
        t = np.asarray(t_s, dtype=float)
        v0 = self.params["velocity_km_s"]
        if self.model == "linear":
            return np.full_like(t, v0, dtype=float)
        return v0 + self.params["acceleration_m_s2"] / 1000.0 * t


def _fit_polynomial(t_s, r_rsun, sigma_rsun, degree: int, model: str) -> FitResult:
    t = np.asarray(t_s, dtype=float)
    r = np.asarray(r_rsun, dtype=float)
    if t.ndim != 1 or t.shape != r.shape:
        raise ValueError("t_s and r_rsun must be 1-D arrays of equal length.")
    n, k = len(t), degree + 1
    if n < k:
        raise ValueError(f"The {model} model has {k} parameters and needs at "
                         f"least {k} points; got {n}.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Times must be strictly increasing.")

    weighted = sigma_rsun is not None
    if weighted:
        sigma = np.asarray(sigma_rsun, dtype=float)
        if sigma.shape != r.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("sigma_rsun must match r_rsun and be finite and "
                             "positive; pass None if unavailable.")
        w = 1.0 / sigma
    else:
        w = np.ones_like(r)

    X = np.vander(t / _HOUR, k, increasing=True)  # columns: 1, tau, tau^2
    Xw = X * w[:, None]
    coef, *_ = np.linalg.lstsq(Xw, r * w, rcond=None)

    resid = r - X @ coef
    rss = float(np.sum(resid ** 2))
    dof = n - k

    cov = np.linalg.inv(Xw.T @ Xw)
    if not weighted:
        cov = cov * (rss / dof) if dof > 0 else np.full_like(cov, np.nan)
    sig = np.sqrt(np.diag(cov))

    if weighted:
        chi2 = float(np.sum((resid * w) ** 2))
        reduced = chi2 / dof if dof > 0 else float("nan")
        stat = chi2
        basis = "weighted chi-square (assumes the stated uncertainties are correct)"
    else:
        chi2 = reduced = float("nan")
        stat = n * np.log(rss / n) if rss > 0 else -np.inf
        basis = "unweighted least squares: n*ln(RSS/n)"
    aic = stat + 2 * k
    bic = stat + k * np.log(n)
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else float("nan")

    params = {"r0_rsun": coef[0], "velocity_km_s": coef[1] * _F_V}
    sigmas = {"r0_rsun": sig[0], "velocity_km_s": sig[1] * _F_V}
    if degree == 2:
        params["acceleration_m_s2"] = 2.0 * coef[2] * _F_A
        sigmas["acceleration_m_s2"] = 2.0 * sig[2] * _F_A

    return FitResult(
        model=model,
        params={key: float(val) for key, val in params.items()},
        param_sigma={key: float(val) for key, val in sigmas.items()},
        n_points=n, n_params=k, dof=dof, rss_rsun2=rss,
        rmse_rsun=float(np.sqrt(rss / n)),
        chi2=chi2, reduced_chi2=reduced,
        aic=float(aic), aicc=float(aicc), bic=float(bic),
        criterion_basis=basis, residuals_rsun=resid, coef_hours=coef,
    )


def fit_linear(t_s, r_rsun, sigma_rsun=None) -> FitResult:
    """Constant-velocity fit."""
    return _fit_polynomial(t_s, r_rsun, sigma_rsun, 1, "linear")


def fit_quadratic(t_s, r_rsun, sigma_rsun=None) -> FitResult:
    """Constant-acceleration fit."""
    return _fit_polynomial(t_s, r_rsun, sigma_rsun, 2, "quadratic")


def compare_models(linear: FitResult, quadratic: FitResult) -> dict:
    """Information-criterion comparison; a statistical statement only.

    favored_model is 'quadratic' / 'linear' when AICc differs by more than 2,
    else 'inconclusive'. It is 'not assessable' below the recommended number
    of points, where a 3-parameter curve can fit almost anything.
    """
    result = {
        "criterion": None,
        "delta": float("nan"),
        "delta_bic": float(quadratic.bic - linear.bic),
        "favored_model": "not assessable",
        "reason": "",
        "caution": CAUTION,
    }
    if quadratic.n_points < config.MIN_POINTS_RECOMMENDED:
        result["reason"] = (f"need at least {config.MIN_POINTS_RECOMMENDED} "
                            f"points; have {quadratic.n_points}")
        return result

    use_aicc = np.isfinite(linear.aicc) and np.isfinite(quadratic.aicc)
    result["criterion"] = "AICc" if use_aicc else "AIC"
    delta = (quadratic.aicc - linear.aicc) if use_aicc else (quadratic.aic - linear.aic)
    result["delta"] = float(delta)  # negative favors quadratic
    if delta < -2:
        result["favored_model"] = "quadratic"
    elif delta > 2:
        result["favored_model"] = "linear"
    else:
        result["favored_model"] = "inconclusive"
    return result