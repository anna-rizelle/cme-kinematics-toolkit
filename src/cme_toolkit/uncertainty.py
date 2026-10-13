"""Monte Carlo uncertainty propagation for CME trajectory fits.

Method
------
Given measured positions r_i with 1-sigma uncertainties s_i, repeat N times:
    r_i' = r_i + s_i * eps,   eps ~ N(0, 1), independent for every point
then refit the linear and quadratic models. The spread of the refitted
parameters shows how much they move if the measurement noise is redrawn.

Scope and limits
----------------
* Error model: independent Gaussian noise with the stated sigma per point.
* Intervals are centred on the fit to the measured data, not on the (unknown)
  truth, and are only as good as the sigma values supplied.
* NOT included: model error (the trajectory may not be quadratic),
  systematic error (consistent leading-edge bias, calibration offsets),
  projection effects, and correlated errors.
* Because both models are linear in their parameters, the Monte Carlo spread
  should match the analytic covariance from trajectory_fitting. Monte Carlo
  earns its keep for model-selection stability and for nonlinear derived
  quantities (used in later phases).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .trajectory_fitting import CAUTION, FitResult, compare_models, fit_linear, fit_quadratic

PARAMETERS = {
    "linear": ("r0_rsun", "velocity_km_s"),
    "quadratic": ("r0_rsun", "velocity_km_s", "acceleration_m_s2"),
}
_UNITS = {"r0_rsun": "R_sun", "velocity_km_s": "km/s", "acceleration_m_s2": "m/s^2"}
_FAVORED_KEYS = ("quadratic", "linear", "inconclusive", "not assessable")
MIN_TRIALS = 100


@dataclass(frozen=True)
class MonteCarloResult:
    n_trials: int
    seed: int | None
    error_model: str
    sigma_rsun: np.ndarray
    sigma_is_assumption: bool
    baseline: dict[str, FitResult]      # fits to the original measured data
    samples: dict[str, pd.DataFrame]    # model -> one row per trial
    favored_counts: dict[str, int]      # AICc preference counts across trials


def monte_carlo_fits(t_s, r_rsun, sigma_rsun, n_trials: int = 2000,
                     seed: int | None = None,
                     sigma_is_assumption: bool = False) -> MonteCarloResult:
    """Perturb the positions, refit both models, and collect the results.

    sigma_rsun may be a single number or one value per point. Set
    sigma_is_assumption=True when sigma is an illustrative guess rather than a
    measured value, so that reports label it as such.
    """
    if n_trials < MIN_TRIALS:
        raise ValueError(f"n_trials must be at least {MIN_TRIALS}; got {n_trials}.")
    t = np.asarray(t_s, dtype=float)
    r = np.asarray(r_rsun, dtype=float)
    if np.ndim(sigma_rsun) == 0:
        sigma = np.full_like(r, float(sigma_rsun))
    else:
        sigma = np.asarray(sigma_rsun, dtype=float)

    # These raise ValueError for bad times, shapes or sigma values.
    baseline = {"linear": fit_linear(t, r, sigma),
                "quadratic": fit_quadratic(t, r, sigma)}

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=(n_trials, len(r))) * sigma

    rows = {"linear": [], "quadratic": []}
    counts = {key: 0 for key in _FAVORED_KEYS}
    for k in range(n_trials):
        r_k = r + noise[k]
        lin = fit_linear(t, r_k, sigma)
        quad = fit_quadratic(t, r_k, sigma)
        rows["linear"].append([lin.params[p] for p in PARAMETERS["linear"]])
        rows["quadratic"].append([quad.params[p] for p in PARAMETERS["quadratic"]])
        counts[compare_models(lin, quad)["favored_model"]] += 1

    samples = {m: pd.DataFrame(rows[m], columns=PARAMETERS[m]) for m in rows}
    return MonteCarloResult(n_trials, seed, "independent Gaussian", sigma,
                            sigma_is_assumption, baseline, samples, counts)


def summarize_monte_carlo(mc: MonteCarloResult, levels=(0.68, 0.95)) -> pd.DataFrame:
    """One row per (model, parameter): fit value, analytic sigma, MC statistics, intervals."""
    rows = []
    for model, df in mc.samples.items():
        fit = mc.baseline[model]
        for p in df.columns:
            x = df[p].to_numpy()
            row = {
                "model": model,
                "parameter": p,
                "fit_value": fit.params[p],
                "analytic_sigma": fit.param_sigma[p],
                "mc_mean": float(x.mean()),
                "mc_std": float(x.std(ddof=1)),
                "mc_median": float(np.median(x)),
            }
            for lev in levels:
                lo, hi = np.percentile(x, [50.0 * (1.0 - lev), 50.0 * (1.0 + lev)])
                tag = int(round(lev * 100))
                row[f"ci{tag}_low"] = float(lo)
                row[f"ci{tag}_high"] = float(hi)
            rows.append(row)
    return pd.DataFrame(rows)


def _sigma_line(mc: MonteCarloResult) -> str:
    lo, hi = float(mc.sigma_rsun.min()), float(mc.sigma_rsun.max())
    size = f"{lo:.3f}" if np.isclose(lo, hi) else f"{lo:.3f}-{hi:.3f}"
    if mc.sigma_is_assumption:
        return (f"Position uncertainty: {size} R_sun "
                "(ILLUSTRATIVE ASSUMPTION, not a measured value)")
    return f"Position uncertainty: {size} R_sun (as supplied in the input data)"


def format_monte_carlo_report(event_id: str, mc: MonteCarloResult) -> str:
    table = summarize_monte_carlo(mc)
    out = [
        f"=== Monte Carlo uncertainty: {event_id} ===",
        f"Method: {mc.n_trials} trials, {mc.error_model} noise added to the measured "
        f"positions, both models refit each time; seed = {mc.seed}.",
        _sigma_line(mc),
        "Intervals are percentile intervals of the refitted values.",
    ]
    for model in ("linear", "quadratic"):
        out.append(f"  {model} model:")
        for _, row in table[table["model"] == model].iterrows():
            unit = _UNITS[row["parameter"]]
            out.append(
                f"    {row['parameter']:<18} fit {row['fit_value']:9.3f} {unit};  "
                f"MC std {row['mc_std']:.3f} (analytic {row['analytic_sigma']:.3f});  "
                f"68%: [{row['ci68_low']:.3f}, {row['ci68_high']:.3f}];  "
                f"95%: [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
            )
    shares = ", ".join(f"{key} {100.0 * mc.favored_counts[key] / mc.n_trials:.1f}%"
                       for key in _FAVORED_KEYS)
    out.append(f"Model preferred by AICc across trials: {shares}")
    out.append("Scope: measurement noise only. Model error, systematic error, projection "
               "effects and correlated errors are not included. " + CAUTION)
    return "\n".join(out)