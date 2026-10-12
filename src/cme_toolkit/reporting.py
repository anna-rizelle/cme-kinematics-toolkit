"""Tables and text reports built from EventAnalysis objects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import __version__, config
from .trajectory_fitting import CAUTION, FitResult

if TYPE_CHECKING:
    from .pipeline import EventAnalysis
    from .validation import ValidationReport

_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def describe_fit(fit: FitResult) -> str:
    p, s = fit.params, fit.param_sigma
    v_label = "v" if fit.model == "linear" else "v(t0)"
    parts = [
        f"r0 = {p['r0_rsun']:.3f} +/- {s['r0_rsun']:.3f} R_sun",
        f"{v_label} = {p['velocity_km_s']:.1f} +/- {s['velocity_km_s']:.1f} km/s",
    ]
    if "acceleration_m_s2" in p:
        parts.append(f"a = {p['acceleration_m_s2']:.1f} +/- {s['acceleration_m_s2']:.1f} m/s^2")
    return (f"  {fit.model:<9} " + "; ".join(parts)
            + f"; RMSE = {fit.rmse_rsun:.4f} R_sun; dof = {fit.dof}")


def processed_observations_table(analyses: list[EventAnalysis]) -> pd.DataFrame:
    """One row per observation, with fitted values and residuals (measured - fit)."""
    frames = []
    for a in analyses:
        obs = a.observations
        frames.append(pd.DataFrame({
            "event_id": a.event_id,
            "timestamp_utc": obs["timestamp"].dt.strftime(_TS_FORMAT),
            "elapsed_s": a.t_s,
            "position_solar_radii": a.r_rsun,
            "position_uncertainty_solar_radii": obs["position_uncertainty_solar_radii"],
            "instrument": obs["instrument"],
            "source_image": obs["source_image"],
            "data_origin": obs["data_origin"],
            "linear_fit_rsun": a.linear.predict_position(a.t_s),
            "linear_residual_rsun": a.linear.residuals_rsun,
            "quadratic_fit_rsun": a.quadratic.predict_position(a.t_s),
            "quadratic_residual_rsun": a.quadratic.residuals_rsun,
        }))
    return pd.concat(frames, ignore_index=True)


def finite_difference_table(analyses: list[EventAnalysis]) -> pd.DataFrame:
    """Long-format interval velocities and accelerations (at interval midpoints)."""
    frames = []
    for a in analyses:
        t0 = a.observations["timestamp"].iloc[0]
        for quantity, df, vcol, scol, unit in (
            ("interval_velocity", a.fd.velocity, "velocity_km_s", "velocity_sigma_km_s", "km/s"),
            ("interval_acceleration", a.fd.acceleration, "acceleration_m_s2",
             "acceleration_sigma_m_s2", "m/s^2"),
        ):
            mid = df["time_s"].to_numpy()
            stamps = (t0 + pd.to_timedelta(mid, unit="s")).strftime(_TS_FORMAT)
            frames.append(pd.DataFrame({
                "event_id": a.event_id,
                "quantity": quantity,
                "midpoint_elapsed_s": mid,
                "midpoint_timestamp_utc": list(stamps),
                "value": df[vcol].to_numpy(),
                "sigma": df[scol].to_numpy(),
                "unit": unit,
            }))
    return pd.concat(frames, ignore_index=True)


def fit_summary_table(analyses: list[EventAnalysis]) -> pd.DataFrame:
    rows = []
    for a in analyses:
        cmp = a.comparison
        for fit in (a.linear, a.quadratic):
            p, s = fit.params, fit.param_sigma
            rows.append({
                "event_id": a.event_id,
                "model": fit.model,
                "n_points": fit.n_points,
                "n_params": fit.n_params,
                "dof": fit.dof,
                "r0_rsun": p["r0_rsun"],
                "r0_sigma_rsun": s["r0_rsun"],
                "velocity_km_s": p["velocity_km_s"],
                "velocity_sigma_km_s": s["velocity_km_s"],
                "velocity_meaning": ("constant speed" if fit.model == "linear"
                                     else "speed at first observation"),
                "acceleration_m_s2": p.get("acceleration_m_s2", np.nan),
                "acceleration_sigma_m_s2": s.get("acceleration_m_s2", np.nan),
                "rmse_rsun": fit.rmse_rsun,
                "rss_rsun2": fit.rss_rsun2,
                "chi2": fit.chi2,
                "reduced_chi2": fit.reduced_chi2,
                "aic": fit.aic,
                "aicc": fit.aicc,
                "bic": fit.bic,
                "criterion_basis": fit.criterion_basis,
                "comparison_favored_model": cmp["favored_model"],
                "comparison_criterion": cmp["criterion"],
                "comparison_delta": cmp["delta"],
            })
    return pd.DataFrame(rows)


def text_report(analyses: list[EventAnalysis], source, report: ValidationReport) -> str:
    out = [
        f"CME Kinematics Toolkit v{__version__} - analysis report",
        f"Input file: {Path(source).name}",
        f"Solar radius used: {config.SOLAR_RADIUS_KM:,.0f} km",
        "Positions are PROJECTED plane-of-sky distances; speeds are lower bounds on true 3D speeds.",
        "Times are UTC (timestamps without a zone are assumed to be UTC).",
        "Uncertainties describe measurement noise only (no systematic or model error).",
    ]
    for issue in report.warnings:
        out.append(str(issue))

    for a in analyses:
        origin = ", ".join(sorted(a.observations["data_origin"].dropna().unique())) or "unspecified"
        out.append(f"\n=== {a.event_id}  ({len(a.t_s)} points, data_origin: {origin}) ===")
        out.append("Interval velocity (time_s = interval midpoint):")
        out.append(a.fd.velocity.round(3).to_string(index=False))
        out.append("Interval acceleration (noisy; see kinematics.py docstring):")
        out.append(a.fd.acceleration.round(3).to_string(index=False))

        weighting = ("weighted by the stated uncertainties" if a.sigma_rsun is not None
                     else "UNWEIGHTED (no uncertainties given)")
        out.append(f"Trajectory fits ({weighting}):")
        out.append(describe_fit(a.linear))
        out.append(describe_fit(a.quadratic))
        if a.sigma_rsun is not None:
            out.append(f"  reduced chi-square: linear {a.linear.reduced_chi2:.2f}, "
                       f"quadratic {a.quadratic.reduced_chi2:.2f} "
                       "(about 1 if uncertainties are realistic)")
        cmp = a.comparison
        if cmp["favored_model"] == "not assessable":
            out.append(f"  Model comparison: not assessable ({cmp['reason']}).")
        else:
            out.append(f"  Model comparison: {cmp['favored_model']} "
                       f"(delta {cmp['criterion']} = {cmp['delta']:.1f}, quadratic minus linear; "
                       f"delta BIC = {cmp['delta_bic']:.1f})")

    out.append("\nNote: " + CAUTION)
    return "\n".join(out)


def write_outputs(analyses, report, source, out_dir) -> tuple[list[Path], str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for name, table in (
        ("processed_observations.csv", processed_observations_table(analyses)),
        ("finite_differences.csv", finite_difference_table(analyses)),
        ("fit_summary.csv", fit_summary_table(analyses)),
    ):
        path = out / name
        table.to_csv(path, index=False, float_format="%.8g")
        files.append(path)

    text = text_report(analyses, source, report)
    summary_path = out / "summary.txt"
    summary_path.write_text(text + "\n", encoding="utf-8")
    files.append(summary_path)
    return files, text