"""Plots for one analysed event. Saves PNG files; never opens a window."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .pipeline import EventAnalysis

_COLORS = {"data": "black", "linear": "tab:blue", "quadratic": "tab:red"}
_R_LABEL = r"$R_\odot$"


def _safe_name(event_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", event_id)


def _yerr(values):
    arr = None if values is None else np.asarray(values, dtype=float)
    return None if arr is None or np.all(np.isnan(arr)) else arr


def _tag(a: EventAnalysis) -> str:
    return " [SYNTHETIC DATA]" if (a.observations["data_origin"] == "synthetic").any() else ""


def _x_label(a: EventAnalysis) -> str:
    t0 = a.observations["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"Time since {t0} (h)"


def _save(fig: Figure, path: Path) -> Path:
    fig.savefig(path, dpi=150)
    return path


def plot_position(a: EventAnalysis, path: Path) -> Path:
    fig = Figure(figsize=(7, 4.5), constrained_layout=True)
    ax = fig.subplots()
    fine = np.linspace(a.t_s[0], a.t_s[-1], 300)
    ax.errorbar(a.t_s / 3600, a.r_rsun, yerr=_yerr(a.sigma_rsun), fmt="o",
                color=_COLORS["data"], capsize=3, label="measured (projected)")
    ax.plot(fine / 3600, a.linear.predict_position(fine), "--", color=_COLORS["linear"],
            label=f"linear fit (v = {a.linear.params['velocity_km_s']:.0f} km/s)")
    ax.plot(fine / 3600, a.quadratic.predict_position(fine), "-", color=_COLORS["quadratic"],
            label=f"quadratic fit (a = {a.quadratic.params['acceleration_m_s2']:.1f} m/s$^2$)")
    ax.set_xlabel(_x_label(a))
    ax.set_ylabel(rf"Projected distance ({_R_LABEL})")
    ax.set_title(f"{a.event_id}: position vs time{_tag(a)}", fontsize=10)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_velocity(a: EventAnalysis, path: Path) -> Path:
    fig = Figure(figsize=(7, 4.5), constrained_layout=True)
    ax = fig.subplots()
    tv = a.fd.velocity["time_s"].to_numpy()
    fine = np.linspace(a.t_s[0], a.t_s[-1], 300)
    ax.errorbar(tv / 3600, a.fd.velocity["velocity_km_s"], yerr=_yerr(a.fd.velocity["velocity_sigma_km_s"]),
                fmt="o", color=_COLORS["data"], capsize=3, label="interval velocity")
    ax.plot(fine / 3600, a.linear.velocity_km_s(fine), "--", color=_COLORS["linear"], label="linear fit")
    ax.plot(fine / 3600, a.quadratic.velocity_km_s(fine), "-", color=_COLORS["quadratic"], label="quadratic fit")
    ax.set_xlabel(_x_label(a))
    ax.set_ylabel("Projected velocity (km/s)")
    ax.set_title(f"{a.event_id}: velocity vs time{_tag(a)}", fontsize=10)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_acceleration(a: EventAnalysis, path: Path) -> Path:
    fig = Figure(figsize=(7, 4.5), constrained_layout=True)
    ax = fig.subplots()
    ta = a.fd.acceleration["time_s"].to_numpy()
    ax.errorbar(ta / 3600, a.fd.acceleration["acceleration_m_s2"],
                yerr=_yerr(a.fd.acceleration["acceleration_sigma_m_s2"]),
                fmt="o", color=_COLORS["data"], capsize=3, label="interval acceleration")
    ax.axhline(0.0, ls="--", color=_COLORS["linear"], label="linear fit (a = 0)")
    ax.axhline(a.quadratic.params["acceleration_m_s2"], color=_COLORS["quadratic"],
               label=f"quadratic fit (a = {a.quadratic.params['acceleration_m_s2']:.1f} m/s$^2$)")
    ax.text(0.02, 0.04, "Point-to-point values are noise-dominated;\nsee README for interpretation.",
            transform=ax.transAxes, fontsize=8, va="bottom")
    ax.set_xlabel(_x_label(a))
    ax.set_ylabel("Projected acceleration (m/s$^2$)")
    ax.set_title(f"{a.event_id}: acceleration vs time{_tag(a)}", fontsize=10)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_residuals(a: EventAnalysis, path: Path) -> Path:
    fig = Figure(figsize=(7, 6), constrained_layout=True)
    axes = fig.subplots(2, 1, sharex=True)
    for ax, fit in zip(axes, (a.linear, a.quadratic)):
        ax.errorbar(a.t_s / 3600, fit.residuals_rsun, yerr=_yerr(a.sigma_rsun), fmt="o",
                    color=_COLORS[fit.model], capsize=3)
        ax.axhline(0.0, color="gray", lw=1)
        ax.set_ylabel(rf"measured $-$ fit ({_R_LABEL})")
        ax.set_title(f"{fit.model}: RMSE = {fit.rmse_rsun:.4f} " + _R_LABEL, fontsize=9)
    axes[-1].set_xlabel(_x_label(a))
    fig.suptitle(f"{a.event_id}: residuals{_tag(a)}", fontsize=10)
    return _save(fig, path)


def plot_event(a: EventAnalysis, out_dir) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(a.event_id)
    return [
        plot_position(a, out / f"{stem}_position.png"),
        plot_velocity(a, out / f"{stem}_velocity.png"),
        plot_acceleration(a, out / f"{stem}_acceleration.png"),
        plot_residuals(a, out / f"{stem}_residuals.png"),
    ]