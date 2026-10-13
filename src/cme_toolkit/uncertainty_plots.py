"""Plots for Monte Carlo results and the acceleration sensitivity study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .uncertainty import MonteCarloResult, _sigma_line
from .visualization import _safe_name


def _hist(ax, samples, fit_value: float, xlabel: str) -> None:
    x = np.asarray(samples, dtype=float)
    ax.hist(x, bins=40, color="0.75", edgecolor="0.45")
    lo, hi = np.percentile(x, [16.0, 84.0])
    ax.axvspan(lo, hi, color="tab:blue", alpha=0.15, label="68% interval")
    ax.axvline(fit_value, color="tab:red", label="fit to measured data")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("trials", fontsize=9)


def plot_monte_carlo(event_id: str, mc: MonteCarloResult, out_dir,
                     synthetic: bool = False) -> Path:
    """Histograms of refitted parameters plus a v(t0)-vs-a scatter."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    quad, lin = mc.samples["quadratic"], mc.samples["linear"]
    bq, bl = mc.baseline["quadratic"].params, mc.baseline["linear"].params

    fig = Figure(figsize=(9, 7), constrained_layout=True)
    (ax_a, ax_v0), (ax_vl, ax_sc) = fig.subplots(2, 2)
    _hist(ax_a, quad["acceleration_m_s2"], bq["acceleration_m_s2"],
          "quadratic fit: acceleration (m/s$^2$)")
    _hist(ax_v0, quad["velocity_km_s"], bq["velocity_km_s"],
          "quadratic fit: speed at first observation (km/s)")
    _hist(ax_vl, lin["velocity_km_s"], bl["velocity_km_s"],
          "linear fit: constant speed (km/s)")
    ax_a.legend(fontsize=8)

    ax_sc.scatter(quad["velocity_km_s"], quad["acceleration_m_s2"], s=3, alpha=0.3, color="0.3")
    corr = float(np.corrcoef(quad["velocity_km_s"], quad["acceleration_m_s2"])[0, 1])
    ax_sc.set_xlabel("v(t0) (km/s)", fontsize=9)
    ax_sc.set_ylabel("acceleration (m/s$^2$)", fontsize=9)
    ax_sc.set_title(f"parameter correlation = {corr:.2f}", fontsize=9)

    tag = " [SYNTHETIC DATA]" if synthetic else ""
    fig.suptitle(f"{event_id}: Monte Carlo ({mc.n_trials} trials, seed {mc.seed}){tag}\n"
                 f"{_sigma_line(mc)}", fontsize=9)
    path = out / f"{_safe_name(event_id)}_monte_carlo.png"
    fig.savefig(path, dpi=150)
    return path


def plot_acceleration_study(df: pd.DataFrame, path) -> Path:
    """Left: acceleration uncertainty vs observation window. Right: detection rate vs SNR."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(11, 4.8), constrained_layout=True)
    ax_a, ax_b = fig.subplots(1, 2)

    for sigma, g in df.groupby("sigma_rsun"):
        ax_a.loglog(g["duration_h"], g["mc_std_a_ms2"], "o",
                    label=rf"$\sigma$ = {sigma:g} $R_\odot$")
    ax_a.set_xlabel("Observation window (h)")
    ax_a.set_ylabel("Std of fitted acceleration (m/s$^2$)")
    ax_a.set_title("Fitted-acceleration uncertainty", fontsize=10)
    ax_a.legend(fontsize=8)

    true_a = float(df["true_a_ms2"].iloc[0])
    pos = df[df["a_over_sigma_a"] > 0]
    if len(pos):
        ax_b.set_xscale("log")
        ax_b.scatter(pos["a_over_sigma_a"], pos["quadratic_favored_fraction"],
                     color="tab:red")
    else:
        ax_b.text(0.5, 0.5, "true a = 0: nothing to detect", ha="center", transform=ax_b.transAxes)
    ax_b.axhline(0.5, color="gray", lw=1, ls="--")
    ax_b.set_ylim(-0.02, 1.02)
    ax_b.set_xlabel("true acceleration / uncertainty of fitted acceleration")
    ax_b.set_ylabel("fraction of trials preferring quadratic (AICc)")
    ax_b.set_title(f"Detection rate (true a = {true_a:g} m/s$^2$)", fontsize=10)

    fig.suptitle("SYNTHETIC study: all parameters are illustrative assumptions", fontsize=10)
    fig.savefig(path, dpi=150)
    return path