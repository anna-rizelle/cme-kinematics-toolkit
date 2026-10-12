"""Orchestration: load -> validate -> analyse each event -> write outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import load_observations
from .kinematics import FiniteDifferenceResult, elapsed_seconds, finite_differences
from .reporting import write_outputs
from .trajectory_fitting import FitResult, compare_models, fit_linear, fit_quadratic
from .validation import ValidationReport
from .visualization import plot_event


@dataclass
class EventAnalysis:
    event_id: str
    observations: pd.DataFrame       # this event's rows, index reset
    t_s: np.ndarray                  # seconds since first observation
    r_rsun: np.ndarray
    sigma_rsun: np.ndarray | None    # None if uncertainties are incomplete
    fd: FiniteDifferenceResult
    linear: FitResult
    quadratic: FitResult
    comparison: dict


@dataclass
class PipelineResult:
    analyses: list[EventAnalysis]
    report: ValidationReport
    report_text: str
    files: list[Path]


def analyze_event(event_id: str, group: pd.DataFrame) -> EventAnalysis:
    group = group.reset_index(drop=True)
    t = elapsed_seconds(group["timestamp"])
    r = group["position_solar_radii"].to_numpy(dtype=float)
    s = group["position_uncertainty_solar_radii"]
    sigma = s.to_numpy(dtype=float) if s.notna().all() else None

    fd = finite_differences(t, r, sigma)
    linear = fit_linear(t, r, sigma)
    quadratic = fit_quadratic(t, r, sigma)
    return EventAnalysis(event_id, group, t, r, sigma, fd, linear, quadratic,
                         compare_models(linear, quadratic))


def analyze_all(observations: pd.DataFrame) -> list[EventAnalysis]:
    return [analyze_event(eid, g)
            for eid, g in observations.groupby("event_id", sort=False)]


def run_pipeline(csv_path, out_dir="results", make_plots: bool = True,
                 column_map: dict[str, str] | None = None) -> PipelineResult:
    """Raises DataValidationError (before writing anything) on invalid input."""
    observations, report = load_observations(csv_path, column_map)
    analyses = analyze_all(observations)

    files, text = write_outputs(analyses, report, csv_path, out_dir)
    if make_plots:
        for a in analyses:
            files += plot_event(a, Path(out_dir) / "plots")
    return PipelineResult(analyses, report, text, files)