"""Validation of CME observation tables.

Philosophy: report every problem with its CSV line number, never silently
drop or repair a questionable row. Errors block analysis; warnings do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" or "warning"
    message: str
    event_id: str | None = None
    line: int | None = None  # CSV line number (header is line 1)

    def __str__(self) -> str:
        where = []
        if self.line is not None:
            where.append(f"line {self.line}")
        if self.event_id is not None:
            where.append(f"event {self.event_id}")
        prefix = f" ({', '.join(where)})" if where else ""
        return f"{self.severity.upper()}{prefix}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    def add(self, severity: str, message: str,
            event_id: str | None = None, line: int | None = None) -> None:
        self.issues.append(Issue(severity, message, event_id, line))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def format(self) -> str:
        return "\n".join(str(i) for i in self.issues)


class DataValidationError(Exception):
    """Raised when the input data has one or more errors."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("Input data failed validation:\n" + report.format())


def _strip(value):
    return value.strip() if isinstance(value, str) else value


def _line(idx) -> int:
    return int(idx) + 2  # row 0 is CSV line 2 (line 1 is the header)


def validate_observations(raw: pd.DataFrame):
    """Validate a table whose columns already use the canonical names.

    Returns (typed_dataframe_or_None, ValidationReport). The typed frame keeps
    ALL rows; unparseable values become NaN/NaT rather than being removed.
    """
    report = ValidationReport()

    missing = [c for c in config.REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        report.add(
            "error",
            f"Missing required column(s): {missing}. Columns found: "
            f"{list(raw.columns)}. If your file uses different names, pass a "
            "column_map (canonical name -> your column name).",
        )
        return None, report
    if len(raw) == 0:
        report.add("error", "The file has a header but no data rows.")
        return None, report

    typed = pd.DataFrame(index=raw.index)
    typed["event_id"] = raw["event_id"].map(_strip)
    typed["timestamp"] = pd.to_datetime(
        raw["timestamp"], utc=True, errors="coerce", format="ISO8601"
    )
    typed["position_solar_radii"] = pd.to_numeric(
        raw["position_solar_radii"], errors="coerce"
    ).astype("float64")
    typed["instrument"] = raw["instrument"].map(_strip)

    has_sigma = "position_uncertainty_solar_radii" in raw.columns
    if has_sigma:
        typed["position_uncertainty_solar_radii"] = pd.to_numeric(
            raw["position_uncertainty_solar_radii"], errors="coerce"
        ).astype("float64")
    else:
        typed["position_uncertainty_solar_radii"] = np.nan
    for col in ("source_image", "data_origin"):
        typed[col] = raw[col].map(_strip) if col in raw.columns else np.nan

    # ---- row-level checks -------------------------------------------------
    for idx in raw.index:
        line = _line(idx)
        ev = typed.at[idx, "event_id"]
        ev_label = ev if isinstance(ev, str) and ev else None
        if ev_label is None:
            report.add("error", "event_id is missing.", line=line)

        if pd.isna(raw.at[idx, "timestamp"]):
            report.add("error", "timestamp is missing.", ev_label, line)
        elif pd.isna(typed.at[idx, "timestamp"]):
            report.add(
                "error",
                f"timestamp '{raw.at[idx, 'timestamp']}' cannot be parsed as "
                "an ISO 8601 timestamp (e.g. 2000-01-01T00:12:00Z).",
                ev_label, line,
            )

        p_raw = raw.at[idx, "position_solar_radii"]
        p = typed.at[idx, "position_solar_radii"]
        if pd.isna(p_raw):
            report.add("error", "position_solar_radii is missing.", ev_label, line)
        elif pd.isna(p):
            report.add("error",
                       f"position_solar_radii value '{p_raw}' is not a number.",
                       ev_label, line)
        elif not np.isfinite(p):
            report.add("error", "position_solar_radii is not finite.", ev_label, line)
        elif p <= 0:
            report.add("error",
                       f"position_solar_radii = {p} must be positive.",
                       ev_label, line)
        elif p < config.PLAUSIBLE_MIN_RSUN or p > config.PLAUSIBLE_MAX_RSUN:
            report.add(
                "warning",
                f"position_solar_radii = {p} is outside the usual coronagraph "
                f"range ({config.PLAUSIBLE_MIN_RSUN}-{config.PLAUSIBLE_MAX_RSUN} "
                "R_sun). Check that the values are in solar radii, not pixels or km.",
                ev_label, line,
            )

        if has_sigma:
            s_raw = raw.at[idx, "position_uncertainty_solar_radii"]
            s = typed.at[idx, "position_uncertainty_solar_radii"]
            if not pd.isna(s_raw):
                if pd.isna(s):
                    report.add("error",
                               f"position_uncertainty_solar_radii '{s_raw}' is not a number.",
                               ev_label, line)
                elif not np.isfinite(s) or s <= 0:
                    report.add("error",
                               "position_uncertainty_solar_radii must be positive and finite.",
                               ev_label, line)

    if has_sigma:
        n_missing = int(raw["position_uncertainty_solar_radii"].isna().sum())
        if 0 < n_missing < len(raw):
            report.add("warning",
                       f"{n_missing} row(s) have no uncertainty while others do; "
                       "uncertainty propagation will be skipped for affected events.")
    n_no_inst = int(typed["instrument"].isna().sum())
    if n_no_inst:
        report.add("warning", f"{n_no_inst} row(s) have an empty instrument.")

    # ---- event-level checks ----------------------------------------------
    usable = typed["timestamp"].notna() & np.isfinite(typed["position_solar_radii"])
    for event_id, group in typed[usable].groupby("event_id", sort=False):
        n = len(group)
        if n < config.MIN_POINTS_REQUIRED:
            report.add("error",
                       f"only {n} usable observation(s); at least "
                       f"{config.MIN_POINTS_REQUIRED} are required.", event_id)
        elif n < config.MIN_POINTS_RECOMMENDED:
            report.add("warning",
                       f"only {n} observations; fits will have very few degrees "
                       "of freedom.", event_id)

        for ts, same in group.groupby("timestamp"):
            if len(same) > 1:
                lines = ", ".join(str(_line(i)) for i in same.index)
                report.add("error",
                           f"duplicate timestamp {ts.isoformat()} at CSV lines {lines}.",
                           event_id)

        steps = group["timestamp"].diff().dt.total_seconds()
        for idx in steps.index[steps < 0]:
            report.add("error",
                       "timestamp is earlier than the previous row. Rows must be "
                       "in chronological order (the program will not reorder them for you).",
                       event_id, _line(idx))

        dr = group["position_solar_radii"].diff()
        for idx in dr.index[(dr <= 0) & (steps > 0)]:
            report.add("warning",
                       "position does not increase relative to the previous "
                       "observation. Possible mis-measurement; kept as-is.",
                       event_id, _line(idx))

        if group["instrument"].dropna().nunique() > 1:
            report.add("warning",
                       "multiple instruments in one event (different plate scales "
                       "and systematics); check calibration consistency.", event_id)

    return typed[list(config.ALL_COLUMNS)], report