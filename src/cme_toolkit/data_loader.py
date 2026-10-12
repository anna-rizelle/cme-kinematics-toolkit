"""Load a CSV of CME observations, applying an explicit column map."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .validation import DataValidationError, ValidationReport, validate_observations


def apply_column_map(raw: pd.DataFrame, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Rename file columns to canonical names.

    column_map maps canonical name -> name used in the file, e.g.
    {"timestamp": "obs_time_utc"}. Nothing is guessed.
    """
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    if column_map:
        unknown = [k for k in column_map if k not in config.ALL_COLUMNS]
        if unknown:
            raise ValueError(f"Unknown canonical column(s) in column_map: {unknown}. "
                             f"Valid names: {list(config.ALL_COLUMNS)}")
        absent = [v for v in column_map.values() if v not in raw.columns]
        if absent:
            raise ValueError(f"column_map refers to column(s) not in the file: {absent}. "
                             f"Columns found: {list(raw.columns)}")
        raw = raw.rename(columns={actual: canon for canon, actual in column_map.items()})
    return raw


def load_observations(path: str | Path,
                      column_map: dict[str, str] | None = None
                      ) -> tuple[pd.DataFrame, ValidationReport]:
    """Read, validate and type a CME observation CSV.

    Raises DataValidationError if there are any errors. Returns
    (observations, report); the report may still contain warnings.
    """
    raw = pd.read_csv(path, dtype=str, skipinitialspace=True)
    raw = apply_column_map(raw, column_map)
    observations, report = validate_observations(raw)
    if report.has_errors:
        raise DataValidationError(report)
    return observations, report