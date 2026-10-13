"""Descriptive summaries of near-Earth conditions inside an analysis window.

This module describes what the OMNI data show in a window. It does NOT decide
whether a disturbance was caused by any particular CME; that needs independent
evidence (see ASSOCIATION_CAUTION).

Rules:
* Every statistic comes with its valid-hour count and valid fraction.
* If a variable's valid fraction is below min_valid_fraction (an illustrative
  assumption, default 0.8) its statistics are withheld (NaN), not computed from
  a thin sample. Set it to 0 to disable.
* Kp is a 3-hourly index repeated in every hourly row, so Kp is also summarised
  as a number of 3-hour blocks (hourly Kp rows are not independent samples).
* Windows are half-open: [start, stop). Naive times are taken as UTC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .solar_data import _expected_grid, _to_utc

DEFAULT_MIN_VALID_FRACTION = 0.8   # illustrative assumption
DEFAULT_BZ_THRESHOLD_NT = -5.0     # illustrative assumption
KP_BLOCK = "3h"

ASSOCIATION_CAUTION = (
    "These are descriptive statistics of conditions in the window. A disturbance "
    "inside the window is not evidence that a particular CME caused it: other CMEs, "
    "high-speed streams and corotating interaction regions also drive geomagnetic "
    "activity, and the window itself is an assumption."
)

_SHORT = {"dst_nt": "dst", "kp": "kp", "bz_gsm_nt": "bz", "speed_km_s": "speed"}


def _ts(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class AnalysisWindow:
    start: pd.Timestamp
    stop: pd.Timestamp   # exclusive
    label: str = ""

    def __post_init__(self):
        object.__setattr__(self, "start", _to_utc(self.start))
        object.__setattr__(self, "stop", _to_utc(self.stop))
        if not self.start < self.stop:
            raise ValueError("Window start must be before stop.")

    @property
    def duration_h(self) -> float:
        return (self.stop - self.start).total_seconds() / 3600.0


def window_after(reference_time, start_offset_h: float, duration_h: float,
                 label: str = "") -> AnalysisWindow:
    """Window beginning start_offset_h hours after reference_time and lasting duration_h hours."""
    if start_offset_h < 0:
        raise ValueError("start_offset_h must be >= 0.")
    if duration_h <= 0:
        raise ValueError("duration_h must be > 0.")
    start = _to_utc(reference_time) + pd.Timedelta(hours=start_offset_h)
    return AnalysisWindow(start, start + pd.Timedelta(hours=duration_h), label)


@dataclass(frozen=True)
class WindowSummary:
    window: AnalysisWindow
    stats: dict
    flags: list
    settings: dict


def summarize_window(data: pd.DataFrame, window: AnalysisWindow, *,
                     min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
                     bz_threshold_nt: float = DEFAULT_BZ_THRESHOLD_NT) -> WindowSummary:
    """Summarise an OmniData.data table (columns bz_gsm_nt, speed_km_s, kp, dst_nt)."""
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1.")
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
        raise ValueError("data must be indexed by timezone-aware (UTC) time stamps.")

    sel = data[(data.index >= window.start) & (data.index < window.stop)]
    n_expected = len(_expected_grid(window.start, window.stop))
    nan = float("nan")
    stats: dict = {"n_expected_hours": n_expected}
    flags: list = []
    if n_expected == 0:
        flags.append("the window contains no hourly time stamps")

    usable: dict = {}
    for col, key in _SHORT.items():
        if col not in data.columns:
            continue
        s = sel[col].dropna()
        frac = len(s) / n_expected if n_expected else nan
        stats[f"{key}_n_valid"] = int(len(s))
        stats[f"{key}_valid_fraction"] = float(frac)
        if len(s) == 0:
            flags.append(f"{col}: no valid data in this window")
            usable[col] = None
        elif frac < min_valid_fraction:
            flags.append(f"{col}: only {100 * frac:.0f}% of hours valid (minimum "
                         f"{100 * min_valid_fraction:.0f}%); statistics withheld")
            usable[col] = None
        else:
            usable[col] = s
            if frac < 1.0:
                flags.append(f"{col}: {100 * frac:.0f}% of hours valid; extremes may be missed")

    if "dst_nt" in usable:
        s = usable["dst_nt"]
        stats["dst_min_nt"] = float(s.min()) if s is not None else nan
        stats["dst_min_time_utc"] = _ts(s.idxmin()) if s is not None else None
    if "kp" in usable:
        s = usable["kp"]
        stats["kp_max"] = float(s.max()) if s is not None else nan
        stats["kp_max_time_utc"] = _ts(s.idxmax()) if s is not None else None
        kp_all = sel["kp"].dropna()
        blocks = kp_all.index.floor(KP_BLOCK)
        stats["kp_n_3h_blocks_with_data"] = int(blocks.nunique())
        if (kp_all.groupby(blocks).nunique() > 1).any():
            flags.append("kp: values differ within a 3-hour block (unexpected); check the data")
    if "bz_gsm_nt" in usable:
        s = usable["bz_gsm_nt"]
        stats["bz_min_nt"] = float(s.min()) if s is not None else nan
        stats["bz_min_time_utc"] = _ts(s.idxmin()) if s is not None else None
        stats["bz_hours_below_threshold"] = (
            int((s < bz_threshold_nt).sum()) if s is not None else nan)
    if "speed_km_s" in usable:
        s = usable["speed_km_s"]
        stats["speed_max_km_s"] = float(s.max()) if s is not None else nan
        stats["speed_max_time_utc"] = _ts(s.idxmax()) if s is not None else None
        stats["speed_mean_km_s"] = float(s.mean()) if s is not None else nan

    settings = {"min_valid_fraction": float(min_valid_fraction),
                "bz_threshold_nt": float(bz_threshold_nt)}
    return WindowSummary(window, stats, flags, settings)


def summaries_to_frame(summaries) -> pd.DataFrame:
    """One row per window, including the window definition and every setting used."""
    rows = []
    for s in summaries:
        rows.append({
            "label": s.window.label,
            "window_start_utc": _ts(s.window.start),
            "window_stop_utc": _ts(s.window.stop),
            "window_duration_h": s.window.duration_h,
            **s.settings,
            **s.stats,
            "flags": "; ".join(s.flags),
        })
    return pd.DataFrame(rows)


def format_window_report(summary: WindowSummary) -> str:
    w, st = summary.window, summary.stats
    out = [
        f"=== {w.label or 'window'} ===",
        f"Window (UTC): {w.start:%Y-%m-%d %H:%M} to {w.stop:%Y-%m-%d %H:%M} ({w.duration_h:g} h)",
        f"Settings (illustrative assumptions): minimum valid fraction "
        f"{summary.settings['min_valid_fraction']:g}; Bz threshold "
        f"{summary.settings['bz_threshold_nt']:g} nT",
        f"Hourly samples expected in the window: {st['n_expected_hours']}",
    ]
    rows = [("Dst minimum", "dst_min_nt", "nT", "dst_min_time_utc", "dst"),
            ("Kp maximum", "kp_max", "", "kp_max_time_utc", "kp"),
            ("Bz (GSM) minimum", "bz_min_nt", "nT", "bz_min_time_utc", "bz"),
            ("Solar-wind speed maximum", "speed_max_km_s", "km/s", "speed_max_time_utc", "speed")]
    for name, vkey, unit, tkey, short in rows:
        if f"{short}_n_valid" not in st:
            continue
        value = st[vkey]
        shown = ("withheld (too little valid data)" if pd.isna(value)
                 else f"{value:.1f} {unit} at {st[tkey]}".replace("  ", " "))
        out.append(f"  {name}: {shown}   [valid hours: {st[f'{short}_n_valid']}/{st['n_expected_hours']}]")
    if "kp_n_3h_blocks_with_data" in st:
        out.append(f"  Kp 3-hour blocks with data: {st['kp_n_3h_blocks_with_data']}")
    if "bz_hours_below_threshold" in st and not pd.isna(st["bz_hours_below_threshold"]):
        out.append(f"  Hours with Bz below {summary.settings['bz_threshold_nt']:g} nT: "
                   f"{st['bz_hours_below_threshold']}")
    if "speed_mean_km_s" in st and not pd.isna(st["speed_mean_km_s"]):
        out.append(f"  Mean solar-wind speed: {st['speed_mean_km_s']:.1f} km/s")
    for flag in summary.flags:
        out.append(f"  FLAG: {flag}")
    out.append("Note: " + ASSOCIATION_CAUTION)
    return "\n".join(out)