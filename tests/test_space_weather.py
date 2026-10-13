import numpy as np
import pandas as pd
import pytest

from cme_toolkit.space_weather import (
    AnalysisWindow,
    format_window_report,
    summaries_to_frame,
    summarize_window,
    window_after,
)


def _data():
    idx = pd.date_range(pd.Timestamp("2000-01-01T00:30:00Z"), periods=12, freq="h",
                        name="time_utc")
    return pd.DataFrame({
        "bz_gsm_nt": [1, -2, -6, -8, -3, 0, 2, -7, -1, 0, 1, 2],
        "speed_km_s": [400, 410, 420, 450, 500, 520, 510, 480, 470, 460, 450, 440],
        "kp": [3, 3, 3, 4.3, 4.3, 4.3, 5.7, 5.7, 5.7, 4, 4, 4],
        "dst_nt": [-10, -12, -15, -30, -55, -80, -60, -40, -30, -20, -15, -10],
    }, index=idx, dtype=float)


W = AnalysisWindow("2000-01-01T00:00:00Z", "2000-01-01T12:00:00Z", "synthetic window")


def test_statistics_values_and_times():
    d = _data()
    summary = summarize_window(d, W)
    st = summary.stats
    assert st["n_expected_hours"] == 12
    assert st["dst_min_nt"] == -80 and st["dst_min_time_utc"] == "2000-01-01T05:30:00Z"
    assert st["kp_max"] == 5.7 and st["kp_max_time_utc"] == "2000-01-01T06:30:00Z"
    assert st["kp_n_3h_blocks_with_data"] == 4
    assert st["bz_min_nt"] == -8 and st["bz_min_time_utc"] == "2000-01-01T03:30:00Z"
    assert st["bz_hours_below_threshold"] == 3
    assert st["speed_max_km_s"] == 520 and st["speed_max_time_utc"] == "2000-01-01T05:30:00Z"
    assert st["speed_mean_km_s"] == pytest.approx(d["speed_km_s"].mean())
    assert all(st[f"{k}_valid_fraction"] == 1.0 for k in ("dst", "kp", "bz", "speed"))
    assert summary.flags == []


def test_insufficient_coverage_withholds_the_statistic_but_keeps_the_counts():
    d = _data()
    d.loc[d.index[:3], "speed_km_s"] = np.nan  # 9 of 12 valid = 75% < 80%
    summary = summarize_window(d, W)
    st = summary.stats
    assert np.isnan(st["speed_max_km_s"]) and st["speed_max_time_utc"] is None
    assert st["speed_n_valid"] == 9 and st["speed_valid_fraction"] == pytest.approx(0.75)
    assert any("withheld" in f for f in summary.flags)
    assert st["dst_min_nt"] == -80  # other variables are unaffected


def test_lower_threshold_reports_the_value_with_a_warning():
    d = _data()
    d.loc[d.index[:3], "speed_km_s"] = np.nan
    summary = summarize_window(d, W, min_valid_fraction=0.5)
    assert summary.stats["speed_max_km_s"] == 520
    assert any("extremes may be missed" in f for f in summary.flags)


def test_window_is_half_open_and_excludes_rows_outside():
    w = AnalysisWindow("2000-01-01T02:00:00Z", "2000-01-01T05:00:00Z")
    st = summarize_window(_data(), w).stats
    assert st["n_expected_hours"] == 3 and st["dst_n_valid"] == 3
    assert st["dst_min_nt"] == -55 and st["dst_min_time_utc"] == "2000-01-01T04:30:00Z"


def test_inconsistent_kp_within_a_block_is_flagged():
    d = _data()
    d.loc[d.index[1], "kp"] = 3.3
    flags = summarize_window(d, W).flags
    assert any("within a 3-hour block" in f for f in flags)


def test_window_with_no_data_gives_nan_and_flags():
    w = AnalysisWindow("2001-01-01T00:00:00Z", "2001-01-01T12:00:00Z")
    summary = summarize_window(_data(), w)
    st = summary.stats
    assert st["dst_n_valid"] == 0 and np.isnan(st["dst_min_nt"]) and st["dst_min_time_utc"] is None
    assert sum("no valid data" in f for f in summary.flags) == 4


def test_window_after_arithmetic():
    w = window_after("2000-01-01T11:00:00Z", 12, 24, "x")
    assert w.start == pd.Timestamp("2000-01-01T23:00:00Z")
    assert w.stop == pd.Timestamp("2000-01-02T23:00:00Z")
    assert w.duration_h == 24


def test_invalid_windows_raise():
    with pytest.raises(ValueError):
        window_after("2000-01-01T00:00:00Z", -1, 24)
    with pytest.raises(ValueError):
        window_after("2000-01-01T00:00:00Z", 0, 0)
    with pytest.raises(ValueError, match="before"):
        AnalysisWindow("2000-01-02", "2000-01-01")


def test_summary_frame_records_window_and_settings():
    df = summaries_to_frame([summarize_window(_data(), W, bz_threshold_nt=-6.0)])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["window_start_utc"] == "2000-01-01T00:00:00Z"
    assert row["bz_threshold_nt"] == -6.0 and row["min_valid_fraction"] == 0.8
    assert row["bz_hours_below_threshold"] == 2  # -8 and -7 are below -6
    assert row["dst_min_nt"] == -80 and "flags" in df.columns


def test_naive_window_times_are_utc():
    w = AnalysisWindow("2000-01-01", "2000-01-01T12:00:00")
    assert str(w.start.tz) == "UTC"


def test_naive_data_index_is_rejected():
    d = _data()
    d.index = d.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        summarize_window(d, W)


def test_bad_min_valid_fraction_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        summarize_window(_data(), W, min_valid_fraction=1.5)


def test_works_with_a_subset_of_columns():
    st = summarize_window(_data()[["dst_nt"]], W).stats
    assert st["dst_min_nt"] == -80 and "kp_max" not in st and "speed_max_km_s" not in st


def test_report_shows_counts_and_the_association_caution():
    text = format_window_report(summarize_window(_data(), W))
    assert "Dst minimum" in text and "-80.0" in text and "valid hours" in text
    assert "not evidence" in text