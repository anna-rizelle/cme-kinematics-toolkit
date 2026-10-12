import numpy as np
import pandas as pd
import pytest

from cme_toolkit.pipeline import run_pipeline
from cme_toolkit.synthetic import make_trajectory
from cme_toolkit.validation import DataValidationError

TIMES = [0, 720, 1440, 2160, 2880, 3600, 4320, 5040]


def _write_synthetic_csv(path):
    rng = np.random.default_rng(1)
    t0 = pd.Timestamp("2000-01-01T00:00:00", tz="UTC")
    a = make_trajectory("SYN_A", t0, TIMES, 3.0, 400.0, 0.0, 0.02, rng)
    b = make_trajectory("SYN_B", t0, TIMES, 3.0, 300.0, 30.0, 0.02, rng)
    pd.concat([a, b], ignore_index=True).to_csv(path, index=False)


def test_end_to_end_writes_all_outputs_and_recovers_truth(tmp_path):
    csv_path = tmp_path / "syn.csv"
    _write_synthetic_csv(csv_path)
    out = tmp_path / "out"
    result = run_pipeline(csv_path, out)

    for name in ("processed_observations.csv", "finite_differences.csv",
                 "fit_summary.csv", "summary.txt"):
        assert (out / name).stat().st_size > 0
    pngs = sorted((out / "plots").glob("*.png"))
    assert len(pngs) == 8 and all(p.stat().st_size > 0 for p in pngs)
    assert all(p.exists() for p in result.files)

    summary = pd.read_csv(out / "fit_summary.csv")
    assert len(summary) == 4
    quad_b = summary[(summary["event_id"] == "SYN_B") & (summary["model"] == "quadratic")].iloc[0]
    assert abs(quad_b["acceleration_m_s2"] - 30.0) < 4 * quad_b["acceleration_sigma_m_s2"]
    lin_a = summary[(summary["event_id"] == "SYN_A") & (summary["model"] == "linear")].iloc[0]
    assert abs(lin_a["velocity_km_s"] - 400.0) < 4 * lin_a["velocity_sigma_km_s"]

    proc = pd.read_csv(out / "processed_observations.csv")
    assert len(proc) == 16
    assert np.allclose(proc["position_solar_radii"] - proc["linear_fit_rsun"],
                       proc["linear_residual_rsun"], atol=1e-6)
    assert "synthetic" in result.report_text.lower()


def test_invalid_input_raises_and_writes_nothing(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("event_id,timestamp,position_solar_radii,instrument\n"
                   "E1,2000-01-01T00:00:00Z,abc,X\n")
    out = tmp_path / "out"
    with pytest.raises(DataValidationError):
        run_pipeline(bad, out)
    assert not out.exists()