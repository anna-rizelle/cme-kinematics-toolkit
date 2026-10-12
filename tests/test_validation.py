import pytest

from cme_toolkit.data_loader import load_observations
from cme_toolkit.validation import DataValidationError

HEADER = "event_id,timestamp,position_solar_radii,instrument"
GOOD = [
    "E1,2000-01-01T00:00:00Z,3.0,SYNTHETIC",
    "E1,2000-01-01T00:12:00Z,3.5,SYNTHETIC",
    "E1,2000-01-01T00:24:00Z,4.0,SYNTHETIC",
    "E1,2000-01-01T00:36:00Z,4.5,SYNTHETIC",
    "E1,2000-01-01T00:48:00Z,5.0,SYNTHETIC",
]


def write_csv(tmp_path, rows, header=HEADER):
    path = tmp_path / "obs.csv"
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def error_text(tmp_path, rows, header=HEADER):
    with pytest.raises(DataValidationError) as exc:
        load_observations(write_csv(tmp_path, rows, header))
    return str(exc.value)


def test_valid_file_loads(tmp_path):
    obs, report = load_observations(write_csv(tmp_path, GOOD))
    assert len(obs) == 5 and not report.has_errors


def test_naive_timestamps_are_treated_as_utc(tmp_path):
    rows = [r.replace("Z,", ",") for r in GOOD]
    obs, _ = load_observations(write_csv(tmp_path, rows))
    assert str(obs["timestamp"].dt.tz) == "UTC"


def test_duplicate_timestamp_is_an_error(tmp_path):
    rows = GOOD[:2] + ["E1,2000-01-01T00:12:00Z,3.7,SYNTHETIC"] + GOOD[2:]
    assert "duplicate timestamp" in error_text(tmp_path, rows)


def test_out_of_order_rows_are_an_error(tmp_path):
    rows = [GOOD[0], GOOD[2], GOOD[1], GOOD[3], GOOD[4]]
    assert "earlier than the previous row" in error_text(tmp_path, rows)


def test_nonnumeric_position_is_an_error(tmp_path):
    rows = GOOD[:2] + ["E1,2000-01-01T00:24:00Z,abc,SYNTHETIC"] + GOOD[3:]
    assert "not a number" in error_text(tmp_path, rows)


def test_missing_position_is_an_error(tmp_path):
    rows = GOOD[:2] + ["E1,2000-01-01T00:24:00Z,,SYNTHETIC"] + GOOD[3:]
    assert "missing" in error_text(tmp_path, rows)


def test_nonpositive_position_is_an_error(tmp_path):
    rows = GOOD[:2] + ["E1,2000-01-01T00:24:00Z,-1.0,SYNTHETIC"] + GOOD[3:]
    assert "must be positive" in error_text(tmp_path, rows)


def test_unparseable_timestamp_is_an_error(tmp_path):
    rows = GOOD[:2] + ["E1,yesterday,4.0,SYNTHETIC"] + GOOD[3:]
    assert "cannot be parsed" in error_text(tmp_path, rows)


def test_too_few_points_is_an_error(tmp_path):
    assert "at least 3" in error_text(tmp_path, GOOD[:2])


def test_missing_required_column_is_an_error(tmp_path):
    rows = [r.rsplit(",", 1)[0] for r in GOOD]
    assert "Missing required column" in error_text(tmp_path, rows, "event_id,timestamp,position_solar_radii")


def test_column_map_renames_explicitly(tmp_path):
    rows = [r for r in GOOD]
    path = write_csv(tmp_path, rows, "id,obs_time,r_rsun,inst")
    cmap = {"event_id": "id", "timestamp": "obs_time",
            "position_solar_radii": "r_rsun", "instrument": "inst"}
    obs, _ = load_observations(path, column_map=cmap)
    assert len(obs) == 5


def test_questionable_position_warns_but_keeps_every_row(tmp_path):
    rows = [GOOD[0], GOOD[1], "E1,2000-01-01T00:24:00Z,3.4,SYNTHETIC", GOOD[3], GOOD[4]]
    obs, report = load_observations(write_csv(tmp_path, rows))
    assert len(obs) == 5 and not report.has_errors
    assert any("does not increase" in w.message for w in report.warnings)