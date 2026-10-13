import json

import numpy as np
import pytest
import requests

from cme_toolkit.solar_data import DataRetrievalError, HapiError, OmniHapiClient

INFO = {  # SYNTHETIC metadata that mimics the structure seen from the real server
    "HAPI": "2.0",
    "status": {"code": 1200, "message": "OK"},
    "startDate": "1963-01-01T00:00:00Z",
    "stopDate": "2000-01-01T03:00:00Z",
    "parameters": [
        {"name": "Time", "type": "isotime", "units": "UTC", "length": 24, "fill": None},
        {"name": "Rot1800", "type": "integer", "units": None, "fill": "9999"},
        {"name": "BZ_GSM1800", "type": "double", "units": "nT", "fill": "999.9",
         "description": "synthetic test metadata"},
        {"name": "V1800", "type": "double", "units": "Km/s", "fill": "9999.0",
         "description": "synthetic test metadata"},
        {"name": "KP1800", "type": "integer", "units": None, "fill": "99",
         "description": "synthetic test metadata"},
        {"name": "DST1800", "type": "integer", "units": "nT", "fill": "99999",
         "description": "synthetic test metadata"},
    ],
}
ROWS = [  # SYNTHETIC values; columns: Time, BZ_GSM1800, V1800, KP1800, DST1800
    "2000-01-01T00:30:00.000Z,1.0,400,30,-10",
    "2000-01-01T01:30:00.000Z,-2.0,410,30,-12",
    "2000-01-01T02:30:00.000Z,999.9,9999.0,30,-15",
    "2000-01-01T03:30:00.000Z,-3.5,450,43,99999",
    "2000-01-01T04:30:00.000Z,-4.0,455,43,-30",
    "2000-01-01T05:30:00.000Z,-1.0,460,43,-28",
]
START, STOP = "2000-01-01T00:00:00Z", "2000-01-01T06:00:00Z"
ERROR_BODY = json.dumps({"HAPI": "2.0", "status": {"code": 1411,
                         "message": "HAPI error 1411: Parameter out of order"}})


class FakeResponse:
    def __init__(self, text, status_code=200, url="https://example.invalid/hapi/x"):
        self.text, self.status_code, self.url = text, status_code, url
        self.content = text.encode("utf-8")
        self.headers = {"Content-Type": "text/csv"}


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, timeout=None):
        path = url.rsplit("/", 1)[-1]
        self.calls.append((path, dict(params or {})))
        queue = self.routes[path]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def make(tmp_path, data=None, info=INFO):
    routes = {"info": [FakeResponse(json.dumps(info))],
              "data": data if data is not None else [FakeResponse("\n".join(ROWS) + "\n")]}
    session = FakeSession(routes)
    client = OmniHapiClient(cache_dir=tmp_path / "cache", session=session,
                            pause_s=0, backoff_s=0, retries=2)
    return client, session


def data_calls(session):
    return [c for c in session.calls if c[0] == "data"]


def test_fetch_parses_converts_and_marks_fill_as_nan(tmp_path):
    client, _ = make(tmp_path)
    res = client.fetch(START, STOP)
    d = res.data
    assert list(d.columns) == ["bz_gsm_nt", "speed_km_s", "kp", "dst_nt"]
    assert d.shape == (6, 4) and str(d.index.tz) == "UTC"
    assert d.index[0].strftime("%H:%M") == "00:30"
    assert d["kp"].tolist() == pytest.approx([3.0, 3.0, 3.0, 4.3, 4.3, 4.3])
    assert np.isnan(d["bz_gsm_nt"].iloc[2]) and np.isnan(d["speed_km_s"].iloc[2])
    assert np.isnan(d["dst_nt"].iloc[3])
    assert res.coverage["rows_absent_from_response"] == 0
    assert res.coverage["valid_counts"]["bz_gsm_nt"] == 5


def test_requests_use_server_order_and_hapi2_time_names(tmp_path):
    client, session = make(tmp_path)
    res = client.fetch(START, STOP, variables=["dst_nt", "kp", "speed_km_s", "bz_gsm_nt"])
    params = data_calls(session)[0][1]
    assert params["parameters"] == "BZ_GSM1800,V1800,KP1800,DST1800"  # server order
    assert "time.min" in params and "time.max" in params
    assert list(res.data.columns) == ["dst_nt", "kp", "speed_km_s", "bz_gsm_nt"]  # user order


def test_hapi3_servers_use_start_and_stop(tmp_path):
    client, session = make(tmp_path, info=dict(INFO, HAPI="3.1"))
    client.fetch(START, STOP)
    params = data_calls(session)[0][1]
    assert "start" in params and "stop" in params and "time.min" not in params


def test_error_body_with_http_200_is_not_parsed_as_data(tmp_path):
    client, _ = make(tmp_path, data=[FakeResponse(ERROR_BODY)])
    with pytest.raises(HapiError) as exc:
        client.fetch(START, STOP)
    assert exc.value.code == 1411 and "out of order" in str(exc.value)
    assert list((tmp_path / "cache").glob("omni_*.csv")) == []


def test_second_fetch_uses_cache_and_force_refresh_does_not(tmp_path):
    client, session = make(tmp_path)
    first = client.fetch(START, STOP)
    n_calls = len(session.calls)
    second = client.fetch(START, STOP)
    assert not first.from_cache and second.from_cache
    assert len(session.calls) == n_calls  # no new network traffic
    client.fetch(START, STOP, force_refresh=True)
    assert len(session.calls) > n_calls


def test_sidecar_metadata_records_provenance(tmp_path):
    client, _ = make(tmp_path)
    res = client.fetch(START, STOP)
    meta_file = next((tmp_path / "cache").glob("omni_*.meta.json"))
    sidecar = json.loads(meta_file.read_text(encoding="utf-8"))
    for key in ("url", "parameters", "retrieved_utc", "response_sha256", "start", "stop"):
        assert key in sidecar
    assert res.metadata["variables"]["dst_nt"]["fill_value"] == "99999"
    assert res.metadata["dataset_id"] == "OMNI2_H0_MRG1HR"
    assert "MIDPOINT" in res.metadata["time_convention"]


def test_missing_hours_stay_visible_as_nan_rows(tmp_path):
    rows = ROWS[:2] + ROWS[3:]  # the 02:30 row is absent from the response
    client, _ = make(tmp_path, data=[FakeResponse("\n".join(rows) + "\n")])
    res = client.fetch(START, STOP)
    assert len(res.data) == 6 and res.data.iloc[2].isna().all()
    assert res.coverage["rows_received"] == 5 and res.coverage["rows_absent_from_response"] == 1


def test_transient_failure_is_retried(tmp_path):
    client, session = make(tmp_path, data=[requests.ConnectionError("boom"),
                                           FakeResponse("\n".join(ROWS) + "\n")])
    assert len(client.fetch(START, STOP).data) == 6
    assert len(data_calls(session)) == 2


def test_persistent_failure_gives_a_clear_error(tmp_path):
    client, _ = make(tmp_path, data=[requests.Timeout("slow")])
    with pytest.raises(DataRetrievalError, match="failed after 3 attempts"):
        client.fetch(START, STOP)


def test_http_4xx_is_not_retried(tmp_path):
    client, session = make(tmp_path, data=[FakeResponse("nope", status_code=404)])
    with pytest.raises(DataRetrievalError, match="HTTP 404"):
        client.fetch(START, STOP)
    assert len(data_calls(session)) == 1


def test_wrong_column_count_is_an_error(tmp_path):
    rows = [r + ",5" for r in ROWS]
    client, _ = make(tmp_path, data=[FakeResponse("\n".join(rows) + "\n")])
    with pytest.raises(DataRetrievalError, match="columns"):
        client.fetch(START, STOP)


def test_non_numeric_value_is_an_error(tmp_path):
    rows = [ROWS[0].replace(",400,", ",abc,")] + ROWS[1:]
    client, _ = make(tmp_path, data=[FakeResponse("\n".join(rows) + "\n")])
    with pytest.raises(DataRetrievalError, match="non-numeric"):
        client.fetch(START, STOP)


def test_empty_response_means_no_data_and_is_not_cached(tmp_path):
    client, _ = make(tmp_path, data=[FakeResponse("")])
    res = client.fetch(START, STOP)
    assert len(res.data) == 6 and res.data.isna().all().all()
    assert res.coverage["rows_received"] == 0 and res.coverage["rows_absent_from_response"] == 6
    assert list((tmp_path / "cache").glob("omni_*.csv")) == []


def test_window_beyond_dataset_end_is_flagged(tmp_path):
    client, _ = make(tmp_path)  # synthetic metadata ends at 2000-01-01T03:00Z
    assert client.fetch(START, STOP).coverage["window_extends_beyond_dataset_end"] is True


def test_unknown_variable_raises(tmp_path):
    client, _ = make(tmp_path)
    with pytest.raises(ValueError, match="Unknown variable"):
        client.fetch(START, STOP, variables=["bogus"])


def test_start_must_precede_stop(tmp_path):
    client, _ = make(tmp_path)
    with pytest.raises(ValueError, match="before"):
        client.fetch(STOP, START)


def test_naive_times_are_treated_as_utc(tmp_path):
    client, session = make(tmp_path)
    client.fetch("2000-01-01", "2000-01-01T06:00:00")
    assert data_calls(session)[0][1]["time.min"] == "2000-01-01T00:00:00Z"

NO_DATA_BODY = json.dumps({"HAPI": "2.0",
                           "status": {"code": 1201, "message": "OK - no data for time range"}})


def test_no_data_answer_is_an_empty_result_not_an_error(tmp_path):
    client, _ = make(tmp_path, data=[FakeResponse(NO_DATA_BODY)])
    res = client.fetch(START, STOP)
    assert len(res.data) == 6 and res.data.isna().all().all()
    assert res.coverage["rows_received"] == 0
    assert all(v is None for v in res.coverage["last_valid_time_utc"].values())
    assert list((tmp_path / "cache").glob("omni_*.csv")) == []


def test_last_valid_time_is_reported_per_variable(tmp_path):
    rows = ROWS[:-1] + [ROWS[-1].replace(",460,", ",9999.0,")]
    client, _ = make(tmp_path, data=[FakeResponse("\n".join(rows) + "\n")])
    cov = client.fetch(START, STOP).coverage
    assert cov["last_expected_time_utc"] == "2000-01-01T05:30:00Z"
    assert cov["last_valid_time_utc"]["speed_km_s"] == "2000-01-01T04:30:00Z"
    assert cov["last_valid_time_utc"]["dst_nt"] == "2000-01-01T05:30:00Z"