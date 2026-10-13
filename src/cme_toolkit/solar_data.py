"""OMNI hourly solar-wind and geomagnetic data from NASA CDAWeb's HAPI server.

This is the ONLY module that knows the server URL, the dataset ID, the HAPI
parameter names and the response format. If NASA changes the service, only this
file should need to change.

Verified against the live server (scripts/probe_omni_hapi.py):
* HAPI version 2.0 uses time.min / time.max (HAPI 3 renamed them start / stop);
  the right pair is chosen from the version the server reports.
* Parameters MUST be requested in the dataset's own order. Otherwise the server
  answers HTTP 200, labelled text/csv, with a JSON error body (code 1411).
  Every response body is therefore inspected, not just the HTTP status.
* CSV rows have no header. Time stamps are the MIDPOINT of each hourly average.
* Kp is stored as Kp*10, and each 3-hour value is repeated in every hourly row.
* Fill values are given per parameter in the dataset metadata.
Not yet verified: behaviour for windows beyond the dataset end. An empty body is
treated as "no data" and reported through the coverage summary.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SERVER = "https://cdaweb.gsfc.nasa.gov/hapi"
DATASET_ID = "OMNI2_H0_MRG1HR"
DATASET_DOI = "10.48322/1shr-ht18"
USER_AGENT = "cme-kinematics-toolkit/0.1 (student research project)"

# friendly column name -> HAPI parameter name
VARIABLES = {
    "bz_gsm_nt": "BZ_GSM1800",
    "speed_km_s": "V1800",
    "kp": "KP1800",
    "dst_nt": "DST1800",
}
UNITS = {
    "bz_gsm_nt": "nT",
    "speed_km_s": "km/s",
    "kp": "Kp index (3-hourly value repeated in each hourly row)",
    "dst_nt": "nT",
}
_KP_SCALE = 10.0
TIME_CONVENTION = ("Each time stamp is the MIDPOINT of an hourly average "
                   "(the hour [hh:00, hh+1:00) is stamped hh:30), UTC.")


class DataRetrievalError(Exception):
    """Network, format or content problem while retrieving data."""


class HapiError(DataRetrievalError):
    """The server reported an error in the response body."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(f"HAPI server reported an error (code {code}): {message}")


@dataclass
class OmniData:
    data: pd.DataFrame   # index time_utc (hh:30 stamps); NaN = missing/fill/absent
    metadata: dict       # provenance, units, fill values, conventions
    coverage: dict       # what was expected vs received
    from_cache: bool


def _to_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _fmt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_fresh(path: Path, max_age_days) -> bool:
    if not path.exists():
        return False
    return max_age_days is None or (time.time() - path.stat().st_mtime) < max_age_days * 86400


NO_DATA_CODE = 1201


def _body_signals_no_data(text: str) -> bool:
    """Inspect a response body.

    Returns True for the server's 'no data for this time range' answer (code
    1201). Raises HapiError for any other error. The server can report errors
    with HTTP 200, so the body of every response is checked.
    """
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return False
    code, message = None, stripped[:200]
    try:
        status = json.loads(stripped).get("status", {})
        code, message = status.get("code"), status.get("message", message)
    except (ValueError, AttributeError):
        pass
    if code == NO_DATA_CODE:
        return True
    raise HapiError(code, message)


def _parse_csv(text: str, hapi_names: list[str], fills: dict) -> pd.DataFrame:
    """Headerless HAPI CSV -> DataFrame; fill values become NaN, nothing else is altered."""
    try:
        raw = pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataRetrievalError(f"Could not parse the response as CSV: {exc}") from exc
    if raw.shape[1] != 1 + len(hapi_names):
        raise DataRetrievalError(
            f"Expected {1 + len(hapi_names)} columns (time + {hapi_names}) but the response "
            f"has {raw.shape[1]} columns.")
    times = pd.to_datetime(raw[0], utc=True, errors="coerce", format="ISO8601")
    if times.isna().any():
        raise DataRetrievalError(f"Unparseable time stamp: {raw[0][times.isna()].iloc[0]!r}")
    index = pd.DatetimeIndex(times, name="time_utc")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise DataRetrievalError("Time stamps are not strictly increasing.")

    columns = {}
    for i, name in enumerate(hapi_names, start=1):
        numeric = pd.to_numeric(raw[i], errors="coerce")
        bad = numeric.isna() & (raw[i].str.strip() != "")
        if bad.any():
            raise DataRetrievalError(
                f"{name}: non-numeric value {raw[i][bad].iloc[0]!r} in row {int(bad.idxmax()) + 1}.")
        values = numeric.to_numpy(dtype=float, copy=True)
        fill = fills.get(name)
        if fill is not None:
            values[np.isclose(values, fill, rtol=0, atol=1e-6)] = np.nan
        columns[name] = values
    return pd.DataFrame(columns, index=index)


def _expected_grid(start: pd.Timestamp, stop: pd.Timestamp) -> pd.DatetimeIndex:
    first = start.floor("h") + pd.Timedelta(minutes=30)
    if first < start:
        first += pd.Timedelta(hours=1)
    return pd.date_range(first, stop, freq="h", inclusive="left")


class OmniHapiClient:
    """Fetch OMNI hourly data with a local on-disk cache.

    Cached files (in cache_dir): omni_<key>.csv (the raw response) and
    omni_<key>.meta.json (URL, parameters, retrieval time, SHA-256, ...).
    Recent OMNI/Dst values are preliminary and get revised, so for recent windows
    use force_refresh=True or max_age_days to re-download.
    """

    def __init__(self, cache_dir="data/raw/omni_cache", session=None, pause_s: float = 1.0,
                 timeout_s: float = 60.0, retries: int = 2, backoff_s: float = 2.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if session is None:
            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
        self.session = session
        self.pause_s, self.timeout_s = pause_s, timeout_s
        self.retries, self.backoff_s = retries, backoff_s
        self._last_request = None

    # ---- network ---------------------------------------------------------
    def _wait_turn(self) -> None:
        if self._last_request is not None:
            remaining = self.pause_s - (time.monotonic() - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request = time.monotonic()

    def _get(self, path: str, params: dict):
        url = f"{SERVER}/{path}"
        last_error = None
        for attempt in range(self.retries + 1):
            self._wait_turn()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout_s)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}"
                elif resp.status_code >= 400:
                    raise DataRetrievalError(
                        f"HTTP {resp.status_code} from {resp.url}: {resp.text[:200]}")
                else:
                    return resp
            if attempt < self.retries:
                time.sleep(self.backoff_s * (2 ** attempt))
        raise DataRetrievalError(
            f"Request to {url} failed after {self.retries + 1} attempts ({last_error}).")

    # ---- dataset metadata ------------------------------------------------
    def _info(self, force_refresh: bool, max_age_days):
        path = self.cache_dir / f"info_{DATASET_ID}.json"
        meta_path = self.cache_dir / f"info_{DATASET_ID}.meta.json"
        if not force_refresh and _is_fresh(path, max_age_days) and meta_path.exists():
            return (json.loads(path.read_text(encoding="utf-8")),
                    json.loads(meta_path.read_text(encoding="utf-8")))
        resp = self._get("info", {"id": DATASET_ID})
        try:
            info = json.loads(resp.text)
        except ValueError as exc:
            raise DataRetrievalError("The dataset metadata response was not valid JSON.") from exc
        code = info.get("status", {}).get("code")
        if code not in (None, 1200):
            raise HapiError(code, info.get("status", {}).get("message"))
        if not isinstance(info.get("parameters"), list):
            raise DataRetrievalError("The dataset metadata has no 'parameters' list.")
        meta = {"url": resp.url, "retrieved_utc": _now(),
                "response_sha256": hashlib.sha256(resp.content).hexdigest()}
        path.write_text(resp.text, encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return info, meta

    # ---- main entry point ------------------------------------------------
    def fetch(self, start, stop, variables=None, *, force_refresh: bool = False,
              max_age_days: float | None = None) -> OmniData:
        """Hourly data with time stamps in [start, stop). Naive times are taken as UTC."""
        start_ts, stop_ts = _to_utc(start), _to_utc(stop)
        if not start_ts < stop_ts:
            raise ValueError("start must be before stop.")
        friendly = list(variables) if variables is not None else list(VARIABLES)
        unknown = [v for v in friendly if v not in VARIABLES]
        if unknown:
            raise ValueError(f"Unknown variable(s) {unknown}. Valid names: {list(VARIABLES)}")

        info, info_meta = self._info(force_refresh, max_age_days)
        by_name = {p["name"]: p for p in info["parameters"]}
        order = [p["name"] for p in info["parameters"]]
        wanted = [VARIABLES[v] for v in friendly]
        absent = [n for n in wanted if n not in by_name]
        if absent:
            raise DataRetrievalError(f"Parameter(s) {absent} are not in {DATASET_ID}.")
        names = sorted(wanted, key=order.index)  # the server requires its own order
        fills = {n: float(by_name[n]["fill"]) for n in names if by_name[n].get("fill") is not None}
        version = str(info.get("HAPI", ""))
        start_str, stop_str = _fmt(start_ts), _fmt(stop_ts)

        key = hashlib.sha256("|".join([DATASET_ID, version, start_str, stop_str, *names])
                             .encode()).hexdigest()[:16]
        csv_path = self.cache_dir / f"omni_{key}.csv"
        meta_path = self.cache_dir / f"omni_{key}.meta.json"

        from_cache = False
        if not force_refresh and _is_fresh(csv_path, max_age_days) and meta_path.exists():
            text = csv_path.read_text(encoding="utf-8")
            request_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            from_cache = True
        else:
            tmin, tmax = ("time.min", "time.max") if version.startswith("2") else ("start", "stop")
            resp = self._get("data", {"id": DATASET_ID, tmin: start_str, tmax: stop_str,
                                      "parameters": ",".join(names)})
            text = resp.text
            if _body_signals_no_data(text):
                text = ""  # the server answered "no data for this time range"
            request_meta = {
                "url": resp.url, "parameters": names, "start": start_str, "stop": stop_str,
                "retrieved_utc": _now(), "response_bytes": len(resp.content),
                "response_sha256": hashlib.sha256(resp.content).hexdigest(),
            }

        if text.strip():
            table = _parse_csv(text, names, fills)
        else:  # empty body: the server has no records in this window
            table = pd.DataFrame(columns=names, dtype=float,
                                 index=pd.DatetimeIndex([], tz="UTC", name="time_utc"))
        if not from_cache and text.strip():  # only validated responses are cached
            csv_path.write_text(text, encoding="utf-8")
            meta_path.write_text(json.dumps(request_meta, indent=2), encoding="utf-8")

        table = table.rename(columns={VARIABLES[v]: v for v in friendly})
        if "kp" in table.columns:
            table["kp"] = table["kp"] / _KP_SCALE
        table = table[friendly]

        grid = _expected_grid(start_ts, stop_ts)
        data = table.reindex(grid.union(table.index))  # absent rows become explicit NaN rows
        data.index.name = "time_utc"

        coverage = {
            "expected_rows": int(len(grid)),
            "rows_received": int(len(table)),
            "rows_absent_from_response": int(len(grid.difference(table.index))),
            "rows_outside_requested_window": int(len(table.index.difference(grid))),
            "valid_counts": {c: int(data[c].notna().sum()) for c in data.columns},
            "missing_counts": {c: int(data[c].isna().sum()) for c in data.columns},
            "dataset_stop_date_at_info_retrieval": info.get("stopDate"),
            "window_extends_beyond_dataset_end": False,
        }
        
        coverage["last_expected_time_utc"] = _fmt(grid[-1]) if len(grid) else None
        coverage["last_valid_time_utc"] = {
            c: (_fmt(data.index[data[c].notna().to_numpy()][-1]) if data[c].notna().any() else None)
            for c in data.columns
        }
        try:
            coverage["window_extends_beyond_dataset_end"] = bool(stop_ts > _to_utc(info["stopDate"]))
        except (KeyError, ValueError, TypeError):
            pass

        metadata = {
            "dataset_id": DATASET_ID,
            "dataset_doi": DATASET_DOI,
            "server": SERVER,
            "hapi_version": version,
            "time_convention": TIME_CONVENTION,
            "request": request_meta,
            "info_retrieved_utc": info_meta.get("retrieved_utc"),
            "variables": {
                v: {"hapi_name": VARIABLES[v], "units": UNITS[v],
                    "server_units": by_name[VARIABLES[v]].get("units"),
                    "fill_value": by_name[VARIABLES[v]].get("fill"),
                    "description": by_name[VARIABLES[v]].get("description"),
                    "conversion": "divided by 10 (server stores Kp*10)" if v == "kp" else None}
                for v in friendly
            },
            "notes": ["Missing values are NaN; fill values and absent hours are never replaced.",
                      "Dst quality (provisional / quick-look) depends on the period; see the "
                      "variable description."],
        }
        return OmniData(data, metadata, coverage, from_cache)