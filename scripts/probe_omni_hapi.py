"""Probe NASA CDAWeb's HAPI server for the OMNI hourly dataset (OMNI2_H0_MRG1HR).

Purpose: look at the REAL response formats before the toolkit's parser is
written, so nothing is guessed. Every raw response is saved, plus a manifest
(URL, status, UTC retrieval time, byte count, SHA-256), under data/raw/omni_probe/.
Makes 4 small requests with a pause between them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

SERVER = "https://cdaweb.gsfc.nasa.gov/hapi"
DATASET = "OMNI2_H0_MRG1HR"
WANTED = ["V1800", "BZ_GSM1800", "KP1800", "DST1800"]
TIMEOUT_S = 60
PAUSE_S = 1.0
USER_AGENT = "cme-kinematics-toolkit/0.1 (student research project)"


def fetch(session, name, path, params, out_dir, manifest):
    url = f"{SERVER}/{path}"
    try:
        resp = session.get(url, params=params, timeout=TIMEOUT_S)
    except requests.RequestException as exc:
        print(f"[{name}] REQUEST FAILED: {exc}")
        manifest.append({"name": name, "url": url, "params": params, "error": str(exc)})
        return None
    body = resp.content
    (out_dir / name).write_bytes(body)
    manifest.append({
        "name": name,
        "url": resp.url,
        "status": resp.status_code,
        "content_type": resp.headers.get("Content-Type"),
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    })
    print(f"[{name}] HTTP {resp.status_code}, {len(body)} bytes, {resp.headers.get('Content-Type')}")
    print(f"        {resp.url}")
    time.sleep(PAUSE_S)
    return resp


def request_data(session, out_dir, manifest, name, time_names, start, stop, parameters):
    tmin, tmax = time_names
    params = {"id": DATASET, tmin: start, tmax: stop, "parameters": ",".join(parameters)}
    return fetch(session, name, "data", params, out_dir, manifest)


def show_lines(resp, head, tail=0):
    lines = resp.text.splitlines()
    print(f"        {len(lines)} lines returned")
    for line in lines[:head]:
        print("   ", line)
    if tail:
        print("    ...")
        for line in lines[-tail:]:
            print("   ", line)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2003-10-28T00:00:00Z")
    parser.add_argument("--stop", default="2003-11-01T00:00:00Z")
    parser.add_argument("--out-dir", default="data/raw/omni_probe")
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    manifest: list[dict] = []

    print("== 1. capabilities ==")
    resp = fetch(session, "capabilities.json", "capabilities", None, out, manifest)
    version = None
    if resp is not None and resp.ok:
        try:
            caps = resp.json()
            version = str(caps.get("HAPI"))
            print("HAPI version reported:", version)
            print("output formats:", caps.get("outputFormats"))
        except ValueError:
            print("Response was not JSON; first 300 characters:\n", resp.text[:300])

    print("\n== 2. info (dataset metadata) ==")
    resp = fetch(session, "info_full.json", "info", {"id": DATASET}, out, manifest)
    wanted_in_order = list(WANTED)
    if resp is not None and resp.ok:
        try:
            info = resp.json()
        except ValueError:
            info = {}
        params = info.get("parameters", [])
        names = [p.get("name") for p in params]
        print("top-level keys:", sorted(info.keys()))
        print("startDate / stopDate:", info.get("startDate"), "/", info.get("stopDate"))
        print(f"{len(names)} parameters; first 6 in server order: {names[:6]}")
        if params:
            print("first parameter entry:", json.dumps(params[0]))
        for p in params:
            if p.get("name") in WANTED:
                keys = ("name", "type", "units", "fill", "length", "description")
                print("   ", {k: p.get(k) for k in keys})
        missing = [n for n in WANTED if n not in names]
        if missing:
            print("NOT FOUND in this dataset:", missing)
        wanted_in_order = [n for n in names if n in WANTED] or list(WANTED)
    print("Requesting parameters in this order:", wanted_in_order)

    print("\n== 3. data (parameters in server order) ==")
    primary = ("time.min", "time.max") if version and version.startswith("2") else ("start", "stop")
    other = ("start", "stop") if primary[0] == "time.min" else ("time.min", "time.max")
    resp = request_data(session, out, manifest, "data_native_order.csv",
                        primary, args.start, args.stop, wanted_in_order)
    if resp is not None and resp.status_code >= 400:
        print("Rejected. Server said:", resp.text[:300])
        print(f"Retrying with time parameter names {other} ...")
        primary = other
        resp = request_data(session, out, manifest, "data_native_order_retry.csv",
                            primary, args.start, args.stop, wanted_in_order)
    if resp is not None and resp.ok:
        show_lines(resp, head=6, tail=2)

    print("\n== 4. same request, parameters in REVERSED order ==")
    resp = request_data(session, out, manifest, "data_reversed_order.csv",
                        primary, args.start, args.stop, list(reversed(wanted_in_order)))
    if resp is not None and resp.ok:
        show_lines(resp, head=3)
    elif resp is not None:
        print("Server said:", resp.text[:300])

    (out / "probe_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nRaw responses and probe_manifest.json saved in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())