"""probe the helioviewer api for lasco images and headers.

saves raw responses and a manifest under data/raw/helioviewer_probe/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image

API = "https://api.helioviewer.org/v2"
USER_AGENT = "cme-kinematics-toolkit/0.1 (student research project)"
TIMEOUT_S = 60
PAUSE_S = 1.0
# header keys that matter for geometry, time and instrument
KEY_PATTERNS = ("CRPIX", "CRVAL", "CDELT", "CROTA", "PC1_", "PC2_", "CD1", "CD2",
                "RSUN", "R_SUN", "SOLAR", "DSUN", "DATE", "NAXIS", "DETECTOR", "INSTR",
                "TELESCOP", "CTYPE", "CUNIT", "XCEN", "YCEN", "SUN", "ROLL", "EXPTIME",
                "BUNIT", "FILTER", "POLAR", "SCALE", "PLATE", "BITPIX", "OBSRVTRY")


def get(session, name, path, params, out_dir, manifest):
    url = f"{API}/{path}"
    try:
        resp = session.get(url, params=params, timeout=TIMEOUT_S)
    except requests.RequestException as exc:
        print(f"[{name}] request failed: {exc}")
        manifest.append({"name": name, "url": url, "params": params, "error": str(exc)})
        return None
    (out_dir / name).write_bytes(resp.content)
    manifest.append({
        "name": name, "url": resp.url, "status": resp.status_code,
        "content_type": resp.headers.get("Content-Type"),
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bytes": len(resp.content), "sha256": hashlib.sha256(resp.content).hexdigest(),
    })
    print(f"[{name}] http {resp.status_code}, {len(resp.content)} bytes, "
          f"{resp.headers.get('Content-Type')}")
    time.sleep(PAUSE_S)
    return resp


def find_sources(node, path=()):
    """yield (path, node) for every dict that has a sourceId."""
    if isinstance(node, dict):
        if "sourceId" in node:
            yield path, node
        for key, value in node.items():
            if isinstance(value, dict):
                yield from find_sources(value, path + (key,))


def show_header(text):
    pairs = re.findall(r"<([A-Za-z_][\w\-]*)>([^<]*)</\1>", text)
    print(f"    header: {len(pairs)} key/value pairs; geometry, time and instrument keys:")
    seen = set()
    for key, value in pairs:
        if key in seen:
            continue
        seen.add(key)
        if any(p in key.upper() for p in KEY_PATTERNS):
            print(f"      {key} = {value.strip()[:70]}")


def describe_image(path, preview_path):
    try:
        with Image.open(path) as im:
            im.load()
            arr = np.asarray(im)
            print(f"    pillow read ok: mode {im.mode}, size {im.size}, dtype {arr.dtype}, "
                  f"min {arr.min()}, max {arr.max()}, mean {arr.mean():.1f}")
            scaled = (arr.astype(float) - arr.min()) / (np.ptp(arr) or 1.0) * 255
            Image.fromarray(scaled.astype("uint8")).save(preview_path)
            print(f"    preview saved: {preview_path}")
    except Exception as exc:  # probe only: report whatever goes wrong
        print(f"    pillow could not read the jp2: {type(exc).__name__}: {exc}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", default="2003-10-28T11:30:00Z",
                        help="utc time of a lasco frame, e.g. 2003-10-28T11:30:00Z")
    parser.add_argument("--out-dir", default="data/raw/helioviewer_probe")
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    manifest: list[dict] = []

    print("== 1. data sources ==")
    resp = get(session, "data_sources.json", "getDataSources", {}, out, manifest)
    if resp is None or not resp.ok:
        return 1
    sources = resp.json()
    lasco = [(p, n) for p, n in find_sources(sources) if "LASCO" in p]
    if not lasco:
        print("no lasco entries found; top-level keys:", sorted(sources))
        return 1
    for path, node in lasco:
        print(f"  {'/'.join(path)}: sourceId {node['sourceId']}, "
              f"{node.get('start')} to {node.get('end')}")

    for path, node in lasco:
        tag = "_".join(path).replace(" ", "")
        sid = node["sourceId"]
        print(f"\n== 2. {'/'.join(path)} near {args.time} ==")
        resp = get(session, f"{tag}_closest.json", "getClosestImage",
                   {"date": args.time, "sourceId": sid}, out, manifest)
        if resp is None or not resp.ok:
            continue
        info = resp.json()
        print("    closest image:", json.dumps(info))
        resp = get(session, f"{tag}_header.xml", "getJP2Header",
                   {"id": info.get("id")}, out, manifest)
        if resp is not None and resp.ok:
            show_header(resp.text)
        resp = get(session, f"{tag}.jp2", "getJP2Image",
                   {"date": args.time, "sourceId": sid}, out, manifest)
        if resp is not None and resp.ok:
            describe_image(out / f"{tag}.jp2", out / f"{tag}_preview.png")

    (out / "probe_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nraw responses and probe_manifest.json saved in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())