"""Download OMNI hourly solar-wind and geomagnetic data for a time window.

Writes a CSV and a metadata JSON (source URL, retrieval time, units, fill values,
coverage) to data/processed/. Raw responses are cached in data/raw/omni_cache/,
and neither folder is committed to git.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cme_toolkit.solar_data import DataRetrievalError, OmniHapiClient


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="e.g. 2003-10-28 or 2003-10-28T00:00:00Z (UTC)")
    parser.add_argument("--stop", required=True, help="exclusive end, same formats")
    parser.add_argument("--cache-dir", default="data/raw/omni_cache")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")
    args = parser.parse_args(argv)

    try:
        result = OmniHapiClient(cache_dir=args.cache_dir).fetch(
            args.start, args.stop, force_refresh=args.refresh)
    except (DataRetrievalError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    cov, meta = result.coverage, result.metadata
    print(f"Source: {'local cache' if result.from_cache else 'NASA CDAWeb HAPI server'}")
    print(f"Request: {meta['request']['url']}")
    print(meta["time_convention"])
    print(f"Expected rows: {cov['expected_rows']}; received: {cov['rows_received']}; "
          f"absent from response: {cov['rows_absent_from_response']}; "
          f"outside window: {cov['rows_outside_requested_window']}")
    for name in result.data.columns:
        print(f"  {name:<11} valid {cov['valid_counts'][name]:>5}   missing {cov['missing_counts'][name]:>5}")
        last_expected = cov["last_expected_time_utc"]
    for name, last_valid in cov["last_valid_time_utc"].items():
        if last_valid is None:
            print(f"WARNING: {name} has NO valid data in this window.")
        elif last_expected and last_valid < last_expected:
            print(f"WARNING: {name} has no valid data after {last_valid}.")
    if cov["window_extends_beyond_dataset_end"]:
        print(f"WARNING: the window extends past the dataset end date "
              f"({cov['dataset_stop_date_at_info_retrieval']}).")
    if "dst_nt" in meta["variables"]:
        print("Dst note:", meta["variables"]["dst_nt"]["description"])
    print("\n" + result.data.head(6).to_string())

    stem = "omni_" + "_".join(meta["request"][k].replace("-", "").replace(":", "")
                              for k in ("start", "stop"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out / f"{stem}.csv", out / f"{stem}.json"
    result.data.to_csv(csv_path, index_label="time_utc", date_format="%Y-%m-%dT%H:%M:%SZ")
    json_path.write_text(json.dumps({"metadata": meta, "coverage": cov}, indent=2), encoding="utf-8")
    print(f"\nWrote {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())