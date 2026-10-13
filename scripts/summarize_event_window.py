"""Summarise near-Earth solar-wind and geomagnetic conditions in a window after a reference time.

The window is an explicit, recorded assumption; a disturbance inside it is not
evidence that a particular CME caused it. Defaults are illustrative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cme_toolkit.solar_data import DataRetrievalError, OmniHapiClient
from cme_toolkit.space_weather import (
    DEFAULT_BZ_THRESHOLD_NT,
    DEFAULT_MIN_VALID_FRACTION,
    format_window_report,
    summaries_to_frame,
    summarize_window,
    window_after,
)


def _json_safe(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-time", required=True,
                        help="UTC time the window is measured from, e.g. the first CME observation")
    parser.add_argument("--start-offset-h", type=float, default=12.0,
                        help="hours after the reference time at which the window starts (default 12)")
    parser.add_argument("--duration-h", type=float, default=120.0,
                        help="window length in hours (default 120)")
    parser.add_argument("--label", default="event window")
    parser.add_argument("--min-valid-fraction", type=float, default=DEFAULT_MIN_VALID_FRACTION)
    parser.add_argument("--bz-threshold-nt", type=float, default=DEFAULT_BZ_THRESHOLD_NT)
    parser.add_argument("--cache-dir", default="data/raw/omni_cache")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")
    args = parser.parse_args(argv)

    try:
        window = window_after(args.reference_time, args.start_offset_h, args.duration_h, args.label)
        result = OmniHapiClient(cache_dir=args.cache_dir).fetch(
            window.start, window.stop, force_refresh=args.refresh)
        summary = summarize_window(result.data, window,
                                   min_valid_fraction=args.min_valid_fraction,
                                   bz_threshold_nt=args.bz_threshold_nt)
    except (DataRetrievalError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    cov = result.coverage
    if cov["window_extends_beyond_dataset_end"]:
        summary.flags.append("the window extends past the dataset end date "
                             f"({cov['dataset_stop_date_at_info_retrieval']})")
    print(f"Reference time: {args.reference_time}; start offset {args.start_offset_h:g} h; "
          f"duration {args.duration_h:g} h")
    print(f"Data source: {'local cache' if result.from_cache else 'NASA CDAWeb HAPI server'}")
    print(result.metadata["time_convention"])
    print(format_window_report(summary))

    stem = f"space_weather_{window.start:%Y%m%dT%H%M}_{window.stop:%Y%m%dT%H%M}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out / f"{stem}.csv", out / f"{stem}.json"
    summaries_to_frame([summary]).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "reference_time": args.reference_time,
        "start_offset_h": args.start_offset_h,
        "duration_h": args.duration_h,
        "settings": summary.settings,
        "stats": _json_safe(summary.stats),
        "flags": summary.flags,
        "omni_metadata": result.metadata,
        "omni_coverage": cov,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    