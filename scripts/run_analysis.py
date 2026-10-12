"""Run the CME kinematics analysis on a CSV of observations.

Writes processed_observations.csv, finite_differences.csv, fit_summary.csv,
summary.txt and plots/*.png into the output directory.
"""

from __future__ import annotations

import argparse
import sys

from cme_toolkit.pipeline import run_pipeline
from cme_toolkit.validation import DataValidationError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="path to the observation CSV")
    parser.add_argument("--out-dir", default="results", help="output directory (default: results)")
    parser.add_argument("--no-plots", action="store_true", help="skip plot generation")
    args = parser.parse_args(argv)

    try:
        result = run_pipeline(args.csv_path, args.out_dir, make_plots=not args.no_plots)
    except DataValidationError as exc:
        print(exc)
        return 1

    print(result.report_text)
    print("\nFiles written:")
    for path in result.files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())