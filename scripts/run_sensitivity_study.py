"""Run the synthetic acceleration sensitivity study and save a CSV and a plot.

All trajectories are SYNTHETIC; sigma, cadence and acceleration values are
illustrative assumptions, not LASCO calibration numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cme_toolkit.sensitivity import acceleration_study
from cme_toolkit.uncertainty_plots import plot_acceleration_study


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results", help="output directory (default: results)")
    parser.add_argument("--trials", type=int, default=500, help="trials per case (default 500)")
    parser.add_argument("--accel", type=float, default=25.0, help="true acceleration in m/s^2 (default 25)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    df = acceleration_study(
        sigmas_rsun=[0.01, 0.03, 0.1],
        n_points_list=[5, 10, 20],
        cadences_min=[6, 12, 24],
        a_ms2=args.accel, n_trials=args.trials, seed=args.seed,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "acceleration_study.csv"
    df.to_csv(csv_path, index=False, float_format="%.6g")
    png_path = plot_acceleration_study(df, out / "plots" / "acceleration_study.png")

    show = df[["sigma_rsun", "n_points", "cadence_min", "duration_h",
               "analytic_std_a_ms2", "mc_std_a_ms2", "a_over_sigma_a",
               "quadratic_favored_fraction"]]
    print("SYNTHETIC study (illustrative assumptions; true acceleration "
          f"{args.accel:g} m/s^2)")
    print(show.round(3).to_string(index=False))
    print(f"\nWrote {csv_path} and {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())