"""Monte Carlo uncertainty analysis for each event in a CME observation CSV.

Events whose rows all have position uncertainties use them. Events without
uncertainties are skipped unless --assumed-sigma is given; that value is then
labelled as an ILLUSTRATIVE ASSUMPTION in every output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from cme_toolkit.data_loader import load_observations
from cme_toolkit.kinematics import elapsed_seconds
from cme_toolkit.uncertainty import (
    format_monte_carlo_report,
    monte_carlo_fits,
    summarize_monte_carlo,
)
from cme_toolkit.uncertainty_plots import plot_monte_carlo
from cme_toolkit.validation import DataValidationError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="path to the observation CSV")
    parser.add_argument("--trials", type=int, default=2000, help="Monte Carlo trials (default 2000)")
    parser.add_argument("--seed", type=int, default=12345, help="random seed (default 12345)")
    parser.add_argument("--assumed-sigma", type=float, default=None,
                        help="illustrative position uncertainty in R_sun for events that have none")
    parser.add_argument("--out-dir", default="results", help="output directory (default: results)")
    parser.add_argument("--no-plots", action="store_true", help="skip plot generation")
    args = parser.parse_args(argv)

    try:
        obs, report = load_observations(args.csv_path)
    except DataValidationError as exc:
        print(exc)
        return 1
    for issue in report.warnings:
        print(issue)

    out = Path(args.out_dir)
    reports, tables, plots = [], [], []
    for event_id, g in obs.groupby("event_id", sort=False):
        t = elapsed_seconds(g["timestamp"])
        r = g["position_solar_radii"].to_numpy(float)
        s = g["position_uncertainty_solar_radii"]
        if s.notna().all():
            sigma, assumed = s.to_numpy(float), False
        elif args.assumed_sigma is not None:
            sigma, assumed = args.assumed_sigma, True
        else:
            print(f"Skipping {event_id}: no position uncertainties in the file. Supply them, "
                  "or pass --assumed-sigma VALUE (R_sun) to use an illustrative assumption.")
            continue

        mc = monte_carlo_fits(t, r, sigma, n_trials=args.trials, seed=args.seed,
                              sigma_is_assumption=assumed)
        text = format_monte_carlo_report(event_id, mc)
        print(text + "\n")
        reports.append(text)

        table = summarize_monte_carlo(mc)
        table.insert(0, "event_id", event_id)
        table["n_trials"] = mc.n_trials
        table["seed"] = mc.seed
        table["sigma_is_assumption"] = mc.sigma_is_assumption
        tables.append(table)

        if not args.no_plots:
            synthetic = bool((g["data_origin"] == "synthetic").any())
            plots.append(plot_monte_carlo(event_id, mc, out / "plots", synthetic))

    if not tables:
        print("No events were analysed.")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(out / "monte_carlo_summary.csv",
                                                index=False, float_format="%.8g")
    (out / "monte_carlo_report.txt").write_text("\n\n".join(reports) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'monte_carlo_summary.csv'} and {out / 'monte_carlo_report.txt'}")
    for p in plots:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())