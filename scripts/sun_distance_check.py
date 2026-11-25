"""print sun-spacecraft distance and solar angular radius for utc times."""

from __future__ import annotations

import argparse
import sys

from cme_toolkit.image_geometry import AU_KM
from cme_toolkit.solar_geometry import (
    nominal_radius_bias,
    sun_earth_distance_km,
    sun_radius_arcsec_at,
    sun_soho_distance_km,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", nargs="+", required=True,
                        help="utc times, e.g. 2003-10-28T11:30:06Z")
    args = parser.parse_args(argv)

    print(f"{'time (utc)':<22}{'earth-sun':>10}{'soho-sun':>10}{'radius':>10}"
          f"{'bound':>8}{'nominal bias':>14}")
    print(f"{'':<22}{'(au)':>10}{'(au)':>10}{'(arcsec)':>10}{'(%)':>8}{'(%)':>14}")
    for t in args.time:
        try:
            earth = sun_earth_distance_km(t)
            soho, _ = sun_soho_distance_km(t)
            theta, rel = sun_radius_arcsec_at(t)
        except ValueError as exc:
            print(f"{t}: {exc}")
            continue
        print(f"{t:<22}{earth / AU_KM:>10.4f}{soho / AU_KM:>10.4f}{theta:>10.1f}"
              f"{100 * rel:>8.2f}{100 * nominal_radius_bias(theta):>14.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())