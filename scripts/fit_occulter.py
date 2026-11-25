"""fit the occulter circle in a probed lasco jp2 and compare it with the header center."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from cme_toolkit.image_geometry import fit_occulter, header_center_to_array

NOMINAL_RSUN_ARCSEC = 959.64  # value implied by the helioviewer rsun field


def header_value(text, key):
    match = re.search(rf"<{key}>([^<]*)</{key}>", text)
    return float(match.group(1)) if match else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", default="data/raw/helioviewer_probe_mine")
    parser.add_argument("--tag", default="SOHO_LASCO_C2_white-light")
    parser.add_argument("--rise", type=float, default=20.0,
                        help="gray levels above the disk interior that mark its edge")
    args = parser.parse_args(argv)

    probe = Path(args.probe_dir)
    text = (probe / f"{args.tag}_header.xml").read_text(encoding="utf-8")
    crpix1, crpix2 = header_value(text, "CRPIX1"), header_value(text, "CRPIX2")
    scale = header_value(text, "CDELT1")
    if None in (crpix1, crpix2, scale):
        print("missing CRPIX1, CRPIX2 or CDELT1 in the header")
        return 1
    with Image.open(probe / f"{args.tag}.jp2") as im:
        arr = np.asarray(im)
    height, width = arr.shape[:2]

    a = header_center_to_array(crpix1, crpix2, height)  # flipped
    d = (crpix1 - 1.0, crpix2 - 1.0)                    # not flipped
    print(f"{args.tag}: {width}x{height}, scale {scale} arcsec/px")
    print(f"header candidates: A (flipped) = ({a[0]:.2f}, {a[1]:.2f}); "
          f"D (not flipped) = ({d[0]:.2f}, {d[1]:.2f})")

    worked = False
    for name, guess in (("A", a), ("D", d), ("image center", (width / 2, height / 2))):
        try:
            fit = fit_occulter(arr, guess, rise=args.rise)
        except ValueError as exc:
            print(f"\nstarting from {name}: fit failed ({exc})")
            continue
        worked = True
        fx, fy = fit.center_xy
        print(f"\nstarting from {name}:")
        print(f"  fitted center = ({fx:.2f}, {fy:.2f}); radius {fit.radius_px:.1f} px "
              f"({fit.radius_px * scale / NOMINAL_RSUN_ARCSEC:.2f} nominal R_sun)")
        print(f"  edge points used {fit.n_used} of {fit.n_angles}; rms {fit.rms_px:.2f} px; "
              f"threshold {fit.threshold:.1f}")
        print(f"  distance to A: {np.hypot(fx - a[0], fy - a[1]):.1f} px; "
              f"to D: {np.hypot(fx - d[0], fy - d[1]):.1f} px")
    return 0 if worked else 1


if __name__ == "__main__":
    sys.exit(main())