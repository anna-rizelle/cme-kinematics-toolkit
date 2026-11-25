"""overlay candidate sun centers and radius rings on a probed lasco jp2."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

NOMINAL_RSUN_ARCSEC = 959.64  # 1 au value implied by the helioviewer rsun field


def header_value(text, key):
    match = re.search(rf"<{key}>([^<]*)</{key}>", text)
    return float(match.group(1)) if match else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", default="data/raw/helioviewer_probe_mine")
    parser.add_argument("--tag", default="SOHO_LASCO_C2_white-light")
    args = parser.parse_args(argv)

    probe = Path(args.probe_dir)
    text = (probe / f"{args.tag}_header.xml").read_text(encoding="utf-8")
    cx, cy = header_value(text, "CRPIX1"), header_value(text, "CRPIX2")
    scale = header_value(text, "CDELT1")
    if None in (cx, cy, scale):
        print("missing CRPIX1, CRPIX2 or CDELT1 in the header")
        return 1
    with Image.open(probe / f"{args.tag}.jp2") as im:
        arr = np.asarray(im)
    height, width = arr.shape[:2]

    # array coordinates: x = column, y = row from the top, pixel centers at integers
    a = (cx - 1, height - cy)  # fits 1-based, y counted upward (image flipped)
    d = (cx - 1, cy - 1)       # fits 1-based, y counted downward (not flipped)
    print(f"image {width}x{height}; crpix = ({cx}, {cy}); scale {scale} arcsec/px")
    print(f"candidate A (flipped):     x = {a[0]:.3f}, y = {a[1]:.3f}")
    print(f"candidate D (not flipped): x = {d[0]:.3f}, y = {d[1]:.3f}")
    print("ring radii in pixels at 1..6 nominal R_sun:",
          [round(k * NOMINAL_RSUN_ARCSEC / scale, 1) for k in range(1, 7)])

    fig = Figure(figsize=(13, 6.5), constrained_layout=True)
    ax_full, ax_zoom = fig.subplots(1, 2)
    theta = np.linspace(0, 2 * np.pi, 361)
    for ax in (ax_full, ax_zoom):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        for k in range(1, 7):
            r = k * NOMINAL_RSUN_ARCSEC / scale
            ax.plot(a[0] + r * np.cos(theta), a[1] + r * np.sin(theta),
                    color="yellow", lw=0.6, ls="--")
        ax.plot(*a, "+", color="red", ms=16, mew=1.5, label="A: flipped")
        ax.plot(*d, "x", color="cyan", ms=11, mew=1.5, label="D: not flipped")
    ax_full.set_xlim(0, width - 1)
    ax_full.set_ylim(height - 1, 0)
    half = 220
    ax_zoom.set_xlim(a[0] - half, a[0] + half)
    ax_zoom.set_ylim(a[1] + half, a[1] - half)
    ax_zoom.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"{args.tag}: header center candidates (yellow rings at 1 to 6 nominal R_sun about A)")

    out_path = probe / f"{args.tag}_center_check.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())