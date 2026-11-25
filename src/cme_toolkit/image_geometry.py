"""pixel geometry for coronagraph images.

array coordinates: x = column, y = row from the top, pixel centers at integers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from . import config

AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206264.80624709636


def header_center_to_array(crpix1, crpix2, height):
    """fits reference pixel (1-based, y upward) to array coordinates."""
    return float(crpix1) - 1.0, float(height) - float(crpix2)


def sun_angular_radius_arcsec(distance_km, radius_km=config.SOLAR_RADIUS_KM):
    """apparent angular radius of the sun, arcsin(R / d), in arcsec."""
    if distance_km <= radius_km:
        raise ValueError("distance must exceed the solar radius.")
    return float(np.arcsin(radius_km / distance_km) * ARCSEC_PER_RAD)


def pixel_radius_to_rsun(r_px, scale_arcsec_px, sun_radius_arcsec):
    """projected distance in solar radii: tan(alpha) / sin(theta_sun)."""
    alpha = np.asarray(r_px, dtype=float) * scale_arcsec_px / ARCSEC_PER_RAD
    return np.tan(alpha) / np.sin(sun_radius_arcsec / ARCSEC_PER_RAD)


def rsun_to_pixel_radius(rsun, scale_arcsec_px, sun_radius_arcsec):
    """inverse of pixel_radius_to_rsun."""
    sin_theta = np.sin(sun_radius_arcsec / ARCSEC_PER_RAD)
    alpha = np.arctan(np.asarray(rsun, dtype=float) * sin_theta)
    return alpha * ARCSEC_PER_RAD / scale_arcsec_px


def array_to_rsun(x, y, center_xy, scale_arcsec_px, sun_radius_arcsec):
    """projected distance from the sun center in solar radii."""
    r_px = np.hypot(np.asarray(x, dtype=float) - center_xy[0],
                    np.asarray(y, dtype=float) - center_xy[1])
    return pixel_radius_to_rsun(r_px, scale_arcsec_px, sun_radius_arcsec)


@dataclass(frozen=True)
class OcculterFit:
    center_xy: tuple
    radius_px: float
    rms_px: float
    n_used: int
    n_angles: int
    threshold: float


def _ray_edges(img, guess, threshold, n_angles, r_max):
    """first threshold crossing along rays from the guess: list of (angle, radius)."""
    radii = np.arange(0.0, r_max, 0.5)
    edges = []
    for ang in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
        xs = guess[0] + radii * np.cos(ang)
        ys = guess[1] + radii * np.sin(ang)
        vals = gaussian_filter1d(map_coordinates(img, [ys, xs], order=1, mode="nearest"), 2.0)
        above = vals > threshold
        if not above.any():
            continue
        i = int(np.argmax(above))
        if i == 0:
            continue
        frac = (threshold - vals[i - 1]) / (vals[i] - vals[i - 1])
        edges.append((ang, radii[i - 1] + frac * 0.5))
    return edges


def fit_occulter(img, guess_xy, *, rise=20.0, n_angles=72, iterations=2):
    """fit a circle to the edge of the dark disk around guess_xy.

    the edge is where intensity first rises `rise` levels above the disk interior.
    raises ValueError if no clear disk is found.
    """
    img = np.asarray(img, dtype=float)
    if img.ndim != 2:
        raise ValueError("img must be a 2-d array.")
    h, w = img.shape
    cx, cy = float(guess_xy[0]), float(guess_xy[1])
    if not (0 <= cx < w - 1 and 0 <= cy < h - 1):
        raise ValueError("guess is outside the image.")
    x0, y0 = int(round(cx)), int(round(cy))
    inside = float(np.median(img[max(y0 - 4, 0):y0 + 5, max(x0 - 4, 0):x0 + 5]))
    threshold = inside + rise

    for _ in range(iterations):
        r_max = min(cx, cy, w - 1 - cx, h - 1 - cy) - 1.0
        if r_max < 10:
            raise ValueError("guess is too close to the image border.")
        edges = _ray_edges(img, (cx, cy), threshold, n_angles, r_max)
        if len(edges) < 12:
            raise ValueError("no dark disk with a clear edge found around the guess.")
        ang = np.array([e[0] for e in edges])
        rad = np.array([e[1] for e in edges])
        med = np.median(rad)
        mad = 1.4826 * np.median(np.abs(rad - med))
        keep = np.abs(rad - med) <= max(3.0 * mad, 3.0)  # drops pylon and streak rays
        if keep.sum() < 12:
            raise ValueError("edge points are too scattered to fit a circle.")
        px = cx + rad[keep] * np.cos(ang[keep])
        py = cy + rad[keep] * np.sin(ang[keep])
        design = np.column_stack([px, py, np.ones_like(px)])
        a, b, c = np.linalg.lstsq(design, -(px ** 2 + py ** 2), rcond=None)[0]
        cx, cy = -a / 2.0, -b / 2.0
        radius = float(np.sqrt(max(cx ** 2 + cy ** 2 - c, 0.0)))

    resid = np.hypot(px - cx, py - cy) - radius
    return OcculterFit((cx, cy), radius, float(np.sqrt(np.mean(resid ** 2))),
                       int(keep.sum()), len(edges), float(threshold))