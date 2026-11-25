"""sun-spacecraft distance and solar angular radius for lasco images.

soho is modelled as a fixed offset sunward of earth (l1). the offset and its
bound are approximations, not an ephemeris of the spacecraft.
"""

from __future__ import annotations

import pandas as pd
from astropy import units as u
from astropy.coordinates import get_sun
from astropy.time import Time

from .image_geometry import sun_angular_radius_arcsec

SOHO_L1_OFFSET_KM = 1.5e6         # sunward of earth (esa and nasa pages)
SOHO_OFFSET_BOUND_KM = 2.0e5      # halo orbit semi-axis along the sun-earth line
HELIOVIEWER_RSUN_ARCSEC = 959.64  # 1 au value implied by the helioviewer rsun field


def _astropy_time(value) -> Time:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return Time(ts.strftime("%Y-%m-%dT%H:%M:%S.%f"), format="isot", scale="utc")


def sun_earth_distance_km(time) -> float:
    """geocentric sun distance; naive times are taken as utc."""
    return float(get_sun(_astropy_time(time)).distance.to(u.km).value)


def sun_soho_distance_km(time):
    """(distance, bound) in km; the bound is a half-range, not a standard deviation."""
    return sun_earth_distance_km(time) - SOHO_L1_OFFSET_KM, SOHO_OFFSET_BOUND_KM


def sun_radius_arcsec_at(time):
    """(angular radius in arcsec, relative half-range from the soho offset)."""
    d, bound = sun_soho_distance_km(time)
    return sun_angular_radius_arcsec(d), bound / d


def nominal_radius_bias(theta_arcsec):
    """fractional overestimate of distances when the 1 au nominal radius is used."""
    return theta_arcsec / HELIOVIEWER_RSUN_ARCSEC - 1.0