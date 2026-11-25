import pandas as pd
import pytest

from cme_toolkit.image_geometry import AU_KM
from cme_toolkit.solar_geometry import (
    nominal_radius_bias,
    sun_earth_distance_km,
    sun_radius_arcsec_at,
    sun_soho_distance_km,
)

EVENT = "2003-10-28T11:30:06Z"


def test_earth_sun_distance_matches_the_helioviewer_aia_value():
    d = sun_earth_distance_km("2022-01-01T00:04:57Z")
    assert d == pytest.approx(147_091_270.0, abs=1e5)  # sdo orbits earth, hence the tolerance


def test_perihelion_and_aphelion_distances():
    assert sun_earth_distance_km("2023-01-04T16:00:00Z") / AU_KM == pytest.approx(0.9833, abs=5e-4)
    assert sun_earth_distance_km("2023-07-06T20:00:00Z") / AU_KM == pytest.approx(1.0167, abs=5e-4)


def test_soho_is_closer_than_earth_by_the_l1_offset():
    d, bound = sun_soho_distance_km(EVENT)
    assert sun_earth_distance_km(EVENT) - d == pytest.approx(1.5e6)
    assert bound == pytest.approx(2.0e5)


def test_angular_radius_on_the_2003_event_date():
    theta, rel = sun_radius_arcsec_at(EVENT)
    assert 973.0 < theta < 979.0
    assert 0.0012 < rel < 0.0015


def test_nominal_radius_overestimates_distances_when_the_sun_is_closer_than_1_au():
    theta, _ = sun_radius_arcsec_at(EVENT)
    assert 0.014 < nominal_radius_bias(theta) < 0.020
    theta_july, _ = sun_radius_arcsec_at("2023-07-06T20:00:00Z")
    assert nominal_radius_bias(theta_july) < 0


def test_naive_and_aware_times_agree():
    a = sun_earth_distance_km("2003-10-28T11:30:06")
    b = sun_earth_distance_km(pd.Timestamp("2003-10-28T11:30:06Z"))
    assert a == pytest.approx(b)


def test_invalid_time_raises():
    with pytest.raises(ValueError):
        sun_earth_distance_km("not a time")