import numpy as np
import pytest

from cme_toolkit.image_geometry import (
    ARCSEC_PER_RAD,
    AU_KM,
    array_to_rsun,
    fit_occulter,
    header_center_to_array,
    pixel_radius_to_rsun,
    rsun_to_pixel_radius,
    sun_angular_radius_arcsec,
)


def _disk(center, radius=60.0, size=300, outside=120.0, noise=2.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size), outside)
    img[np.hypot(xx - center[0], yy - center[1]) < radius] = 0.0
    return np.clip(img + rng.normal(0.0, noise, img.shape), 0.0, 255.0)


def _with_streak(img, center, angle_deg=40.0, r0=55.0, r1=140.0, half_width=3.0):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    ang = np.radians(angle_deg)
    u = (xx - center[0]) * np.cos(ang) + (yy - center[1]) * np.sin(ang)
    v = -(xx - center[0]) * np.sin(ang) + (yy - center[1]) * np.cos(ang)
    out = img.copy()
    out[(u > r0) & (u < r1) & (np.abs(v) < half_width)] = 0.0
    return out


def test_header_center_follows_fits_convention():
    assert header_center_to_array(512.634, 505.293, 1024) == pytest.approx((511.634, 518.707))
    # the center of a 4096-pixel image is 2048.5 in fits and 2047.5 in array coordinates
    assert header_center_to_array(2048.5, 2048.5, 4096) == pytest.approx((2047.5, 2047.5))


def test_sun_angular_radius_at_one_au():
    assert sun_angular_radius_arcsec(AU_KM) == pytest.approx(959.2, abs=0.15)
    assert sun_angular_radius_arcsec(AU_KM, radius_km=696_000.0) == pytest.approx(959.6, abs=0.15)


def test_sun_angular_radius_scales_inversely_with_distance():
    ratio = sun_angular_radius_arcsec(0.9 * AU_KM) / sun_angular_radius_arcsec(AU_KM)
    assert ratio == pytest.approx(1 / 0.9, rel=1e-4)


def test_distance_inside_the_sun_raises():
    with pytest.raises(ValueError, match="exceed"):
        sun_angular_radius_arcsec(600_000.0)


def test_pixel_radius_to_rsun_is_exact_at_large_angle():
    theta = sun_angular_radius_arcsec(0.983 * AU_KM)
    scale = 56.0
    alpha_arcsec = np.arctan(30.0 * np.sin(theta / ARCSEC_PER_RAD)) * ARCSEC_PER_RAD
    assert pixel_radius_to_rsun(alpha_arcsec / scale, scale, theta) == pytest.approx(30.0, rel=1e-9)
    assert pixel_radius_to_rsun(0.0, scale, theta) == 0.0


def test_rsun_pixel_round_trip():
    theta = sun_angular_radius_arcsec(0.99 * AU_KM)
    values = np.array([2.2, 6.0, 15.0, 30.0])
    back = pixel_radius_to_rsun(rsun_to_pixel_radius(values, 11.9, theta), 11.9, theta)
    assert np.allclose(back, values, rtol=1e-12)


def test_array_to_rsun_uses_distance_from_center():
    theta = sun_angular_radius_arcsec(AU_KM)
    expected = pixel_radius_to_rsun(100.0, 11.9, theta)
    assert array_to_rsun(300.0, 200.0, (200.0, 200.0), 11.9, theta) == pytest.approx(expected)
    assert array_to_rsun(200.0, 100.0, (200.0, 200.0), 11.9, theta) == pytest.approx(expected)


def test_fit_occulter_recovers_center_and_radius_from_a_poor_guess():
    center = (150.4, 148.7)
    fit = fit_occulter(_disk(center), (136.0, 160.0))
    assert fit.center_xy == pytest.approx(center, abs=0.5)
    assert fit.radius_px == pytest.approx(60.0, abs=1.5)
    assert fit.n_used >= 60


def test_fit_occulter_ignores_streak_outliers():
    center = (150.4, 148.7)
    fit = fit_occulter(_with_streak(_disk(center), center), (136.0, 160.0))
    assert fit.center_xy == pytest.approx(center, abs=0.7)


def test_fit_occulter_without_a_disk_raises():
    flat = np.full((300, 300), 120.0) + np.random.default_rng(1).normal(0, 2, (300, 300))
    with pytest.raises(ValueError, match="no dark disk"):
        fit_occulter(flat, (150.0, 150.0))


def test_fit_occulter_guess_in_bright_area_raises():
    with pytest.raises(ValueError, match="no dark disk"):
        fit_occulter(_disk((150.4, 148.7)), (250.0, 250.0))


def test_fit_occulter_guess_outside_image_raises():
    with pytest.raises(ValueError, match="outside"):
        fit_occulter(_disk((150.4, 148.7)), (400.0, 100.0))