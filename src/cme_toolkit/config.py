"""Project-wide constants and the CSV schema definition."""

# IAU 2015 nominal solar radius, kilometres.
SOLAR_RADIUS_KM = 695_700.0

REQUIRED_COLUMNS = ("event_id", "timestamp", "position_solar_radii", "instrument")
OPTIONAL_COLUMNS = ("position_uncertainty_solar_radii", "source_image", "data_origin")
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Plausibility limits for a coronagraph leading edge. Violations produce
# WARNINGS (check your units), not errors.
PLAUSIBLE_MIN_RSUN = 1.5
PLAUSIBLE_MAX_RSUN = 35.0

# Velocity needs 2 points and acceleration 3, but a quadratic fit through
# exactly 3 points has zero degrees of freedom, so we recommend 5.
MIN_POINTS_REQUIRED = 3
MIN_POINTS_RECOMMENDED = 5