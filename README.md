# CME Kinematics and Space Weather Analysis Pipeline

Python toolkit for measuring and fitting the kinematics of coronal mass ejections (CMEs)
from coronagraph position-time data.

**Status: Phase 1 (kinematics calculator) is implemented. Everything else is planned.**

## Scientific problem

A CME's leading edge, seen in coronagraph images (e.g. SOHO/LASCO), traces a
position-time curve. From it we estimate speed and acceleration, which feed into
studies of CME propagation and space weather. Differentiating noisy positions
amplifies the noise, so this toolkit pairs finite differences with least-squares
trajectory fits and keeps measurement uncertainty explicit.

## My research background

*(Write 4-6 sentences in your own words: what you measured with JHelioviewer and
LASCO, how you recorded positions, how you compared with the CDAW catalog, and
what you found or learned.)*

## Implemented vs planned

| Feature | Status |
|---|---|
| CSV loading with explicit column mapping and strict validation | Implemented |
| Interval velocity and acceleration with uncertainty propagation | Implemented |
| Linear (constant velocity) and quadratic (constant acceleration) fits | Implemented |
| Model comparison (AICc, BIC) with cautions | Implemented |
| Plots and CSV/text export | Implemented |
| Monte Carlo uncertainty analysis | Planned |
| Solar-wind / geomagnetic data retrieval (OMNI) | Planned |
| Human-guided image measurement | Planned |
| CME / space-weather analysis | Planned |

The toolkit does not detect CMEs, predict geomagnetic storms, or establish causation.

## Install (Windows, PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run

```powershell
python scripts\make_sample_data.py
python scripts\run_analysis.py data\sample\synthetic_cme_observations.csv --out-dir results\sample
python -m pytest -v
```

## Input format

| Column | Required | Meaning |
|---|---|---|
| `event_id` | yes | Groups rows into one CME |
| `timestamp` | yes | ISO 8601, UTC (no zone is assumed UTC) |
| `position_solar_radii` | yes | Projected leading-edge distance, solar radii, > 0 |
| `instrument` | yes | e.g. `LASCO_C2` |
| `position_uncertainty_solar_radii` | no | 1-sigma, solar radii, > 0 |
| `source_image` | no | Image identifier |
| `data_origin` | no | e.g. `synthetic`, `manual_measurement` |

Rows must be in chronological order. Invalid data raises an error listing every problem with
its CSV line number; nothing is silently dropped or reordered.

## Outputs

`processed_observations.csv` (fits and residuals per observation), `finite_differences.csv`,
`fit_summary.csv`, `summary.txt`, and `plots/` (position, velocity, acceleration, residuals).

## Methods

- Solar radius: 695,700 km (IAU nominal). Time in seconds since the first observation.
- Interval velocity: `v = dr/dt` between consecutive points, assigned to the interval midpoint.
  Acceleration: the same difference applied to successive velocities. Irregular spacing is handled
  because each difference uses its own time step.
- Uncertainty propagation assumes independent position errors; acceleration is propagated from the
  three positions it depends on (neighbouring velocities share a point).
- Fits: weighted linear least squares, `r = r0 + v t` and `r = r0 + v0 t + a t^2 / 2`. With
  uncertainties, covariance is `(X^T W X)^-1`; without, it is scaled by `RSS/(n-k)`.
- Model comparison: AICc (difference of more than 2 counts as a preference), BIC reported. Not
  assessed below 5 points.

## Validation

- Synthetic trajectories with known velocity/acceleration are recovered to numerical precision.
- Fits agree with `scipy.optimize.curve_fit`.
- Monte Carlo test: reported 1-sigma intervals cover the true value about 68% of the time.
- Model comparison rarely claims acceleration for constant-velocity data.
- Run `python -m pytest -v`.

## Sample results

Sample data are **synthetic** (`data_origin = synthetic`); they are not CME observations. The
uncertainty (0.03 solar radii) is an illustrative assumption.

![Position](results/sample/plots/SYN_ACCEL_001_position.png)
![Acceleration](results/sample/plots/SYN_ACCEL_001_acceleration.png)

## Interpretation and limitations

- Positions are **projected** plane-of-sky distances; speeds are lower bounds on 3D speeds.
- Point-to-point accelerations are noise-dominated with realistic uncertainties. A better quadratic
  fit does not prove physical acceleration (projection, mis-picked edges, correlated or
  underestimated errors can mimic curvature).
- In the quadratic model, `velocity_km_s` is the speed at the first observation, not a launch speed.
- Constant acceleration over the whole field of view is a simplification.
- Only measurement noise is modeled; systematic and model errors are not.

## Data sources and attribution

Phase 1 uses no external data. Planned sources: SOHO/LASCO (via JHelioviewer / Helioviewer), the
CDAW CME catalog, and NASA OMNIWeb/CDAWeb. Check each provider's current acknowledgement wording
before adding real data, and do not commit large downloads.