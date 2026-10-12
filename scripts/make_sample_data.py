"""Write the synthetic sample dataset to data/sample/ (fixed seed, reproducible)."""

from pathlib import Path

import numpy as np
import pandas as pd

from cme_toolkit.synthetic import make_trajectory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample" / "synthetic_cme_observations.csv"


def main() -> None:
    rng = np.random.default_rng(42)
    t0 = pd.Timestamp("2000-01-01T00:00:00", tz="UTC")
    # 12-minute cadence with one missing frame (irregular spacing on purpose).
    times_s = [0, 720, 1440, 2160, 3600, 4320, 5040, 5760, 6480, 7200]
    sigma = 0.03  # R_sun; ILLUSTRATIVE assumption, not a real LASCO value

    const = make_trajectory("SYN_CONST_001", t0, times_s, 3.0, 450.0, 0.0, sigma, rng)
    accel = make_trajectory("SYN_ACCEL_001", t0 + pd.Timedelta(days=1), times_s,
                            3.0, 300.0, 25.0, sigma, rng)
    df = pd.concat([const, accel], ignore_index=True)
    df["position_solar_radii"] = df["position_solar_radii"].round(5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} SYNTHETIC rows to {OUT}")


if __name__ == "__main__":
    main()