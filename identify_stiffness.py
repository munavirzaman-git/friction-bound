"""
identify_stiffness.py -- estimate the normalised front cornering stiffness c [1/rad]
from one or more CARLA logs written by carla_friction_bound.py.

Uses only linear-region samples (low force, low slip, steady speed), where
y = c * alpha regardless of friction. Fits y = c*alpha through the origin
with a robust (Huber-like, iteratively reweighted) least-squares fit and
reports a bootstrap interval.

Usage:
    python identify_stiffness.py runs/ID_*.csv
Then use the UPPER value printed (safe side: a model stiffness that is too
LOW makes the estimator overestimate friction):
    export CF_NORM=<value>
"""
import sys
import numpy as np
import pandas as pd


def robust_slope(a, y, iters=20, k=1.345):
    c = np.sum(a * y) / np.sum(a * a)
    for _ in range(iters):
        r = y - c * a
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = np.minimum(1.0, k * s / (np.abs(r) + 1e-12))
        c = np.sum(w * a * y) / np.sum(w * a * a)
    return c


def main(paths):
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    m = ((d.v > 5.0) & (d.alpha_f.abs() > 0.003) & (d.alpha_f.abs() < 0.03)
         & (d.y_meas.abs() < 0.3) & (d.ay.abs() < 3.0))
    a, y = d.alpha_f[m].to_numpy(), d.y_meas[m].to_numpy()
    print(f"samples used: {m.sum()} of {len(d)}")
    if m.sum() < 200:
        sys.exit("Too few linear-region samples: drive more curves at moderate speed.")
    c = robust_slope(a, y)
    rng = np.random.default_rng(0)
    boot = [robust_slope(a[i], y[i]) for i in
            (rng.integers(0, len(a), len(a)) for _ in range(300))]
    lo, hi = np.percentile(boot, [5, 95])
    print(f"c estimate     : {c:.2f} 1/rad")
    print(f"90% interval   : [{lo:.2f}, {hi:.2f}]")
    print(f"recommended    : export CF_NORM={hi:.2f}   (upper value = safe side)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
