"""
calibrate_friction.py -- map CARLA's `tire_friction` setting to an effective
tire-road friction coefficient mu (and normalised stiffness c).

For each tire_friction value it runs carla_friction_bound.py with method B1 and a
deliberately optimistic planning friction (--mu_fixed), so the car is driven hard
enough in corners to reach tyre saturation. It then fits
    y = mu * tanh(c * alpha / mu)
to the logged (alpha_f, y_meas) pairs.

Usage (CARLA server running):
    python calibrate_friction.py                       # default sweep
    python calibrate_friction.py --values 3.5 2 1.2 0.8 0.5
    python calibrate_friction.py --fit_only            # refit existing logs

Output: runs/calib/calibration.csv and a printed table.
A fit is reliable only if the log reaches saturation: check 'max_util' >= 0.9.
"""
import argparse, os, subprocess, sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def fit(path):
    d = pd.read_csv(path)
    m = (d.v > 8.0) & (d.y_meas.abs() < 3.0) & (d.e_y.abs() < 0.8) & (d.ay.abs() < 20.0)
    a, y = d.alpha_f[m].to_numpy(), d.y_meas[m].to_numpy()
    if len(a) < 300:
        return dict(n=len(a), c=np.nan, mu=np.nan, max_util=np.nan, ay_max_g=np.nan)
    res = least_squares(lambda p: y - p[1] * np.tanh(p[0] * a / p[1]), [12.0, 1.0],
                        bounds=([3.0, 0.05], [40.0, 4.0]), loss="soft_l1", f_scale=0.05)
    c, mu = res.x
    util = np.percentile(np.abs(y), 99.5) / mu
    ay99 = np.percentile(d.ay.abs(), 99.5) / 9.81
    return dict(n=len(a), c=c, mu=mu, max_util=util, ay_max_g=ay99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", type=float, nargs="+", default=[3.5, 2.5, 1.75, 1.2, 0.8, 0.5])
    ap.add_argument("--mu_fixed", type=float, default=2.5,
                    help="optimistic planning mu so corners are driven at the limit")
    ap.add_argument("--fit_only", action="store_true")
    ap.add_argument("--v_max", type=float, default=40.0)
    args = ap.parse_args()
    os.makedirs("runs/calib", exist_ok=True)
    rows = []
    for f in args.values:
        out = f"runs/calib/tf_{f:.2f}.csv"
        if not args.fit_only:
            print(f"\n=== tire_friction {f} ===", flush=True)
            cmd = [sys.executable, "carla_friction_bound.py", "--method", "B1",
                   "--scenario", "S3", "--mu_fixed", str(args.mu_fixed),
                   "--tire_friction", str(f), "--v_max", str(args.v_max), "--out", out]
            subprocess.run(cmd, check=False)
        if os.path.exists(out):
            r = fit(out); r["tire_friction"] = f; rows.append(r)
    if not rows:
        sys.exit("no logs found")
    df = pd.DataFrame(rows)[["tire_friction", "mu", "c", "max_util", "ay_max_g", "n"]]
    df.to_csv("runs/calib/calibration.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nReliable rows have max_util >= 0.9 (tyres reached saturation).")


if __name__ == "__main__":
    main()
