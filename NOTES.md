# Progress notes — excitation-aware friction bound (target: IEEE RA-L)

## Python simulation (done, reproduced)
- `python run_experiments.py` -> results/ (20 seeds), identical to the paper numbers.
- S1 hidden drop: Proposed 0.2±0.1 s saturation, 0% departures; B3 31.5±3.2 s, 100% departures.
- Failure mode: model cornering stiffness LOWER than true -> friction overestimated (paper corrected).

## CARLA 0.9.16 setup (Town04, vehicle.tesla.model3)
- Geometry from wheel positions: m=1845 kg, L=3.005, lf=1.168, lr=1.836 m.
- Stiffness: c ≈ 10.6–11.4 1/rad (identify_stiffness.py); RE-IDENTIFIED after slip-angle fix: c = 14.54 (CI 14.51-14.56), use CF_NORM=14.56.
- Default CARLA grip (tire_friction 3.5) ≈ μ 3 -> unrealistic, not used.
- tire_friction must be set via pc.wheels = list (copy issue).
- Calibration on highway (peak lateral grip before departure):
  tf 1.0 -> 0.81 g (DRY),  tf 0.6 -> 0.35 g (WET),  tf 0.4 -> 0.18 g,  tf 0.3 -> 0.13 g.
- Route: highway loop, spawn point 2, 1490 m, min radius 165 m (city junction turns avoided).
- S1 wet patch: chain of 40 m friction boxes from 60 m into the longest straight to the end of the next curve
  (s ≈ 387–1070 m).
- v_max 30 m/s so the dry-belief baseline exceeds wet grip (0.55 g needed vs 0.35 g available).
- Steering: switched Stanley-type -> pure pursuit (look-ahead max(8 m, 1 s)) after
  high-speed oscillation; speed: PI with 1 s preview; estimator gated at v > 8 m/s.

## Next
1. S1 test with pure pursuit: B3 should slide only in the wet curve, P should finish.
2. Batch: 6 methods x 10 seeds for S1 and S3; add CARLA table + figure to the paper.
3. Remaining paper TODOs: affiliation, missing reference authors, AI disclosure, literature search.

## Commands
    export CF_NORM=11.5
    python carla_friction_bound.py --method P --scenario S1 --tire_friction 1.0 --tf_wet 0.6 --v_max 30 --out runs/s1_P.csv
