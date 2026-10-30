# Excitation-aware friction bound — reproduction package

## 1. Setup
    python3 -m venv venv && source venv/bin/activate
    pip install numpy scipy matplotlib

## 2. Reproduce all simulation results (about 4 min on one CPU core)
    python run_experiments.py            # 20 seeds, writes ./results
    python run_experiments.py --quick    # 3-seed smoke test

Outputs in ./results:
- mc_results.csv     every run (scenario, method, seed, stiffness scale, metrics)
- summary.json       means, 95% CIs, departure rates, Wilcoxon p-values
- table_main.tex     rows of Table II in the paper
- sweep_ell.csv      staleness-length sweep
- sweep_c.csv        stiffness-mismatch sweep
- fig1_information.pdf, fig2_s1_timeseries.pdf, fig3_tradeoff.pdf

## 3. One run, interactively
    python -c "from fbsim import *; o=run(RunConfig(scenario='S1', method='P', record=True)); print(o['sat_time'], o['time'])"
Methods: B1 B2 B3 A1 A2 P.  Scenarios: S1 S2 S3.

## 4. CARLA validation (not yet run — test carefully)
    ./CarlaUE4.sh -quality-level=Low
    pip install carla==<your server version>
    python carla_friction_bound.py --method P  --scenario S1 --seed 0 --out runs/P_S1_0.csv
    python carla_friction_bound.py --method B3 --scenario S1 --seed 0 --out runs/B3_S1_0.csv
Before trusting CARLA numbers:
1. Identify the normalised front cornering stiffness (CF_NORM env var) with a
   slow steady-state circle test in CARLA; the method is unsafe if it is overestimated.
2. Check the centre-of-mass sign convention (marked TODO in the script).
3. Check that the friction trigger box covers the intended road section.

## 5. Build the paper
Upload ral_friction_bound.tex, the three figure PDFs and table data to Overleaf;
it switches to the official IEEEtran class automatically.
