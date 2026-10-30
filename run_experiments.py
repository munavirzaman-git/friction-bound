"""
run_experiments.py -- reproduces every simulation number and figure in the manuscript.

Usage:
    python run_experiments.py            # full study (N=20 seeds)
    python run_experiments.py --quick    # N=3 smoke test

Outputs (in ./results):
    mc_results.csv, summary.json, sweep_ell.csv, sweep_c.csv,
    fig1_information.pdf, fig2_s1_timeseries.pdf, fig3_tradeoff.pdf,
    table_main.tex
"""
import argparse, csv, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fbsim import (RunConfig, run, METHODS, METHOD_NAMES, tyre_sensitivity,
                   Vehicle, PlannerConfig)
import fbsim

OUT = "results"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "figure.dpi": 150})


def mean_ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, h


def monte_carlo(n_seeds):
    rows = []
    rng = np.random.default_rng(12345)
    c_scales = rng.uniform(0.9, 1.1, size=n_seeds)   # same draws for every method
    for sc in ["S1", "S2", "S3"]:
        for m in METHODS:
            for k in range(n_seeds):
                o = run(RunConfig(scenario=sc, method=m, seed=1000 + k,
                                  c_scale=float(c_scales[k])))
                rows.append(o)
    with open(f"{OUT}/mc_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def summarise(rows):
    summ = {}
    for sc in ["S1", "S2", "S3"]:
        summ[sc] = {}
        for m in METHODS:
            R = [r for r in rows if r["scenario"] == sc and r["method"] == m]
            d = {}
            for key in ["sat_time", "time", "rmse", "e_max", "coverage"]:
                vals = [r[key] for r in R]
                if np.all(np.isnan(vals)):
                    d[key] = None
                else:
                    mu, h = mean_ci(vals)
                    d[key] = [float(mu), float(h)]
            d["departure_rate"] = float(np.mean([r["departed"] for r in R]))
            summ[sc][m] = d
    # paired comparison of saturation time, P vs B3, Wilcoxon signed-rank
    from scipy.stats import wilcoxon
    tests = {}
    for sc in ["S1", "S2", "S3"]:
        p = np.array([r["sat_time"] for r in rows if r["scenario"] == sc and r["method"] == "P"])
        b = np.array([r["sat_time"] for r in rows if r["scenario"] == sc and r["method"] == "B3"])
        diff = b - p
        if np.allclose(diff, 0):
            tests[sc] = None
        else:
            tests[sc] = float(wilcoxon(b, p).pvalue)
    summ["wilcoxon_sat_B3_vs_P"] = tests
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    return summ


def fmt(v, nd=2):
    if v is None:
        return "--"
    return f"{v[0]:.{nd}f}$\\pm${v[1]:.{nd}f}"


def latex_table(summ):
    lines = []
    for sc, title in [("S1", "S1 hidden drop"), ("S2", "S2 low friction"), ("S3", "S3 high friction")]:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}}\\\\")
        for m in METHODS:
            d = summ[sc][m]
            cov = "--" if d["coverage"] is None else f"{d['coverage'][0]:.2f}"
            lines.append(f"{METHOD_NAMES[m]} & {fmt(d['sat_time'],1)} & {d['departure_rate']*100:.0f} & "
                         f"{fmt(d['time'],1)} & {cov}\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    with open(f"{OUT}/table_main.tex", "w") as f:
        f.write("\n".join(lines))


def fig_information():
    rho = np.linspace(0.02, 0.98, 300)
    u = np.arctanh(rho)
    phi = np.tanh(u) - u / np.cosh(u) ** 2
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.semilogy(rho, phi ** 2 / (phi[-1] ** 2), label=r"exact $\phi^2$")
    ax.semilogy(rho, (2 / 3 * u ** 3) ** 2 / (phi[-1] ** 2), "--", label=r"$(\frac{2}{3}u^3)^2$")
    ax.set_xlabel(r"tyre utilisation $\rho$")
    ax.set_ylabel("information (norm.)")
    ax.set_ylim(1e-9, 2)
    ax.grid(True, which="both", lw=0.3)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_information.pdf"); fig.savefig(f"{OUT}/fig1_information.png")
    plt.close(fig)


def fig_timeseries():
    runs = {m: run(RunConfig(scenario="S1", method=m, seed=1000, record=True)) for m in ["B3", "P"]}
    fig, axs = plt.subplots(3, 1, figsize=(3.4, 4.2), sharex=True)
    L = runs["P"]["log"]
    axs[0].plot(L["s"], L["mu_true"], "k", lw=1, label=r"true $\mu$")
    axs[0].plot(L["s"], L["mu_hat"], "C0", lw=0.8, label=r"$\hat\mu$")
    axs[0].plot(L["s"], L["mu_lb"], "C1", lw=0.8, label=r"$\mu_{\rm lb}$")
    axs[0].set_ylabel(r"$\mu$"); axs[0].legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    for m, c in [("B3", "C3"), ("P", "C2")]:
        L = runs[m]["log"]
        axs[1].plot(L["s"], L["v"], c, lw=0.8, label=METHOD_NAMES[m])
        axs[2].plot(L["s"], np.maximum(L["util_f"], L["util_r"]), c, lw=0.8)
    axs[1].set_ylabel("speed [m/s]"); axs[1].legend(frameon=False, loc="lower left")
    axs[2].axhline(0.95, color="k", ls=":", lw=0.6)
    axs[2].set_ylabel("max axle util."); axs[2].set_xlabel("distance $s$ [m]")
    s_drop = 80 + 60 * np.pi / 2 + 80 + 60 * np.pi / 2 + 60 + 60
    for a in axs:
        a.axvline(s_drop, color="0.6", lw=0.6, ls="--")
        a.grid(True, lw=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_s1_timeseries.pdf"); fig.savefig(f"{OUT}/fig2_s1_timeseries.png")
    plt.close(fig)


def sweep_ell(n_seeds):
    rows = []
    for ell in [50.0, 100.0, 150.0, 300.0, np.inf]:
        for sc in ["S1", "S3"]:
            sat, tt = [], []
            for k in range(n_seeds):
                o = run(RunConfig(scenario=sc, method="P", seed=2000 + k, ell=ell))
                sat.append(o["sat_time"]); tt.append(o["time"])
            rows.append(dict(ell=ell, scenario=sc, sat=np.mean(sat), time=np.mean(tt)))
    with open(f"{OUT}/sweep_ell.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def sweep_c(n_seeds):
    rows = []
    for cs in [0.8, 0.9, 1.0, 1.1, 1.2]:
        for sc in ["S1", "S2"]:
            sat, cov = [], []
            for k in range(n_seeds):
                o = run(RunConfig(scenario=sc, method="P", seed=3000 + k, c_scale=cs))
                sat.append(o["sat_time"]); cov.append(o["coverage"])
            rows.append(dict(c_scale=cs, scenario=sc, sat=np.mean(sat), coverage=np.mean(cov)))
    with open(f"{OUT}/sweep_c.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def fig_tradeoff(summ):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    mk = {"B1": "s", "B2": "v", "B3": "D", "A1": "x", "A2": "+", "P": "o"}
    for m in METHODS:
        t1 = summ["S3"][m]["time"][0]       # cost of conservatism on high friction
        v1 = summ["S1"][m]["sat_time"][0]   # safety on hidden drop
        ax.plot(t1, v1, mk[m], ms=5, label=METHOD_NAMES[m])
    ax.set_xlabel("S3 traversal time [s] (lower = faster)")
    ax.set_ylabel("S1 saturation time [s]")
    ax.grid(True, lw=0.3); ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_tradeoff.pdf"); fig.savefig(f"{OUT}/fig3_tradeoff.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    N = 3 if args.quick else 20
    t0 = time.time()
    fig_information()
    rows = monte_carlo(N)
    summ = summarise(rows)
    latex_table(summ)
    fig_timeseries()
    fig_tradeoff(summ)
    se = sweep_ell(max(N // 2, 2))
    sc_ = sweep_c(max(N // 2, 2))
    print(json.dumps(summ, indent=1))
    print("ell sweep:", se)
    print("c sweep:", sc_)
    print(f"done in {time.time() - t0:.0f} s")
