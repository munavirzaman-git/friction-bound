"""
fbsim.py -- Excitation-aware friction-bounded speed planning: simulation library.

Everything needed to reproduce the simulation results of the manuscript:
track generation, dynamic bicycle model with saturating tyres, force/slip
measurement model, friction estimator with regularised forgetting, lower
confidence bound, forward-backward speed planner, path/speed controller,
and per-run metrics.

Units: SI throughout (m, s, kg, N, rad).
"""
from dataclasses import dataclass, field
import numpy as np

G = 9.81


# ----------------------------------------------------------------------------
# Vehicle and tyre
# ----------------------------------------------------------------------------
@dataclass
class Vehicle:
    m: float = 1500.0        # kg
    Iz: float = 2500.0       # kg m^2
    lf: float = 1.2          # m
    lr: float = 1.6          # m
    Cf: float = 80000.0      # N/rad, front axle cornering stiffness
    Cr: float = 100000.0     # N/rad, rear axle cornering stiffness
    delta_max: float = np.deg2rad(30.0)
    delta_rate: float = np.deg2rad(60.0)   # rad/s steering rate limit
    tau_delta: float = 0.08                # s steering lag

    @property
    def L(self):
        return self.lf + self.lr

    @property
    def Fzf(self):
        return self.m * G * self.lr / self.L

    @property
    def Fzr(self):
        return self.m * G * self.lf / self.L

    @property
    def cf(self):            # normalised front cornering stiffness [1/rad]
        return self.Cf / self.Fzf


def tyre_norm_force(alpha, c, mu):
    """Normalised lateral force y = F/Fz = mu * tanh(c*alpha/mu)."""
    return mu * np.tanh(c * alpha / mu)


def tyre_sensitivity(alpha, c, mu):
    """d h / d mu = tanh(u) - u sech^2(u), u = c alpha / mu."""
    u = c * alpha / mu
    return np.tanh(u) - u / np.cosh(u) ** 2


# ----------------------------------------------------------------------------
# Track: piecewise curvature with linear (clothoid) transitions
# ----------------------------------------------------------------------------
class Track:
    def __init__(self, segments, ds=0.25, transition=20.0):
        """segments: list of (length [m], curvature [1/m]) for constant parts.
        Linear curvature ramps of length `transition` are inserted between
        segments with different curvature."""
        kappa = []
        prev_k = segments[0][1]
        for length, k in segments:
            if k != prev_k:
                n_t = int(round(transition / ds))
                kappa.extend(np.linspace(prev_k, k, n_t, endpoint=False))
            kappa.extend([k] * int(round(length / ds)))
            prev_k = k
        self.ds = ds
        self.kappa = np.array(kappa)
        self.s = np.arange(len(self.kappa)) * ds
        self.psi = np.concatenate([[0.0], np.cumsum(self.kappa[:-1] * ds)])
        self.x = np.concatenate([[0.0], np.cumsum(np.cos(self.psi[:-1]) * ds)])
        self.y = np.concatenate([[0.0], np.cumsum(np.sin(self.psi[:-1]) * ds)])
        self.length = self.s[-1]

    def interp(self, s, arr):
        return np.interp(s, self.s, arr)

    def project(self, X, Y, s_guess, window=15.0):
        """Return (s, e_y, psi_ref) of the closest track point near s_guess."""
        i0 = max(0, int((s_guess - window) / self.ds))
        i1 = min(len(self.s), int((s_guess + window) / self.ds) + 1)
        dx = X - self.x[i0:i1]
        dy = Y - self.y[i0:i1]
        j = int(np.argmin(dx * dx + dy * dy)) + i0
        psi = self.psi[j]
        # signed lateral error, positive to the left of the path
        e_y = -np.sin(psi) * (X - self.x[j]) + np.cos(psi) * (Y - self.y[j])
        return self.s[j], e_y, psi


def benchmark_track():
    """Same geometry for all scenarios; only the friction profile changes."""
    R = 60.0
    segs = [
        (80.0, 0.0),
        (R * np.pi / 2, 1 / R),      # left 90 deg  (excitation on high mu)
        (80.0, 0.0),
        (R * np.pi / 2, -1 / R),     # right 90 deg
        (420.0, 0.0),                # long low-excitation straight
        (R * 2 * np.pi / 3, 1 / R),  # left 120 deg (test curve)
        (120.0, 0.0),
    ]
    return Track(segs)


def friction_profile(track, scenario):
    """True friction mu(s) for scenario S1, S2 or S3."""
    s = track.s
    if scenario == "S1":
        # high friction, then a drop 60 m into the long straight
        s_straight = 80 + 60 * np.pi / 2 + 80 + 60 * np.pi / 2 + 3 * 20.0
        mu = np.where(s < s_straight + 60.0, 0.9, 0.4)
    elif scenario == "S2":
        mu = np.full_like(s, 0.35)
    elif scenario == "S3":
        mu = np.full_like(s, 0.9)
    else:
        raise ValueError(scenario)
    return mu


# ----------------------------------------------------------------------------
# Friction estimator with regularised forgetting + lower confidence bound
# ----------------------------------------------------------------------------
@dataclass
class EstimatorConfig:
    lam: float = 0.995          # forgetting factor per update (100 Hz -> ~2 s memory)
    regularise: bool = True     # include (1-lam) J0 term
    mu0: float = 0.6            # prior mean
    mu_min: float = 0.3
    mu_max: float = 1.2
    z: float = 1.645            # one-sided 95 %
    R: float = 0.02 ** 2        # measurement noise variance on y

    @property
    def J0(self):
        sigma0 = (self.mu0 - self.mu_min) / self.z   # prior bound == mu_min
        return 1.0 / sigma0 ** 2


class FrictionEstimator:
    def __init__(self, cfg: EstimatorConfig, c_model: float):
        self.cfg = cfg
        self.c = c_model
        self.mu_hat = cfg.mu0
        self.J = cfg.J0
        self.info_inc = 0.0

    def update(self, alpha, y):
        cfg = self.cfg
        # (a) forgetting step: blend information (and its mean) with the prior
        if cfg.regularise:
            J_minus = cfg.lam * self.J + (1.0 - cfg.lam) * cfg.J0
            mu_minus = (cfg.lam * self.J * self.mu_hat
                        + (1.0 - cfg.lam) * cfg.J0 * cfg.mu0) / J_minus
        else:
            J_minus = cfg.lam * self.J
            mu_minus = self.mu_hat
        # (b) Gauss-Newton measurement update
        phi = tyre_sensitivity(alpha, self.c, mu_minus)
        self.info_inc = phi * phi / cfg.R
        self.J = J_minus + self.info_inc
        resid = y - tyre_norm_force(alpha, self.c, mu_minus)
        self.mu_hat = mu_minus + phi * resid / (self.J * cfg.R)
        self.mu_hat = float(np.clip(self.mu_hat, cfg.mu_min * 0.5, cfg.mu_max))

    @property
    def sigma(self):
        return 1.0 / np.sqrt(self.J)

    @property
    def bound(self):
        return max(self.cfg.mu_min, self.mu_hat - self.cfg.z * self.sigma)


# ----------------------------------------------------------------------------
# Forward-backward speed planner on the friction circle
# ----------------------------------------------------------------------------
@dataclass
class PlannerConfig:
    eta: float = 0.85           # safety factor on friction
    horizon: float = 250.0      # m
    ds: float = 2.0             # m
    v_max: float = 25.0         # m/s legal limit
    a_acc_max: float = 3.0      # m/s^2 powertrain limit
    ell: float = 150.0          # m spatial staleness length (np.inf disables)
    info_th: float = 1.0        # per-sample information marking an informative sample
    mu_min: float = 0.3


def plan_speed(track, s0, v0, mu_now, cfg: PlannerConfig, s_exc=None):
    """Return grid s, planned speed v(s) from s0 over the horizon."""
    s_grid = s0 + np.arange(0.0, cfg.horizon + cfg.ds, cfg.ds)
    s_grid = s_grid[s_grid <= track.length]
    if len(s_grid) < 2:
        s_grid = np.array([s0, s0 + cfg.ds])
    kap = np.abs(track.interp(s_grid, track.kappa))
    if np.isinf(cfg.ell):
        mu_plan = np.full_like(s_grid, mu_now)
    else:
        s_ref = s0 if s_exc is None else s_exc
        mu_plan = cfg.mu_min + (mu_now - cfg.mu_min) * np.exp(-(s_grid - s_ref) / cfg.ell)
    a_tot = cfg.eta * mu_plan * G
    v_lim = np.minimum(cfg.v_max, np.sqrt(a_tot / np.maximum(kap, 1e-6)))
    n = len(s_grid)
    v = v_lim.copy()
    # forward pass (acceleration)
    v[0] = v0                      # current speed is the initial condition
    for i in range(n - 1):
        ay = v[i] ** 2 * kap[i]
        ax = min(cfg.a_acc_max, np.sqrt(max(a_tot[i] ** 2 - ay ** 2, 0.0)))
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * ax * cfg.ds))
    # backward pass (braking); end of horizon: must be able to stop under mu_min
    v[-1] = min(v[-1], np.sqrt(2 * cfg.eta * cfg.mu_min * G * 30.0))
    for i in range(n - 1, 0, -1):
        ay = v[i] ** 2 * kap[i]
        ab = np.sqrt(max(a_tot[i - 1] ** 2 - ay ** 2, 0.0))
        v[i - 1] = min(v[i - 1], np.sqrt(v[i] ** 2 + 2 * ab * cfg.ds))
    return s_grid, v


# ----------------------------------------------------------------------------
# One closed-loop run
# ----------------------------------------------------------------------------
METHODS = ["B1", "B2", "B3", "A1", "A2", "P"]
METHOD_NAMES = {
    "B1": "B1 Fixed-nominal",
    "B2": "B2 Fixed-conservative",
    "B3": "B3 Certainty-equivalent",
    "A1": "A1 No forgetting",
    "A2": "A2 No spatial decay",
    "P": "Proposed",
}


@dataclass
class RunConfig:
    scenario: str = "S1"
    method: str = "P"
    seed: int = 0
    c_scale: float = 1.0        # true / model normalised cornering stiffness
    dt: float = 0.01
    replan_dt: float = 0.2
    sigma_y: float = 0.02       # noise on normalised force measurement
    sigma_alpha: float = np.deg2rad(0.1)   # noise on slip-angle estimate
    v_init: float = 10.0
    ell: float = None           # override planner staleness length (sweeps)
    record: bool = False


def run(cfg: RunConfig, veh: Vehicle = None):
    veh = veh or Vehicle()
    rng = np.random.default_rng(cfg.seed)
    track = benchmark_track()
    mu_true_s = friction_profile(track, cfg.scenario)

    est_cfg = EstimatorConfig(R=cfg.sigma_y ** 2)
    if cfg.method == "A1":
        est_cfg.lam, est_cfg.regularise = 1.0, False
    est = FrictionEstimator(est_cfg, c_model=veh.cf)
    plan_cfg = PlannerConfig(mu_min=est_cfg.mu_min)
    if cfg.ell is not None:
        plan_cfg.ell = cfg.ell
    if cfg.method == "A2":
        plan_cfg.ell = np.inf

    # true tyre stiffness may differ from the model the estimator uses
    Cf_true, Cr_true = veh.Cf * cfg.c_scale, veh.Cr * cfg.c_scale

    def mu_for_planning():
        if cfg.method == "B1":
            return 0.9
        if cfg.method == "B2":
            return est_cfg.mu_min
        if cfg.method == "B3":
            return float(np.clip(est.mu_hat, est_cfg.mu_min, est_cfg.mu_max))
        return est.bound

    # state: X, Y, psi, vx, vy, r, delta
    X, Y, psi = 0.0, 0.0, 0.0
    vx, vy, r, delta = cfg.v_init, 0.0, 0.0, 0.0
    s_cur = 0.0
    t = 0.0
    s_exc = 0.0
    plan_s, plan_v = plan_speed(track, 0.0, vx, mu_for_planning(), plan_cfg, s_exc)
    t_next_plan = cfg.replan_dt
    ax_prev = 0.0
    yaw_acc_prev = 0.0
    ay_prev = 0.0

    log = {k: [] for k in ["t", "s", "v", "vref", "mu_true", "mu_hat", "mu_lb",
                           "mu_plan", "util_f", "util_r", "e_y", "ay", "rho_meas"]}
    sat_time = 0.0
    e_sq, n_e = 0.0, 0
    e_max = 0.0
    covered, n_cov = 0, 0
    departed = False
    t_max = 200.0

    while s_cur < track.length - 5.0 and t < t_max:
        mu = float(track.interp(s_cur, mu_true_s))

        # ---------------- controller ----------------
        s_c, e_y, psi_ref = track.project(X, Y, s_cur)
        s_cur = max(s_cur, s_c)
        e_psi = np.arctan2(np.sin(psi - psi_ref), np.cos(psi - psi_ref))
        kap_prev = track.interp(s_cur + 0.3 * vx, track.kappa)
        delta_cmd = (np.arctan(veh.L * kap_prev) - 1.0 * e_psi
                     - np.arctan(2.5 * e_y / max(vx, 2.0)))
        delta_cmd = np.clip(delta_cmd, -veh.delta_max, veh.delta_max)
        ddelta = np.clip((delta_cmd - delta) / veh.tau_delta, -veh.delta_rate, veh.delta_rate)
        delta += ddelta * cfg.dt

        v_ref = float(np.interp(s_cur, plan_s, plan_v))
        v_ref_ahead = float(np.interp(s_cur + vx * 0.5, plan_s, plan_v))
        a_ff = (v_ref_ahead ** 2 - v_ref ** 2) / (2 * max(vx * 0.5, 0.5))
        ax_cmd = np.clip(a_ff + 1.0 * (v_ref - vx), -0.95 * G, 3.0)

        # ---------------- plant ----------------
        ax = float(np.clip(ax_cmd, -0.95 * mu * G, 0.95 * mu * G))
        mu_eff = np.sqrt(max(mu ** 2 - (ax / G) ** 2, (0.1 * mu) ** 2))
        alpha_f = delta - np.arctan2(vy + veh.lf * r, vx)
        alpha_r = -np.arctan2(vy - veh.lr * r, vx)
        Fyf = veh.Fzf * tyre_norm_force(alpha_f, Cf_true / veh.Fzf, mu_eff)
        Fyr = veh.Fzr * tyre_norm_force(alpha_r, Cr_true / veh.Fzr, mu_eff)
        dvx = ax + vy * r - Fyf * np.sin(delta) / veh.m
        dvy = (Fyf * np.cos(delta) + Fyr) / veh.m - vx * r
        dr = (veh.lf * Fyf * np.cos(delta) - veh.lr * Fyr) / veh.Iz
        ay = dvy + vx * r            # lateral acceleration (body frame)
        X += (vx * np.cos(psi) - vy * np.sin(psi)) * cfg.dt
        Y += (vx * np.sin(psi) + vy * np.cos(psi)) * cfg.dt
        psi += r * cfg.dt
        vx = max(vx + dvx * cfg.dt, 1.0)
        vy += dvy * cfg.dt
        r += dr * cfg.dt
        t += cfg.dt

        util_f = abs(Fyf) / (mu_eff * veh.Fzf)
        util_r = abs(Fyr) / (mu_eff * veh.Fzr)
        if max(util_f, util_r) > 0.95:
            sat_time += cfg.dt
        e_sq += e_y ** 2
        n_e += 1
        e_max = max(e_max, abs(e_y))
        if abs(e_y) > 2.0:
            departed = True

        # ---------------- measurement + estimation ----------------
        # front-axle force reconstructed from a_y and yaw acceleration (eq. 3)
        Fyf_meas = (veh.m * veh.lr * ay + veh.Iz * dr) / (veh.L * np.cos(delta))
        y_meas = Fyf_meas / veh.Fzf + rng.normal(0.0, cfg.sigma_y)
        alpha_meas = alpha_f + rng.normal(0.0, cfg.sigma_alpha)
        est.update(alpha_meas, y_meas)
        if est.info_inc > plan_cfg.info_th:
            s_exc = s_cur

        if cfg.method in ("A1", "A2", "P"):
            covered += int(mu >= est.bound)
            n_cov += 1

        # ---------------- replanning ----------------
        if t >= t_next_plan:
            plan_s, plan_v = plan_speed(track, s_cur, vx, mu_for_planning(), plan_cfg,
                                        s_exc if cfg.method in ('A1', 'P') else None)
            t_next_plan += cfg.replan_dt

        if cfg.record and int(round(t / cfg.dt)) % 5 == 0:
            log["t"].append(t); log["s"].append(s_cur); log["v"].append(vx)
            log["vref"].append(v_ref); log["mu_true"].append(mu)
            log["mu_hat"].append(est.mu_hat); log["mu_lb"].append(est.bound)
            log["mu_plan"].append(mu_for_planning())
            log["util_f"].append(util_f); log["util_r"].append(util_r)
            log["e_y"].append(e_y); log["ay"].append(ay)
            log["rho_meas"].append(abs(y_meas) / mu)

    out = dict(
        scenario=cfg.scenario, method=cfg.method, seed=cfg.seed, c_scale=cfg.c_scale,
        sat_time=sat_time,
        time=t,
        finished=bool(s_cur >= track.length - 5.0),
        rmse=float(np.sqrt(e_sq / max(n_e, 1))),
        e_max=float(e_max),
        departed=departed,
        coverage=(covered / n_cov) if n_cov else np.nan,
    )
    if cfg.record:
        out["log"] = {k: np.array(v) for k, v in log.items()}
        out["track"] = track
    return out
