"""
carla_friction_bound.py -- CARLA validation of the excitation-aware friction bound.

NOT RUN by the author of this script (no CARLA server was available when it was
written). Test it step by step on your machine before using any result.

Requirements: CARLA 0.9.x server running, matching `carla` Python package,
and fbsim.py in the same folder (estimator and planner are reused unchanged).

Usage:
    ./CarlaUE4.sh -quality-level=Low            # start the server
    python carla_friction_bound.py --method P  --scenario S1 --out runs/P_S1.csv
    python carla_friction_bound.py --method B3 --scenario S1 --out runs/B3_S1.csv

What it does
    1. Loads Town04, builds a reference path from waypoints along one lane.
    2. For S1, places a friction trigger box (static.trigger.friction) over a
       straight section so friction drops from the default to --mu_low.
    3. Runs the same estimator (fbsim.FrictionEstimator) and speed planner
       (fbsim.plan_speed) as the Python study, with a Stanley-type steering law
       and a PI speed controller producing throttle/brake.
    4. Logs ground-truth wheel slip/force proxies, estimate, bound, speed,
       lateral error to CSV for the same metrics as the simulation.
"""
import argparse, csv, math, os, sys
import numpy as np

try:
    import carla
except ImportError:
    sys.exit("Install the CARLA Python API matching your server version.")

from fbsim import (EstimatorConfig, FrictionEstimator, PlannerConfig, plan_speed,
                   Track, G)


class WaypointTrack(Track):
    """Track built from CARLA waypoints (world frame converted to right-handed)."""
    def __init__(self, xs, ys, ds=0.25):
        self.ds = ds
        x = np.asarray(xs); y = -np.asarray(ys)       # CARLA y is to the right
        seg = np.hypot(np.diff(x), np.diff(y))
        s_raw = np.concatenate([[0.0], np.cumsum(seg)])
        self.s = np.arange(0.0, s_raw[-1], ds)
        self.x = np.interp(self.s, s_raw, x)
        self.y = np.interp(self.s, s_raw, y)
        dx, dy = np.gradient(self.x, ds), np.gradient(self.y, ds)
        self.psi = np.unwrap(np.arctan2(dy, dx))
        k = np.gradient(self.psi, ds)
        self.kappa = np.convolve(k, np.ones(41) / 41, mode="same")   # smooth
        self.length = self.s[-1]


def build_path(world, length=1200.0, step=2.0):
    m = world.get_map()
    wp = m.get_spawn_points()[0]
    wp = m.get_waypoint(wp.location)
    xs, ys, total = [], [], 0.0
    while total < length:
        xs.append(wp.transform.location.x); ys.append(wp.transform.location.y)
        nxt = wp.next(step)
        if not nxt:
            break
        wp = nxt[0]; total += step
    return WaypointTrack(xs, ys), m.get_waypoint(carla.Location(x=xs[0], y=ys[0]))


def place_friction_box(world, center, yaw_deg, friction, length=400.0, width=12.0):
    bp = world.get_blueprint_library().find("static.trigger.friction")
    bp.set_attribute("friction", str(friction))
    bp.set_attribute("extent_x", str(length * 50.0))   # extents are in cm (half-size)
    bp.set_attribute("extent_y", str(width * 50.0))
    bp.set_attribute("extent_z", str(500.0))
    tf = carla.Transform(center, carla.Rotation(yaw=yaw_deg))
    return world.spawn_actor(bp, tf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="P", choices=["B1", "B2", "B3", "A1", "A2", "P"])
    ap.add_argument("--scenario", default="S1", choices=["S1", "S3"])
    ap.add_argument("--mu_low", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/run.csv")
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print('connecting to CARLA...', flush=True)

    client = carla.Client(args.host, args.port); client.set_timeout(20.0)
    world = client.load_world("Town04")
    settings = world.get_settings()
    settings.synchronous_mode = True; settings.fixed_delta_seconds = 0.01
    world.apply_settings(settings)
    actors = []
    try:
        track, wp0 = build_path(world)
        bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        spawn_tf = carla.Transform(wp0.transform.location + carla.Location(z=0.5),
                                   wp0.transform.rotation)
        veh = world.try_spawn_actor(bp, spawn_tf)
        if veh is None:
            raise RuntimeError("Spawn failed even 0.5 m above the road; try another spawn point")
        actors.append(veh)
        for _ in range(20):          # let the car settle onto the road
            world.tick()
        world.tick()

        # vehicle parameters from CARLA physics (mass, wheelbase); stiffness from a
        # separate low-speed steady-state test -- REQUIRED before trusting results
        phys = veh.get_physics_control()
        m = phys.mass
        wheels = phys.wheels
        front = (wheels[0].position + wheels[1].position) / 2.0
        rear = (wheels[2].position + wheels[3].position) / 2.0
        L = front.distance(rear) / 100.0          # cm -> m
        com = phys.center_of_mass                  # relative to actor, metres
        lf, lr = L / 2.0 + com.x, L / 2.0 - com.x  # TODO verify sign on your version
        Iz = m * (lf * lr) * 1.1                   # TODO: rough; identify properly
        Fzf = m * G * lr / L
        cf_model = float(os.environ.get("CF_NORM", "9.5"))   # TODO identify, 1/rad
        delta_max = math.radians(wheels[0].max_steer_angle)

        if args.scenario == "S1":
            i_mid = int(len(track.s) * 0.65)
            c = carla.Location(x=float(track.x[i_mid]), y=float(-track.y[i_mid]), z=wp0.transform.location.z)
            actors.append(place_friction_box(world, c, -math.degrees(track.psi[i_mid]), args.mu_low))

        est_cfg = EstimatorConfig()
        if args.method == "A1":
            est_cfg.lam, est_cfg.regularise = 1.0, False
        est = FrictionEstimator(est_cfg, cf_model)
        pcfg = PlannerConfig(mu_min=est_cfg.mu_min)
        if args.method == "A2":
            pcfg.ell = float("inf")

        def mu_plan():
            return {"B1": 0.9, "B2": est_cfg.mu_min,
                    "B3": float(np.clip(est.mu_hat, est_cfg.mu_min, est_cfg.mu_max))}.get(
                        args.method, est.bound)

        s_cur, s_exc, t, r_prev = 0.0, 0.0, 0.0, 0.0
        plan_s, plan_v = plan_speed(track, 0.0, 0.0, mu_plan(), pcfg, s_exc)
        e_int = 0.0
        f = open(args.out, "w", newline=""); w = csv.writer(f)
        w.writerow(["t", "s", "v", "vref", "mu_hat", "mu_lb", "e_y", "ay", "alpha_f", "y_meas"])
        spectator = world.get_spectator()
        cam_pos, cam_yaw = None, None
        while s_cur < track.length - 10 and t < 300:
            world.tick(); t += 0.01
            tf = veh.get_transform(); vel = veh.get_velocity(); acc = veh.get_acceleration()
            ang = veh.get_angular_velocity()
            # chase camera: 10 m behind, 4 m up, exponentially smoothed
            yaw_c = tf.rotation.yaw
            if cam_yaw is None:
                cam_yaw = yaw_c
            dyaw = (yaw_c - cam_yaw + 180.0) % 360.0 - 180.0   # shortest angle
            cam_yaw += 0.1 * dyaw
            yr = math.radians(cam_yaw)
            target = np.array([tf.location.x - 10.0 * math.cos(yr),
                               tf.location.y - 10.0 * math.sin(yr),
                               tf.location.z + 4.0])
            cam_pos = target if cam_pos is None else cam_pos + 0.2 * (target - cam_pos)
            spectator.set_transform(carla.Transform(
                carla.Location(x=float(cam_pos[0]), y=float(cam_pos[1]), z=float(cam_pos[2])),
                carla.Rotation(pitch=-15.0, yaw=cam_yaw)))
            X, Y = tf.location.x, -tf.location.y
            psi = -math.radians(tf.rotation.yaw)
            vxw, vyw = vel.x, -vel.y
            vx = math.cos(psi) * vxw + math.sin(psi) * vyw
            vy = -math.sin(psi) * vxw + math.cos(psi) * vyw
            r = -math.radians(ang.z)                       # deg/s, left-handed -> rad/s
            axw, ayw = acc.x, -acc.y
            ay = -math.sin(psi) * axw + math.cos(psi) * ayw
            dr = (r - r_prev) / 0.01; r_prev = r

            s_c, e_y, psi_ref = track.project(X, Y, s_cur)
            s_cur = max(s_cur, s_c)
            e_psi = math.atan2(math.sin(psi - psi_ref), math.cos(psi - psi_ref))
            v = max(vx, 0.5)
            kap = track.interp(s_cur + 0.3 * v, track.kappa)
            delta = math.atan(L * kap) - 1.0 * e_psi - math.atan(2.5 * e_y / max(v, 2.0))
            delta = float(np.clip(delta, -delta_max, delta_max))

            # measurements (front axle) and estimator, as in fbsim
            alpha_f = delta - math.atan2(vy + lf * r, max(v, 1.0))
            Fyf = (m * lr * ay + Iz * dr) / (L * math.cos(delta))
            y_meas = Fyf / Fzf + rng.normal(0.0, 0.02)
            if v > 3.0:
                est.update(alpha_f + rng.normal(0.0, math.radians(0.1)), y_meas)
                if est.info_inc > pcfg.info_th:
                    s_exc = s_cur
            if int(round(t / 0.01)) % 20 == 0:
                plan_s, plan_v = plan_speed(track, s_cur, v, mu_plan(), pcfg,
                                            s_exc if args.method in ("A1", "P") else None)
            s_look = s_cur + max(v, 2.0) * 1.0          # 1 s preview, at least 2 m
            v_ref = float(np.interp(s_look, plan_s, plan_v))
            ev = v_ref - v; e_int = float(np.clip(e_int + ev * 0.01, -5, 5))
            u = 0.4 * ev + 0.05 * e_int
            ctrl = carla.VehicleControl(throttle=float(np.clip(u, 0, 1)),
                                        brake=float(np.clip(-u, 0, 1)),
                                        steer=float(np.clip(-delta / delta_max, -1, 1)))
            veh.apply_control(ctrl)
            w.writerow([t, s_cur, v, v_ref, est.mu_hat, est.bound, e_y, ay, alpha_f, y_meas])
            if int(round(t / 0.01)) % 100 == 0:          # once per simulated second
                f.flush()
                print(f"t={t:6.1f}s  s={s_cur:7.1f}/{track.length:.0f} m  v={v:5.2f}  vref={v_ref:5.2f}  "
                      f"e_y={e_y:6.2f}  mu_hat={est.mu_hat:.2f}  mu_lb={est.bound:.2f}", flush=True)

        f.close()
        print("saved", args.out, flush=True)
    finally:
        for a in actors:
            a.destroy()
        settings.synchronous_mode = False; world.apply_settings(settings)


if __name__ == "__main__":
    main()
