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
        x_i = np.interp(self.s, s_raw, x)
        y_i = np.interp(self.s, s_raw, y)
        k = 41                                   # 41 x 0.25 m = 10 m window
        ker = np.ones(k) / k
        self.x = np.convolve(np.pad(x_i, k // 2, mode="edge"), ker, mode="valid")
        self.y = np.convolve(np.pad(y_i, k // 2, mode="edge"), ker, mode="valid")
        dx, dy = np.gradient(self.x, ds), np.gradient(self.y, ds)
        self.psi = np.unwrap(np.arctan2(dy, dx))
        k = np.gradient(self.psi, ds)
        self.kappa = np.clip(np.convolve(k, np.ones(41) / 41, mode="same"), -0.2, 0.2)
        i = int(np.argmax(np.abs(self.kappa)))
        pass
        self.length = self.s[-1]


def build_path(world, length=1500.0, step=2.0, kappa_max=0.02):
    """Pick the first spawn point whose route (following the lane, preferring the
    branch that keeps heading at junctions) has max |kappa| < kappa_max."""
    m = world.get_map()
    sps = m.get_spawn_points()
    for i, sp in enumerate(sps):
        wp = m.get_waypoint(sp.location)
        if wp is None or wp.is_junction:
            continue
        xs, ys, total, ok = [], [], 0.0, True
        while total < length:
            xs.append(wp.transform.location.x); ys.append(wp.transform.location.y)
            nxt = wp.next(step)
            if not nxt:
                ok = False; break
            yaw0 = wp.transform.rotation.yaw
            wp = min(nxt, key=lambda w: abs((w.transform.rotation.yaw - yaw0 + 180.0) % 360.0 - 180.0))
            total += step
        if not ok or len(xs) < 100:
            continue
        tr = WaypointTrack(xs, ys)
        kmax = float(np.max(np.abs(tr.kappa)))
        if kmax < kappa_max:
            print(f"route: spawn point {i}, length {tr.length:.0f} m, max |kappa| {kmax:.4f} "
                  f"(min radius {1/kmax:.0f} m)", flush=True)
            return tr, m.get_waypoint(carla.Location(x=xs[0], y=ys[0]))
    raise RuntimeError(f"no route with max |kappa| < {kappa_max} found")


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
    ap.add_argument("--method", default="P", choices=["B1", "B2", "B3", "B4", "A1", "A2", "P"])
    ap.add_argument("--scenario", default="S1", choices=["S1", "S3"])
    ap.add_argument("--mu_low", type=float, default=0.4)
    ap.add_argument("--tire_friction", type=float, default=None,
                    help="set CARLA tire_friction on all wheels (calibration)")
    ap.add_argument("--v_max", type=float, default=25.0)
    ap.add_argument("--margin", type=float, default=0.2, help="B4 safety margin")
    ap.add_argument("--camera_dir", default=None, help="save chase-camera images here")
    ap.add_argument("--snap_s", type=float, nargs="*", default=[], help="distances [m] for snapshots")
    ap.add_argument("--video", action="store_true", help="also save every 4th frame for a video")
    ap.add_argument("--tf_wet", type=float, default=0.4, help="CARLA tire_friction inside the wet patch")
    ap.add_argument("--mu_fixed", type=float, default=0.9,
                    help="planning friction used by method B1")
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
        if args.tire_friction is not None:
            pc = veh.get_physics_control()
            wheels_list = pc.wheels                  # this is a COPY
            for wh in wheels_list:
                wh.tire_friction = args.tire_friction
            pc.wheels = wheels_list                  # assign it back
            veh.apply_physics_control(pc)
            for _ in range(10):
                world.tick()
            actual = [round(w.tire_friction, 3) for w in veh.get_physics_control().wheels]
            print(f"tire_friction requested {args.tire_friction}, actual {actual}", flush=True)
            assert all(abs(a - args.tire_friction) < 1e-3 for a in actual), "friction NOT applied" 
        phys = veh.get_physics_control()
        m = phys.mass
        wheels = phys.wheels
        # wheel positions are world coordinates in cm: project onto the actor's forward axis
        tfa = veh.get_transform(); fwd = tfa.get_forward_vector(); o = tfa.location
        def x_body(wh):
            return ((wh.position.x / 100.0 - o.x) * fwd.x +
                    (wh.position.y / 100.0 - o.y) * fwd.y)
        x_front = 0.5 * (x_body(wheels[0]) + x_body(wheels[1]))
        x_rear = 0.5 * (x_body(wheels[2]) + x_body(wheels[3]))
        com = phys.center_of_mass                  # actor frame, metres, x forward
        lf, lr = x_front - com.x, com.x - x_rear
        L = lf + lr
        print(f"geometry: m={m:.0f} kg  L={L:.3f}  lf={lf:.3f}  lr={lr:.3f} m", flush=True)
        assert lf > 0.5 and lr > 0.5, "axle geometry looks wrong -- check wheel order"
        pc2 = veh.get_physics_control()
        print("original steering_curve:", [(round(p.x, 1), round(p.y, 2)) for p in pc2.steering_curve], flush=True)
        pc2.steering_curve = [carla.Vector2D(x=0.0, y=1.0), carla.Vector2D(x=300.0, y=1.0)]
        veh.apply_physics_control(pc2)
        for _ in range(5):
            world.tick()
        cam, snaps, latest = None, sorted(args.snap_s), {"img": None}
        if args.camera_dir:
            os.makedirs(args.camera_dir, exist_ok=True)
            cbp = world.get_blueprint_library().find("sensor.camera.rgb")
            cbp.set_attribute("image_size_x", "960"); cbp.set_attribute("image_size_y", "540")
            cbp.set_attribute("fov", "90")
            cam = world.spawn_actor(cbp, carla.Transform(carla.Location(x=-8.0, z=3.5),
                                    carla.Rotation(pitch=-12.0)), attach_to=veh)
            actors.append(cam)
            cam.listen(lambda im: latest.__setitem__("img", im))
        print("steering_curve set flat; tire_friction now",
              [round(w.tire_friction, 2) for w in veh.get_physics_control().wheels], flush=True)
        Iz = m * (lf * lr) * 1.1                   # TODO: rough; identify properly
        Fzf = m * G * lr / L
        cf_model = float(os.environ.get("CF_NORM", "9.5"))   # TODO identify, 1/rad
        delta_max = math.radians(wheels[0].max_steer_angle)

        if args.scenario == "S1":
            # wet patch: from 60 m into the longest straight to the end of the next curve
            curved = np.abs(track.kappa) > 0.002
            runs, i = [], 0
            while i < len(curved):
                j = i
                while j < len(curved) and curved[j] == curved[i]:
                    j += 1
                runs.append((bool(curved[i]), i, j)); i = j
            straights = [(b, e) for c_, b, e in runs if not c_]
            b, e = max(straights, key=lambda r: r[1] - r[0])
            nxt = [r for r in runs if r[0] and r[1] >= e]
            e_curve = nxt[0][2] if nxt else len(track.s) - 1
            s_wet0, s_wet1 = track.s[b] + 60.0, track.s[min(e_curve, len(track.s) - 1)]
            cmap = world.get_map(); seg = 40.0
            for s_c in np.arange(s_wet0 + seg / 2, s_wet1, seg):
                k = int(s_c / track.ds)
                loc = carla.Location(x=float(track.x[k]), y=float(-track.y[k]))
                wpz = cmap.get_waypoint(loc)
                loc.z = wpz.transform.location.z if wpz else wp0.transform.location.z
                actors.append(place_friction_box(world, loc, -math.degrees(track.psi[k]),
                                                 args.tf_wet, length=seg + 2.0, width=16.0))
            print(f"wet patch: s = {s_wet0:.0f} .. {s_wet1:.0f} m, tire_friction {args.tf_wet}", flush=True)

        est_cfg = EstimatorConfig()
        if args.method == "A1":
            est_cfg.lam, est_cfg.regularise = 1.0, False
        est = FrictionEstimator(est_cfg, cf_model)
        pcfg = PlannerConfig(mu_min=est_cfg.mu_min, v_max=args.v_max)
        if args.method == "A2":
            pcfg.ell = float("inf")

        def mu_plan():
            return {"B1": args.mu_fixed, "B2": est_cfg.mu_min,
                    "B3": float(np.clip(est.mu_hat, est_cfg.mu_min, est_cfg.mu_max)),
                    "B4": float(np.clip(est.mu_hat - args.margin, est_cfg.mu_min, est_cfg.mu_max))}.get(
                        args.method, est.bound)

        s_cur, s_exc, t, r_prev = 0.0, 0.0, 0.0, 0.0
        plan_s, plan_v = plan_speed(track, 0.0, 0.0, mu_plan(), pcfg, s_exc)
        e_int = 0.0
        f = open(args.out, "w", newline=""); w = csv.writer(f)
        w.writerow(["t", "s", "v", "vref", "mu_hat", "mu_lb", "e_y", "ay", "alpha_f", "y_meas", "delta"])
        prog_s, prog_t = 0.0, 0.0
        spectator = world.get_spectator()
        cam_pos, cam_yaw = None, None
        ay_f, r_f, r_f_prev = 0.0, 0.0, 0.0
        a_lp = 0.01 / (0.05 + 0.01)
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
            ay_f += a_lp * (ay - ay_f)
            r_f += a_lp * (r - r_f)
            dr = (r_f - r_f_prev) / 0.01; r_f_prev = r_f

            s_c, e_y, psi_ref = track.project(X, Y, s_cur)
            if abs(e_y) > 3.0 and t > 2.0:
                print(f"STOP: left the lane (e_y={e_y:.1f} m) at t={t:.1f}s s={s_cur:.0f} m", flush=True)
                break
            s_cur = max(s_cur, s_c)
            if s_cur > prog_s + 1.0:
                prog_s, prog_t = s_cur, t
            elif t > 5.0 and t - prog_t > 8.0:
                print(f"STOP: stuck (no progress for 8 s) at t={t:.1f}s s={s_cur:.0f} m", flush=True)
                break
            e_psi = math.atan2(math.sin(psi - psi_ref), math.cos(psi - psi_ref))
            v = max(vx, 0.5)
            # pure pursuit from the rear axle, look-ahead = max(8 m, 1 s of travel)
            Ld = max(8.0, 1.0 * v)
            k_t = int(min((s_cur + Ld) / track.ds, len(track.s) - 1))
            xr = X + x_rear * math.cos(psi); yr = Y + x_rear * math.sin(psi)
            dx, dy = track.x[k_t] - xr, track.y[k_t] - yr
            a_pp = math.atan2(dy, dx) - psi
            a_pp = math.atan2(math.sin(a_pp), math.cos(a_pp))
            delta = math.atan(2.0 * L * math.sin(a_pp) / max(math.hypot(dx, dy), 1.0))
            delta = float(np.clip(delta, -delta_max, delta_max))

            # measurements (front axle) and estimator, as in fbsim
            d_fl = veh.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
            d_fr = veh.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
            delta_meas = -math.radians(0.5 * (d_fl + d_fr))   # left-handed -> ISO
            alpha_f = delta_meas - math.atan2(vy + x_front * r_f, max(v, 1.0))
            Fyf = (m * lr * ay_f + Iz * dr) / (L * math.cos(delta_meas))
            y_meas = Fyf / Fzf + rng.normal(0.0, 0.02)
            if v > 8.0 and abs(y_meas) < 1.3 and abs(ay_f) < 12.0:
                est.update(alpha_f + rng.normal(0.0, math.radians(0.1)), y_meas)
                if est.info_inc > pcfg.info_th:
                    s_exc = s_cur
            if int(round(t / 0.01)) % 20 == 0:
                plan_s, plan_v = plan_speed(track, s_cur, v, mu_plan(), pcfg,
                                            s_exc if args.method in ("A1", "P") else None)
            s_look = s_cur + max(v, 2.0) * 1.0          # 1 s preview, at least 2 m
            v_ref = float(np.interp(s_look, plan_s, plan_v))
            ev = v_ref - v; e_int = float(np.clip(e_int + ev * 0.01, -10, 10))
            u = 0.2 * ev + 0.05 * e_int
            thr_max = float(np.clip(1.0 - 0.15 * abs(ay_f), 0.25, 1.0))   # less throttle when cornering
            ctrl = carla.VehicleControl(throttle=float(np.clip(u, 0, thr_max)),
                                        brake=float(np.clip(-u, 0, 0.5)),
                                        steer=float(np.clip(-delta / delta_max, -1, 1)))
            veh.apply_control(ctrl)
            if cam is not None and latest["img"] is not None:
                k_frame = int(round(t / 0.01))
                if snaps and s_cur >= snaps[0]:
                    latest["img"].save_to_disk(f"{args.camera_dir}/snap_s{int(snaps[0]):04d}.jpg"); snaps.pop(0)
                if args.video and k_frame % 4 == 0:
                    latest["img"].save_to_disk(f"{args.camera_dir}/frame_{k_frame:06d}.jpg")
            w.writerow([t, s_cur, v, v_ref, est.mu_hat, est.bound, e_y, ay_f, alpha_f, y_meas, delta_meas])
            if int(round(t / 0.01)) % 100 == 0:          # once per simulated second
                f.flush()
                print(f"t={t:6.1f}s  s={s_cur:7.1f}/{track.length:.0f} m  v={v:5.2f}  vref={v_ref:5.2f}  "
                      f"e_y={e_y:6.2f}  dcmd={math.degrees(delta):5.2f} dmeas={math.degrees(delta_meas):5.2f}  mu_hat={est.mu_hat:.2f}  mu_lb={est.bound:.2f}", flush=True)

        f.close()
        print("saved", args.out, flush=True)
    finally:
        for a in actors:
            a.destroy()
        settings.synchronous_mode = False; world.apply_settings(settings)


if __name__ == "__main__":
    main()
