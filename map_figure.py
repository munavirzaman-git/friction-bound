"""
map_figure.py -- top-down figure of Town04 with the experiment route and the S1 wet patch.

Requires the CARLA server running (any map loaded; Town04 is loaded here).
    python map_figure.py            -> media/route_map.pdf / .png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import carla
from carla_friction_bound import build_path

plt.rcParams.update({"font.size": 8, "font.family": "serif"})
os.makedirs("media", exist_ok=True)

client = carla.Client("localhost", 2000); client.set_timeout(30.0)
world = client.get_world()
if "Town04" not in world.get_map().name:
    world = client.load_world("Town04")
cmap = world.get_map()

# all drivable lanes, as a light background
wps = cmap.generate_waypoints(2.0)
bx = np.array([w.transform.location.x for w in wps])
by = -np.array([w.transform.location.y for w in wps])     # right-handed

track, _ = build_path(world)

# same wet-patch rule as carla_friction_bound.py (scenario S1)
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
s0, s1 = track.s[b] + 60.0, track.s[min(e_curve, len(track.s) - 1)]
wet = (track.s >= s0) & (track.s <= s1)

fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.scatter(bx, by, s=0.2, color="0.85", lw=0)
ax.plot(track.x, track.y, color="k", lw=1.2, label="route (1490 m)")
ax.plot(track.x[wet], track.y[wet], color="C0", lw=3.0, alpha=0.7, label="wet patch (S1)")
ax.plot(track.x[0], track.y[0], "o", color="C2", ms=5, label="start")
ax.annotate("", xy=(track.x[40], track.y[40]), xytext=(track.x[0], track.y[0]),
            arrowprops=dict(arrowstyle="->", color="C2"))
ax.set_aspect("equal"); ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.legend(frameon=False, fontsize=6, loc="best")
fig.tight_layout()
fig.savefig("media/route_map.pdf"); fig.savefig("media/route_map.png", dpi=200)
print(f"saved media/route_map.pdf  (wet patch s = {s0:.0f}..{s1:.0f} m)")
