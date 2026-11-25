"""Driven paths of the closed-loop trips against the reference route (Field experiment V).

Reads data/closed_loop/traces/*.csv (one file per trip, listed in index.csv; local metric frame: origin at the start
of the reference route, y along its first 5 m) and writes trajectories.png: for each
configuration the driven paths, and the lateral offset from the reference along the route.
A cross marks the point where the safety driver took over.
"""
import csv, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
tdir = os.path.join(here, "..", "data", "closed_loop", "traces")
out = sys.argv[1] if len(sys.argv) > 1 else "trajectories.png"
ORDER = ["DriveVLA-W0", "LTF", "Latent-WAM", "DriveVA", "ReCogDrive", "LTF_plans_held_0.5s",
         "LTF_plans_held_1.0s", "LTF_plans_held_2.0s", "LTF_side_views_blacked_out", "LTF_fine-tuned"]

def col(rows, k): return [float(r[k]) if r[k] != "" else float("nan") for r in rows]
ref = list(csv.DictReader(open(os.path.join(tdir, "reference_route.csv"))))
L = float(ref[-1]["route_s_m"])
trips = {}
for r in csv.DictReader(open(os.path.join(tdir, "index.csv"))):
    rows = list(csv.DictReader(open(os.path.join(tdir, r["file"]))))
    trips.setdefault(r["file"].rsplit("_trip", 1)[0], []).append((rows, r["reached_end"] == "1"))

fig, ax = plt.subplots(2, len(ORDER), figsize=(2.3 * len(ORDER), 7.2), gridspec_kw={"height_ratios": [2.2, 1]})
for j, name in enumerate(ORDER):
    a, b = ax[0, j], ax[1, j]
    a.plot(col(ref, "x_m"), col(ref, "y_m"), color="0.75", lw=6, solid_capstyle="round")
    done = 0
    for rows, end in trips.get(name, []):
        s = col(rows, "route_s_m"); done += end
        c = "tab:blue" if end else "tab:red"
        a.plot(col(rows, "x_m"), col(rows, "y_m"), color=c, lw=1.2)
        b.plot(s, col(rows, "lateral_m"), color=c, lw=1.0)
        if not end:
            a.plot(float(rows[-1]["x_m"]), float(rows[-1]["y_m"]), "x", color=c, ms=7, mew=2)
            b.plot(s[-1], [v for v in col(rows, "lateral_m") if v == v][-1], "x", color=c, ms=6, mew=2)
    n = len(trips.get(name, []))
    a.set_title(f"{name.replace('_', ' ')}\n{done} of {n} shown reach the end", fontsize=8)
    a.set_aspect("equal"); a.set_xlim(-25, 75); a.set_ylim(-10, 135); a.tick_params(labelsize=7)
    b.axhline(0, color="0.6", lw=0.6); b.set_xlim(0, L); b.set_ylim(-4, 4); b.tick_params(labelsize=7)
    b.set_xlabel("distance along route (m)", fontsize=7)
ax[0, 0].set_ylabel("y (m)"); ax[1, 0].set_ylabel("lateral offset (m), + right")
fig.suptitle("Grey: reference route. Blue: trips that reach the end. Red: trips ended by a takeover (cross).", fontsize=9)
fig.tight_layout(); fig.savefig(out, dpi=130); print("wrote", out)
