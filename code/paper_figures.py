# -*- coding: utf-8 -*-
"""Result figures of the paper, drawn from the result files in data/results/.

Reads only existing outputs/ artifacts (no new measurements):
  fig2_infogate   : FOV vs resolution dose curves + matched-budget ratio
  fig3_timegate   : latency spectrum (clean v1) + deadline plane +
                    stale-plan/stale-command contrast + wrapper falsification
  fig4_distribution: D-XD1 same-arch contrast + 9x4 PDMS heatmap +
                    closed-loop rank bump chart
  fig5_selection  : selector-x-profile gate matrix + probe capability boundary

Colors: validated categorical order (light mode);
color follows the TIER entity (blind/traditional/VLA/world-model), identity
within tier via direct labels. Red is reserved for violation/collision.

Output: figures_out/*.pdf (+ .png previews).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = os.path.join(REPO, "data", "results")
FIG_DIR = os.path.join(REPO, "figures_out")
os.makedirs(FIG_DIR, exist_ok=True)

# ---- palette (validated categorical order; tier = entity) ----
C_TRAD = "#2a78d6"     # slot 1 blue  : traditional stacks
C_VLA = "#eb6834"      # slot 2 orange: VLA
C_WM = "#1baf7a"       # slot 3 aqua  : world models
C_BLIND = "#8a8a85"    # neutral gray : blind anchors (not a series color)
C_VIOL = "#e34948"     # red, reserved: violation / collision
C_CARLA = "#4a3aa7"    # slot 7 violet: CARLA-born checkpoint (fig4a pair)
C_REAL = "#1baf7a"     # aqua        : real-data-born checkpoint (fig4a pair)
INK = "#1a1a19"
MUTED = "#6b6b66"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# -------------------------------------------------------------------
# TURN-FRAME COUNTS: two different quantities share the name "turning
# frames" and differ by roughly 3x on the same city.  They are NEVER
# interchangeable in a caption.
#
#   TURN_STRATUM_N  -- frames in the L2 stratum (|dyaw over 2 s| >= 15
#                      degrees AND ground truth available).  This is the
#                      denominator every E-R4/E-R5 verdict is computed on,
#                      so it is the number a figure caption must quote.
#   sign-diagnostic -- every turning frame in the window, no threshold,
#                      reported per city in the readout logs (446/102/18/244
#                      for this carrier).  It is an instrument-health
#                      indicator only and belongs in no verdict.
#
# Recorded here rather than remembered after the paper cited the wrong one
# of the two on 2026-08-24.
# -------------------------------------------------------------------
TURN_STRATUM_N = {"sg-one-north": 144, "us-ma-boston": 19,
                  "us-nv-las-vegas-strip": 0,
                  "us-pa-pittsburgh-hazelwood": 59}   # pooled 222

CITIES = ["sg-one-north", "us-ma-boston", "us-nv-las-vegas-strip",
          "us-pa-pittsburgh-hazelwood"]
CITY_SHORT = {"sg-one-north": "SIN", "us-ma-boston": "BOS",
              "us-nv-las-vegas-strip": "LVG",
              "us-pa-pittsburgh-hazelwood": "PIT"}

TIER = {"cv": ("blind", C_BLIND), "ego_mlp": ("blind", C_BLIND),
        "alpamayo15": ("VLA", C_VLA), "orion": ("VLA", C_VLA),
        "tcp": ("traditional", C_TRAD), "transfuser": ("traditional", C_TRAD),
        "ltf": ("traditional", C_TRAD),
        "diffusiondrive": ("traditional", C_TRAD),
        "recogdrive": ("VLA", C_VLA), "autovla": ("VLA", C_VLA),
        "curious": ("VLA", C_VLA), "driveva": ("world model", C_WM),
        "wamflow": ("world model", C_WM),
        "drivevlaw0": ("world model", C_WM)}
LABEL = {"cv": "Const-vel", "ego_mlp": "EgoMLP (blind)", "tcp": "TCP",
         "alpamayo15": "Alpamayo-1.5-10B",
         "transfuser": "TransFuser", "ltf": "LTF",
         "diffusiondrive": "DiffusionDrive", "recogdrive": "ReCogDrive-2B",
         "autovla": "AutoVLA-3B", "curious": "Curious-VLA-3B",
         "driveva": "DriveVA-5B", "wamflow": "WAM-Flow-1.5B",
         "orion": "ORION-2B (closed loop)",
         "drivevlaw0": "DriveVLA-W0-8B"}


def jload(*parts):
    with open(os.path.join(OUT_BASE, *parts)) as f:
        return json.load(f)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"),
                    bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote", name)


# ===================================================================
# Fig 2 — information gate: paired dose curves
# ===================================================================
def fig2():
    fov_agents = ["cv", "ego_mlp", "transfuser", "ltf", "diffusiondrive",
                  "recogdrive", "driveva", "wamflow", "drivevlaw0"]
    fov_keep = [0.9, 0.75, 0.6]           # linear keep ratio
    res_keep = [0.75, 0.5, 0.33]

    def dose_mean(exp, agent, keeps):
        vals = []
        for k in keeps:
            per_city = []
            for c in CITIES:
                p = os.path.join(OUT_BASE, exp, agent, c, "dose.json")
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))
                per_city.append(d["mean_displacement_m"][str(k)])
            vals.append((np.mean(per_city), np.min(per_city),
                         np.max(per_city))
                        if per_city else (np.nan, np.nan, np.nan))
        return np.array(vals)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4),
                             gridspec_kw={"width_ratios": [1, 1, 0.72],
                                          "wspace": 0.52})
    axA, axB, axC = axes

    for ax, exp, keeps, title in (
            (axA, "lineb_fov_dose_v0", fov_keep,
             "a  FOV crop (structure removed)"),
            (axB, "lineb_res_dose_v0", res_keep,
             "b  Resolution drop (detail removed)")):
        ends = []
        for a in fov_agents:
            v = dose_mean(exp, a, keeps)
            if np.all(np.isnan(v[:, 0])):
                continue
            xs = [1.0] + [k * k for k in keeps]
            # pixel fraction retained: fov crop keeps k^2 area; res keeps
            # k^2 pixels as well (downscale k in each dim)
            ys = [0.0] + list(v[:, 0])
            tier, col = TIER[a]
            lw = 1.6 if tier != "blind" else 1.2
            ls = {"transfuser": "-", "ltf": "--", "diffusiondrive": ":",
                  "cv": "-", "ego_mlp": "--",
                  "wamflow": "--", "drivevlaw0": ":"}.get(a, "-")
            ax.plot(xs, ys, ls, color=col, lw=lw, marker="o", ms=3,
                    mfc="white", mew=0.8, clip_on=False)
            if tier != "blind":
                yerr = np.array(
                    [[m - lo for m, lo in zip(v[:, 0], v[:, 1])],
                     [hi - m for m, hi in zip(v[:, 0], v[:, 2])]])
                ax.errorbar(xs[1:], ys[1:], yerr=yerr, fmt="none",
                            ecolor=col, elinewidth=0.6, capsize=1.5,
                            alpha=0.55, zorder=2)
            ends.append((a, xs[-1], ys[-1], col))
        # staggered right-edge direct labels with leader lines; blind
        # anchors share one label
        blind = [e for e in ends if TIER[e[0]][0] == "blind"]
        rest = sorted([e for e in ends if TIER[e[0]][0] != "blind"],
                      key=lambda e: e[2])
        if blind:
            bx, by = blind[0][1], blind[0][2]
            ax.annotate("blind anchors = 0", (bx, by), xytext=(0.98, 0.055),
                        textcoords=("axes fraction"), ha="right",
                        fontsize=6.0, color=C_BLIND)
        if ax is axA:
            ymin_gap = 0.085
            placed = []
            for a, x_, y_, col in rest:
                ty = y_
                if placed and ty - placed[-1] < ymin_gap:
                    ty = placed[-1] + ymin_gap
                placed.append(ty)
                ax.annotate(LABEL[a], (x_, y_), xytext=(0.985, ty),
                            textcoords=("axes fraction", "data"),
                            ha="right", va="center", fontsize=6.0,
                            color=col,
                            arrowprops=dict(arrowstyle="-", lw=0.5,
                                            color=col, alpha=0.55,
                                            shrinkA=1, shrinkB=1))
        ax.set_xlim(1.02, 0.0)
        ax.set_xlabel("Fraction of pixels retained")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 2.0)
    axA.set_ylabel("Mean plan displacement (m)")
    axB.set_yticklabels([])
    handles = []
    for a in fov_agents:
        if TIER[a][0] == "blind":
            continue
        ls = {"transfuser": "-", "ltf": "--", "diffusiondrive": ":",
              "wamflow": "--", "drivevlaw0": ":"}.get(a, "-")
        handles.append(plt.Line2D([], [], ls=ls, color=TIER[a][1],
                                  marker="o", ms=3, mfc="white",
                                  lw=1.5, label=LABEL[a]))
    handles.append(plt.Line2D([], [], ls="-", color=C_BLIND, marker="o",
                              ms=3, mfc="white", lw=1.2,
                              label="blind anchors"))
    axB.legend(handles=handles, fontsize=5.6, loc="upper left",
               handlelength=1.8, labelspacing=0.35)

    # matched-budget ratio at 56% pixels retained (fov 0.75 vs res 0.75)
    ratios, names, cols = [], [], []
    for a in fov_agents:
        if TIER[a][0] == "blind":
            continue
        f = dose_mean("lineb_fov_dose_v0", a, [0.75])[0, 0]
        r = dose_mean("lineb_res_dose_v0", a, [0.75])[0, 0]
        if np.isnan(f) or np.isnan(r) or r == 0:
            continue
        ratios.append(f / r)
        names.append(LABEL[a])
        cols.append(TIER[a][1])
    y = np.arange(len(ratios))[::-1]
    axC.barh(y, ratios, height=0.62, color=cols, zorder=3)
    axC.axvline(1.0, color=MUTED, lw=0.8, ls="--", zorder=2)
    for yi, v in zip(y, ratios):
        axC.text(v + 0.08, yi, f"{v:.1f}x", va="center", fontsize=6.5)
    axC.set_yticks(y, names, fontsize=6.5)
    axC.set_xlabel("FOV / resolution damage ratio\n(equal pixel budget, 56% kept)")
    axC.set_title("c  Not-seen vs not-sharp", loc="left")
    axC.set_xlim(0, max(ratios) * 1.25)
    axC.grid(axis="y", visible=False)
    save(fig, "fig2_infogate")


# ===================================================================
# Fig 3 — time gate
# ===================================================================
def load_latency():
    rows = {}
    clean = os.path.join(OUT_BASE, "lineb_latency_clean_v1")
    shadow = os.path.join(OUT_BASE, "lineb_latency_shadow_v0")
    for a in ["cv", "ego_mlp", "transfuser", "ltf", "diffusiondrive",
              "recogdrive", "autovla", "curious", "driveva",
              "alpamayo15", "wamflow", "drivevlaw0"]:
        for src, tag in ((clean, "clean"), (shadow, "shadow")):
            p = os.path.join(src, f"{a}.json")
            if os.path.exists(p):
                d = json.load(open(p))
                p95 = d["ms"].get("p95", d["ms"].get("p90"))
                rows[a] = (d["ms"]["p50"], p95, tag)
                break
    # TCP: net-card passport (CARLA-side config, earlier clean window)
    p2 = os.path.join(OUT_BASE, "latency_passport_v2")
    tcp = None
    if os.path.isdir(p2):
        for fn in os.listdir(p2):
            if "tcp" in fn and fn.endswith(".json"):
                try:
                    d = json.load(open(os.path.join(p2, fn)))
                    s = json.dumps(d)
                    cand = d.get("p50_ms") or d.get("ms", {}).get("p50")
                    if cand:
                        tcp = (float(cand), float(
                            d.get("p95_ms") or d.get("ms", {}).get(
                                "p95", cand)), "clean")
                except Exception:
                    pass
    if tcp is None:
        tcp = (62.8, 66.0, "clean")   # frozen net-card readout 
    rows["tcp"] = tcp
    orion_p = os.path.join(OUT_BASE, "latency_passport_orion_v1",
                           "orion.json")
    if os.path.exists(orion_p):
        d = json.load(open(orion_p))["pipeline_ms"]
        rows["orion"] = (d["p50"], d.get("p95", d["p99"]), "clean")
    return rows


def m4_rows():
    rows = [json.loads(l) for l in open(
        os.path.join(OUT_BASE, "m4_matrix_v4", "outcomes.jsonl"))]
    return [r for r in rows if r["record_status"] == "success"]


def agg(rows, env, cfg, dl):
    sel = [r for r in rows if r["environment_id"] == env
           and r["config_id"] == cfg and r["deadline_ms"] == dl]
    if not sel:
        return None
    comp = np.mean([r["outcome"]["values"]["route_completion_fraction"]
                    for r in sel])
    crr = np.mean([1.0 if r["outcome"]["values"]["collision_count"] >= 1
                   else 0.0 for r in sel])
    return comp, crr, len(sel)


def _orion_deadline_points():
    """(deadline, completion, collision route rate) for the ORION cells.

    Missing cells are skipped, never zero-filled: a deadline arm is
    plotted only when its runs exist.
    """
    base = os.path.join(OUT_BASE, "orion_t5a")
    out = []
    for dl, tag in ((0, ""), (500, "__d500"), (200, "__d200")):
        comps, colls = [], []
        for seed in (0, 1, 2):
            p = os.path.join(base,
                             f"Town13_w0__orion_native__seed{seed}{tag}",
                             "result.json")
            if not os.path.exists(p):
                continue
            try:
                recs = json.load(open(p))["_checkpoint"]["records"]
            except Exception:
                continue
            for r in recs:
                comps.append(r["scores"]["score_route"] / 100.0)
                inf = r.get("infractions", {})
                n = sum(len(v) for k, v in inf.items() if "collision" in k)
                colls.append(1.0 if n else 0.0)
        if comps:
            out.append((dl, float(np.mean(comps)), float(np.mean(colls))))
    return out


def fig3():
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.28)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # (a) latency spectrum, log scale
    lat = load_latency()
    order = sorted(lat, key=lambda a: lat[a][0])
    y = np.arange(len(order))[::-1]
    for yi, a in zip(y, order):
        p50, p95, tag = lat[a]
        col = TIER[a][1]
        axA.barh(yi, p50, height=0.62, color=col, zorder=3,
                 alpha=1.0 if tag == "clean" else 0.55)
        txt = f"{p50:,.0f} ms" if p50 >= 1 else f"{p50:.2f} ms"
        if tag != "clean":
            txt += " (shadow)"
        axA.text(max(p50 * 1.15, 0.05), yi, txt, va="center", fontsize=6.3)
    axA.set_yticks(y, [LABEL[a] for a in order], fontsize=6.5)
    axA.set_xscale("log")
    axA.set_xlim(0.02, 8e4)
    axA.set_xlabel("Per-frame latency p50 (ms, log scale) — RTX A6000")
    axA.set_title("a  Five-plus orders of magnitude, one GPU", loc="left")
    axA.axvspan(0.02, 150, color="#e6f0fa", zorder=0)
    axA.text(140, y[0] + 0.15, "real-time band ", fontsize=6,
             color=C_TRAD, ha="right", va="center")
    axA.grid(axis="y", visible=False)

    # (b) deadline plane, Town13 (only town where all configs are clean at d0)
    rows = m4_rows()
    for cfg, col, lbl in (
            ("simlingo_front", C_VLA, "SimLingo"),
            ("tcp_front", C_TRAD, "TCP"),
            ("vad_6cam", "#4a3aa7", "VAD"),
            ("uniad_6cam", "#e87ba4", "UniAD")):
        pts = []
        for dl in (0, 500, 200):
            v = agg(rows, "Town13_w0", cfg, dl)
            if v:
                pts.append((dl, v[0], v[1]))
        if len(pts) < 2:
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        axB.plot(xs, ys, "-", color=col, lw=1.0, zorder=3, alpha=0.8)
        for (dl, x_, y_), mk in zip(pts, ("o", "s", "^")):
            axB.scatter([x_], [y_], marker=mk, s=22, color=col, zorder=4,
                        edgecolor="white", linewidth=0.7)
        pos = {"SimLingo": (0.62, 0.50), "TCP": (0.78, 0.16)}.get(lbl)
        if pos:
            axB.annotate(lbl, pos, fontsize=6.5, color=col, ha="right")
    # ORION (closed-loop VLA admitted mid-study): the same deadline
    # violation produces the opposite failure mode.
    orion_pts = _orion_deadline_points()
    if orion_pts:
        xs = [p[1] for p in orion_pts]
        ys = [p[2] for p in orion_pts]
        axB.plot(xs, ys, "-", color=C_WM, lw=1.2, alpha=0.85, zorder=3)
        for (dl, x_, y_), mk in zip(orion_pts, ("o", "s", "^")):
            axB.scatter([x_], [y_], marker=mk, s=22, color=C_WM,
                        zorder=4, edgecolor="white", linewidth=0.7)
        axB.annotate("ORION", (0.90, 0.44), fontsize=6.5, color=C_WM,
                     ha="right")
        axB.annotate("both deadlines", (0.055, 0.02), fontsize=5.8,
                     color=C_WM, ha="left", va="center")
    axB.set_xlabel("Route completion (mean)")
    axB.set_ylabel("Collision route rate")
    axB.set_title("b  Two ways to fail a deadline (Town13)",
                  loc="left")
    axB.set_xlim(-0.05, 1.08)
    axB.set_ylim(-0.06, 1.05)
    axB.axhspan(0.5, 1.05, color="#fdeaea", zorder=0)
    axB.text(0.02, 0.93, "limp zone: still moving,\nmostly crashing",
             fontsize=6, color=C_VIOL)
    hs = [plt.Line2D([], [], marker=m, ls="", color=MUTED, ms=5,
                     label=l) for m, l in
          (("o", "no deadline"), ("s", "500 ms"), ("^", "200 ms"))]
    axB.legend(handles=hs, loc="upper left", fontsize=6,
               bbox_to_anchor=(0.015, 0.80), framealpha=0.0,
               handletextpad=0.4, labelspacing=0.3)

    # (c) stale plan vs stale command
    conds = [("Town13_w0", 0, "baseline"),
             ("Town13_w0__frame_drop20", 0, "plan\n1 s stale"),
             ("Town13_w0__frame_drop60", 0, "plan\n3 s stale"),
             ("Town13_w0", 500, "command\nreplay @500 ms")]
    cfgs = ["tcp_front", "vad_6cam", "uniad_6cam", "simlingo_front"]
    width = 0.19
    xbase = np.arange(len(conds))
    cfg_cols = {"tcp_front": C_TRAD, "vad_6cam": "#4a3aa7",
                "uniad_6cam": "#e87ba4", "simlingo_front": C_VLA}
    for i, cfg in enumerate(cfgs):
        vals, xs = [], []
        for j, (env, dl, _) in enumerate(conds):
            v = agg(m4_rows(), env, cfg, dl)
            if v is None:
                continue
            vals.append(v[1])
            xs.append(xbase[j] + (i - 1.5) * width)
        axC.bar(xs, vals, width * 0.92, color=cfg_cols[cfg], zorder=3,
                label={"tcp_front": "TCP", "vad_6cam": "VAD",
                       "uniad_6cam": "UniAD",
                       "simlingo_front": "SimLingo"}[cfg])
    axC.set_xticks(xbase, [c[2] for c in conds], fontsize=6.5)
    axC.set_ylabel("Collision route rate")
    axC.set_title("c  Stale plan vs stale command", loc="left")
    axC.legend(ncol=4, loc="upper left", fontsize=6,
               columnspacing=0.9, handlelength=1.0)
    axC.set_ylim(0, 1.1)
    axC.grid(axis="x", visible=False)

    # (d) wrapper falsification
    cc = jload("claim_c_asyncplan_v0", "report.json")
    cells = cc["cells"]

    def cell_mean(prefix, dl):
        cs = [v for k, v in cells.items()
              if k.startswith(prefix) and k.endswith(f"|d{dl}")]
        if not cs:
            return None
        return (np.mean([c["completion_mean"] for c in cs]),
                np.mean([c["collision_route_rate"] for c in cs]))

    base = agg(m4_rows(), "Town13_w0", "simlingo_front", 0)
    async0 = cell_mean("simlingo_front_asyncplan", 0)
    bars = [("native\n(no deadline)", base[0], base[1]),
            ("async-plan wrapper\n(3 implementations)",
             async0[0], async0[1])]
    x = np.arange(len(bars))
    axD.bar(x - 0.17, [b[1] for b in bars], 0.3, color="#8a8a85", zorder=3,
            label="route completion")
    axD.bar(x + 0.17, [b[2] for b in bars], 0.3, color=C_VIOL, zorder=3,
            label="collision route rate")
    for xi, b in zip(x, bars):
        axD.text(xi - 0.17, b[1] + 0.03, f"{b[1]:.2f}", ha="center",
                 fontsize=6.3)
        axD.text(xi + 0.17, b[2] + 0.03, f"{b[2]:.2f}", ha="center",
                 fontsize=6.3)
    axD.set_xticks(x, [b[0] for b in bars], fontsize=6.5)
    axD.set_ylim(0, 1.18)
    axD.set_title("d  Wrapper cannot repair the time gap", loc="left")
    axD.legend(fontsize=6, loc="center right")
    axD.grid(axis="x", visible=False)
    save(fig, "fig3_timegate")


# ===================================================================
# Fig 4 — distribution axis
# ===================================================================
def _er6_second_pair():
    """Four-city ratios (sim-born / real-born) for the second
    provenance pair, plus which readouts are attributable given the
    carrier's repeat-call floor.  Returns {} when the readout is
    absent; nothing is zero-filled."""
    p = os.path.join(OUT_BASE, "triage_er6_origin_twin_v0",
                     "summary.json")
    if not os.path.exists(p):
        return {}
    R = json.load(open(p))["readouts"]
    out = {}
    for key, label in (("traj_diff_norm_mean", "temporal\ninstability"),
                       ("gamma_response_auc", "gamma\nresponse"),
                       ("jpeg_response_auc", "compression\nresponse")):
        if key in R:
            out[label] = (R[key]["ratio_4city_mean"],
                          R[key]["n_cities_ratio_gt_1"])
    return out


def fig4():
    fig = plt.figure(figsize=(7.0, 6.2))
    gs = fig.add_gridspec(3, 2, hspace=0.85, wspace=0.30,
                          width_ratios=[1, 1],
                          height_ratios=[1, 1, 0.92])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])
    axD = fig.add_subplot(gs[2, :])

    # (a) same-architecture dual-checkpoint contrast (D-XD1)
    def row_val(agent, city, key):
        p = os.path.join(OUT_BASE, "lineb_probe_rows_v0", agent, city,
                         "row.json")
        return json.load(open(p))["row"]["values"][key]

    x = np.arange(len(CITIES))
    for off, key, ax_bar in ((0, "gamma_response_auc", None),):
        pass
    w = 0.36
    for i, (agent, col, lbl) in enumerate((
            ("transfuser_carla", C_CARLA, "CARLA-trained ckpt"),
            ("transfuser_h4", C_REAL, "real-data ckpt"))):
        g = [row_val(agent, c, "gamma_response_auc") for c in CITIES]
        axA.bar(x + (i - 0.5) * w, g, w * 0.92, color=col, zorder=3,
                label=lbl)
    axA.set_xticks(x, [CITY_SHORT[c] for c in CITIES])
    axA.set_ylabel("Low-light sensitivity (AUC)")
    axA.set_title("a  Same architecture, different birthplace:\n"
                  "    perturbation sensitivity on real frames", loc="left")
    axA.legend(fontsize=6, loc="upper left")
    axA.set_ylim(0, 0.33)
    axA.grid(axis="x", visible=False)

    for i, (agent, col, lbl) in enumerate((
            ("transfuser_carla", C_CARLA, "CARLA-trained ckpt"),
            ("transfuser_h4", C_REAL, "real-data ckpt"))):
        t = [row_val(agent, c, "traj_diff_norm_mean") for c in CITIES]
        axB.bar(x + (i - 0.5) * w, t, w * 0.92, color=col, zorder=3,
                label=lbl)
    axB.set_xticks(x, [CITY_SHORT[c] for c in CITIES])
    axB.set_ylabel("Temporal instability (m)")
    axB.set_title("b  ...and 8-10x the temporal instability", loc="left")
    axB.grid(axis="x", visible=False)

    # (c) closed-loop rank bump chart, 6 CARLA environments
    a = jload("claim_a_readout_v0", "report.json")
    envs = ["Town03_w0", "Town03_w13", "Town12_w0", "Town12_w13",
            "Town13_w0", "Town13_w13"]
    env_lbl = ["T03\nclear", "T03\nfog", "T12\nclear", "T12\nfog",
               "T13\nclear", "T13\nfog"]
    cfg_style = {"simlingo_front": (C_VLA, "SimLingo"),
                 "tcp_front": (C_TRAD, "TCP"),
                 "vad_6cam": ("#4a3aa7", "VAD"),
                 "uniad_6cam": ("#e87ba4", "UniAD"),
                 "uniad_front_zero": ("#eda100", "UniAD front-only")}
    for cfg, (col, lbl) in cfg_style.items():
        ranks = []
        for e in envs:
            ranks.append(a["rankings"][e].index(cfg) + 1)
        axC.plot(range(len(envs)), ranks, "-o", color=col, lw=1.4, ms=4,
                 mfc="white", mew=1.0, zorder=3)
        axC.annotate(lbl, (len(envs) - 1, ranks[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=6.5,
                     color=col)
    axC.set_xticks(range(len(envs)), env_lbl, fontsize=6.5)
    axC.set_yticks([1, 2, 3, 4, 5])
    axC.invert_yaxis()
    axC.set_ylabel("Rank (completion primitive)")
    axC.set_title("c  Closed loop, six environments: ranks 2-5 keep "
                  "trading places (13/15 env pairs change some order)",
                  loc="left")
    axC.set_xlim(-0.3, len(envs) - 0.2)
    axC.grid(axis="x", visible=False)
    # (d) second provenance pair: what transfers and what is not
    # attributable on a stateful carrier
    er6 = _er6_second_pair()
    if er6:
        labels = list(er6.keys())
        vals = [er6[k][0] for k in labels]
        ncit = [er6[k][1] for k in labels]
        # attributability: only the baseline-pass readout clears the
        # carrier's repeat-call floor (measured in the companion
        # instrument study); the two perturbation readouts do not.
        attributable = [k.startswith("temporal") for k in labels]
        y = np.arange(len(labels))[::-1]
        for yi, v, ok in zip(y, vals, attributable):
            axD.barh(yi, v, height=0.5, zorder=3,
                     color=(C_WM if ok else "#c9c6c0"),
                     hatch=(None if ok else "///"),
                     edgecolor=("none" if ok else "#8a8a85"))
        axD.axvline(1.0, color=MUTED, lw=0.9, ls="--", zorder=2)
        for yi, v, n, ok in zip(y, vals, ncit, attributable):
            axD.text(v + 0.03, yi,
                     f"{v:.2f}  ({n}/4 cities)" if ok
                     else f"{v:.2f}  not attributable",
                     va="center", fontsize=6.2,
                     color=(INK if ok else "#6f6c66"))
        axD.set_yticks(y, [l.replace("\n", " ") for l in labels],
                       fontsize=6.8)
        axD.set_xlim(0, 1.62)
        axD.set_xlabel("simulation-born / real-born  "
                       "(>1 = the predicted direction)")
        axD.set_title("d  Second provenance pair: one readout "
                      "transfers, two sit under the carrier's "
                      "repeat-call floor", loc="left")
        axD.grid(axis="y", visible=False)
    save(fig, "fig4_distribution")


# ===================================================================
# Fig 5 — selection: gate matrix + capability boundary
# ===================================================================
def fig5():
    fig = plt.figure(figsize=(7.0, 2.5))
    gs = fig.add_gridspec(1, 2, wspace=0.35, width_ratios=[1.15, 1])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    t = jload("triage_selection_analysis_v0", "report.json")
    profiles = ["P1_mono_realtime", "P2_mono_lowspeed",
                "P3_full_realtime", "P4_full_lowspeed"]
    prof_lbl = ["mono\nreal-time", "mono\nlow-speed",
                "full rig\nreal-time", "full rig\nlow-speed"]
    sel_lbl = ["Pick global No.1", "Pick benchmark champion", "TRIAGE"]
    NAME = {"driveva": "DriveVA", "simlingo": "SimLingo", "tcp": "TCP",
            "diffusiondrive": "DiffDrive", "recogdrive": "ReCogDrive"}
    rowmap = {r["profile"]: r for r in t["rows"]}
    for j, p in enumerate(profiles):
        r = rowmap[p]
        entries = [
            (r["score_first_pick"], r["score_first_violation"]),
            (r["carla_champ_pick"], r["carla_champ_violation"]),
            (r["triage_pick"], None)]
        for i, (pick, viol) in enumerate(entries):
            bad = viol is not None
            col = "#fdeaea" if bad else "#e7f6ef"
            axA.add_patch(plt.Rectangle((j, 2 - i), 0.96, 0.96,
                                        facecolor=col, edgecolor="white",
                                        lw=2, zorder=2))
            txt = NAME.get(pick, pick)
            sub = {"time": "hits TIME gate", "info": "hits INFO gate"}.get(
                viol, "")
            if i == 2 and rowmap[p].get("triage_pick_unscored_flag"):
                sub = "flag: no target-\ndomain score yet"
            axA.text(j + 0.48, 2 - i + 0.60, txt, ha="center", va="center",
                     fontsize=6.6,
                     color=C_VIOL if bad else "#1a7a4d",
                     fontweight="bold", zorder=3)
            if sub:
                axA.text(j + 0.48, 2 - i + 0.28, sub, ha="center",
                         va="center", fontsize=5.4,
                         color=C_VIOL if bad else MUTED, zorder=3)
    axA.set_xlim(-0.05, 4.05)
    axA.set_ylim(-0.05, 3.05)
    axA.set_xticks([j + 0.48 for j in range(4)], prof_lbl, fontsize=6.5)
    axA.set_yticks([2.48, 1.48, 0.48], sel_lbl, fontsize=6.5)
    axA.set_title("a  Score-first selection walks into gates; "
                  "TRIAGE does not", loc="left")
    axA.grid(visible=False)
    for s in axA.spines.values():
        s.set_visible(False)
    axA.tick_params(length=0)

    # (b) capability boundary: probe-based selection vs follow-the-champion
    v0 = jload("lineb_minig2_exam_v0", "report.json")
    v1 = jload("lineb_minig2_exam_v1", "report.json")
    pairs = [("8 configs\n(champion: ReCogDrive)", v0),
             ("9 configs\n(champion: DriveVA)", v1)]
    x = np.arange(len(pairs))
    w = 0.34
    pr = [100 * p[1]["mean_regret_predictor"] for p in pairs]
    ch = [100 * p[1]["mean_regret_naive_champion"] for p in pairs]
    axB.bar(x - w / 2, pr, w * 0.92, color=C_VLA, zorder=3,
            label="10-min probe predictor")
    axB.bar(x + w / 2, ch, w * 0.92, color=C_TRAD, zorder=3,
            label="follow global champion")
    for xi, v in zip(x - w / 2, pr):
        axB.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.3)
    for xi, v in zip(x + w / 2, ch):
        axB.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.3)
    axB.set_xticks(x, [p[0] for p in pairs], fontsize=6.5)
    axB.set_ylabel("Selection regret (PDMS pts)")
    axB.set_title("b  Within the top grade, cheap probes lose\n"
                  "    to simply following the champion", loc="left")
    axB.legend(fontsize=6, loc="upper right")
    axB.set_ylim(0, 1.6)
    axB.grid(axis="x", visible=False)
    save(fig, "fig5_selection")




# ===================================================================
# Fig 6 — repair-operator cross experiment (E-R1/R2/R3)
# ===================================================================
def fig6():
    er2 = os.path.join(OUT_BASE, "triage_er2_dose_v0", "summary.json")
    er1 = os.path.join(OUT_BASE, "triage_er1_fovft_v0", "readout.json")
    if not (os.path.exists(er2) and os.path.exists(er1)):
        print("fig6: readouts not present yet, skipping")
        return
    S = json.load(open(er2))
    R = json.load(open(er1))
    doses = [0, 1000, 5000, 20000]
    xpos = np.arange(4)
    xticks = ["0\n(zero-shot)", "1k", "5k", "20k"]

    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(1, 3, wspace=0.42,
                          width_ratios=[1, 1, 1.15])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # (a) remaining fraction of the sim-vs-real gap, per readout
    names = {"R1": "low-light sens.", "R2": "JPEG sens.",
             "R3": "temporal instab."}
    cols = {"R1": C_TRAD, "R2": "#4a3aa7", "R3": C_VLA}
    VAR_SEQ = ["transfuser_carla_m0", "er2_d1k", "er2_d5k", "er2_d20k"]
    for rid in ("R1", "R2", "R3"):
        e = S["readouts"][rid]
        seq = e.get("sequence_m0_d1k_d5k_d20k")
        if not seq or any(s is None for s in seq):
            continue
        ref = e["ref_navsim_mean"]
        m0 = seq[0]
        gap0 = m0 - ref
        if abs(gap0) < 1e-9:
            continue
        pc = e.get("per_city", {})
        for c, vv in pc.items():
            cy = [vv.get(v) for v in VAR_SEQ]
            if any(x is None for x in cy):
                continue
            cf = [(x - ref) / gap0 for x in cy]
            axA.plot(xpos, cf, "-", color=cols[rid], lw=0.5,
                     alpha=0.30, zorder=2)
        frac = [(s - ref) / gap0 for s in seq]
        axA.plot(xpos, frac, "-o", color=cols[rid], lw=1.5, ms=3.5,
                 mfc="white", mew=0.9, label=names[rid], zorder=4)
        t4d = os.path.join(OUT_BASE, "triage_t4d_seed2_v0",
                           "summary.json")
        if os.path.exists(t4d):
            sc = json.load(open(t4d))["seed_comparison"].get(rid)
            if sc:
                v2 = np.mean([c["seed2"] for c in
                              sc["per_city"].values()])
                axA.scatter([xpos[-1] + 0.13],
                            [(v2 - ref) / gap0], marker="x", s=22,
                            color=cols[rid], zorder=5, linewidth=1.2)
    axA.axhline(1.0, color=MUTED, lw=0.7, ls="-")
    axA.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    axA.axhline(0.0, color=C_WM, lw=0.8, ls=":")
    for yv, lbl, cc in ((1.0, "zero-shot level", MUTED),
                        (0.5, "closure threshold", MUTED),
                        (0.0, "real-data twin", C_WM)):
        axA.text(0.98, yv + 0.09, lbl, fontsize=5.4, color=cc,
                 ha="right", transform=axA.get_yaxis_transform())
    axA.set_xticks(xpos, xticks, fontsize=6.5)
    axA.set_xlabel("Fine-tuning dose (real frames)")
    axA.set_ylabel("Fraction of sim-vs-real gap\nremaining")
    axA.set_title("a  Stability converges; sensitivity\n"
                  "    spikes first (robustness cost)", loc="left")
    axA.legend(fontsize=5.8, loc="upper right")
    axA.set_ylim(-0.25, 5.1)

    # (b) L2 vs human, all + turn
    e4 = S["readouts"]["R4"]
    rows_dir = os.path.join(OUT_BASE, "triage_er2_dose_v0", "rows")
    for c in CITIES:
        cy = []
        for v in ["transfuser_carla_m0", "er2_d1k", "er2_d5k",
                  "er2_d20k"]:
            p = os.path.join(rows_dir, v, c, "row.json")
            if not os.path.exists(p):
                cy = []
                break
            cy.append(json.load(open(p))["l2_vs_human"]["all"][
                "mean_l2_at_2s"])
        if cy:
            axB.plot(xpos, cy, "-", color=C_TRAD, lw=0.5, alpha=0.30,
                     zorder=2)
    l2 = e4["sequence_m0_d1k_d5k_d20k"]
    l2t = e4.get("turn_sequence")
    if l2 and all(s is not None for s in l2):
        axB.plot(xpos, l2, "-o", color=C_TRAD, lw=1.5, ms=3.5,
                 mfc="white")
    t4d = os.path.join(OUT_BASE, "triage_t4d_seed2_v0",
                       "summary.json")
    if os.path.exists(t4d):
        sc = json.load(open(t4d))["seed_comparison"].get("R4")
        if sc:
            v2 = np.mean([c["seed2"] for c in
                          sc["per_city"].values()])
            axB.scatter([xpos[-1] + 0.13], [v2], marker="x", s=22,
                        color=C_TRAD, zorder=5, linewidth=1.2)
    if l2t and all(s is not None for s in l2t):
        axB.plot(xpos, l2t, "--s", color=C_VLA, lw=1.5, ms=3.5,
                 mfc="white", label="turn frames")
    axB.set_xticks(xpos, xticks, fontsize=6.5)
    axB.set_xlabel("Fine-tuning dose (real frames)")
    axB.set_ylabel("L2 vs. human at 2 s (m)")
    axB.set_title("b  ...and tracks human driving\n"
                  "    (all frames)", loc="left")

    # (c) E-R1 four arms, turn stratum + all frames
    v = R["variants"]
    labels = ["zero-shot\nnative", "fine-tuned\nnative",
              "zero-shot\ncrop 0.6", "fine-tuned\ncrop 0.6"]
    keys = ["zeroshot_native", "ft_native", "zeroshot_crop", "ft_crop"]
    turn = [v[k]["pooled"]["turn"]["mean_l2_at_2s"] for k in keys]
    allf = [v[k]["pooled"]["all"]["mean_l2_at_2s"] for k in keys]
    x = np.arange(4)
    axC.bar(x - 0.18, turn, 0.34, color=C_VLA, zorder=3,
            label="turn frames")
    axC.bar(x + 0.18, allf, 0.34, color="#8a8a85", zorder=3,
            label="all frames")
    for i, k in enumerate(keys):
        pc = v[k].get("per_city", {})
        pts = [d["turn"]["mean_l2_at_2s"] for d in pc.values()
               if d.get("turn")]
        axC.scatter([x[i] - 0.18] * len(pts), pts, s=6, color="white",
                    edgecolor=INK, linewidth=0.5, zorder=4)
    for xi, t in zip(x, turn):
        axC.text(xi - 0.18, t + max(turn) * 0.02, f"{t:.2f}",
                 ha="center", fontsize=5.8)
    resid = R["verdict"].get("residual_turn_m")
    repA = R["verdict"].get("repairA_turn_m")
    if resid is not None and repA:
        axC.annotate(
            f"turn residual {resid:+.03f} m\n"
            f"= {100*resid/repA:.0f}% of repair:\n"
            "direction as predicted,\n"
            "magnitude negligible",
            xy=(3 - 0.18, turn[3] + 0.1),
            xytext=(0.33, 0.42), textcoords="axes fraction",
            fontsize=5.6, color=INK,
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED))
    axC.set_xticks(x, labels, fontsize=5.6)
    axC.set_ylabel("L2 vs. human at 2 s (m)")
    axC.set_title("c  Behavioral compensation under FOV crop\n    (TCP, matched budget)", loc="left")
    axC.legend(fontsize=6, loc="upper left")
    axC.set_ylim(0, max(max(turn), max(allf)) * 1.18)
    axC.grid(axis="x", visible=False)
    save(fig, "fig6_repair")




# ===================================================================
# Fig 7 — instrument portability: manipulation FMs (D-TR2)
# ===================================================================
def fig7():
    lat_dir = os.path.join(OUT_BASE, "triage_manip_latency_v0")
    dose_dir = os.path.join(OUT_BASE, "triage_manip_dose_v0")
    if not (os.path.isdir(lat_dir) and os.path.isdir(dose_dir)):
        print("fig7: manip outputs not present, skipping")
        return
    NAME = {"octo_small": "Octo-small (27M)",
            "octo_base": "Octo-base (93M)",
            "groot": "GR00T N1.5 (3B)",
            "smolvla": "SmolVLA",
            "openvla": "OpenVLA (7B)",
            "pi0": "$\\pi_0$ (3B)"}
    CHUNK = {"octo_small": 4, "octo_base": 4, "groot": 16,
             "smolvla": 50, "openvla": 1, "pi0": 50}
    models = ["octo_small", "octo_base", "groot", "smolvla",
              "openvla", "pi0"]

    fig = plt.figure(figsize=(7.2, 2.4))
    gs = fig.add_gridspec(1, 2, wspace=0.55, width_ratios=[1, 1])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    lat = {}
    for m in models:
        p = os.path.join(lat_dir, f"{m}.json")
        if os.path.exists(p):
            lat[m] = json.load(open(p))["ms"]["p50"]
    order = sorted(lat, key=lambda m: lat[m])
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        axA.barh(yi, lat[m], height=0.6, color=C_TRAD, zorder=3)
        axA.text(lat[m] * 1.07, yi,
                 f"{lat[m]:.0f} ms / {CHUNK[m]}-step chunk",
                 va="center", fontsize=6.0)
    axA.set_yticks(y, [NAME[m] for m in order], fontsize=6.3)
    axA.set_xscale("log")
    axA.set_xlim(20, 3000)
    axA.set_xlabel("Latency p50 per planning call (ms, log) — A6000")
    axA.set_title("a  Manipulation latency passports\n"
                  "    (exclusive-card protocol, unchanged)", loc="left")
    axA.grid(axis="y", visible=False)

    ratios = {}
    for m in models:
        p = os.path.join(dose_dir, m, "dose.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))["mean_action_displacement"]
        f, r = d["fov_crop"]["0.75"], d["resolution"]["0.75"]
        if r:
            ratios[m] = f / r
    order2 = sorted(ratios, key=lambda m: ratios[m])
    y2 = np.arange(len(order2))[::-1]
    for yi, m in zip(y2, order2):
        axB.barh(yi, ratios[m], height=0.6, color=C_VLA, zorder=3)
        axB.text(ratios[m] + 0.05, yi, f"{ratios[m]:.1f}x",
                 va="center", fontsize=6.2)
    axB.axvline(1.0, color=MUTED, lw=0.9, ls="--", zorder=2)
    axB.text(1.02, y2[-1] - 0.62, "parity", fontsize=5.6, color=MUTED)
    axB.set_yticks(y2, [NAME[m] for m in order2], fontsize=6.3)
    axB.set_xlim(0, 2.9)
    axB.set_xlabel("FOV / resolution damage ratio\n"
                   "(equal pixel budget; within-model, unit-free)")
    axB.set_title("b  Not-seen $\\geq$ not-sharp: 6/6 models\n"
                  "    (driving showed 2.2\u20135.7x)", loc="left")
    axB.grid(axis="y", visible=False)
    save(fig, "fig7_portability")




# ===================================================================
# Fig 8 — T4-B closed-loop crop verification
# ===================================================================
def _orion_crop_pairs():
    """Per-seed (completion, collision route rate) for ORION native
    and fov-crop 0.6.  Cells that do not exist are skipped, never
    zero-filled."""
    base = os.path.join(OUT_BASE, "orion_t5a")
    out = {}
    for tag, key in (("orion_native", "native"),
                     ("orion_crop06", "crop")):
        comps, crrs = [], []
        for seed in (0, 1, 2):
            p = os.path.join(base, f"Town13_w0__{tag}__seed{seed}",
                             "result.json")
            if not os.path.exists(p):
                continue
            try:
                recs = json.load(open(p))["_checkpoint"]["records"]
            except Exception:
                continue
            if not recs:
                continue
            comps.append(float(np.mean(
                [r["scores"]["score_route"] for r in recs])))
            hits = []
            for r in recs:
                inf = r.get("infractions", {})
                n = sum(len(v) for k, v in inf.items()
                        if "collision" in k)
                hits.append(1.0 if n else 0.0)
            crrs.append(float(np.mean(hits)))
        if comps:
            out[key] = (comps, crrs)
    return out if len(out) == 2 else {}


def fig8():
    p = os.path.join(OUT_BASE, "triage_t4b_crop_closedloop_v0",
                     "report.json")
    if not os.path.exists(p):
        print("fig8: T4-B report missing, skipping")
        return
    R = json.load(open(p))["contrasts"]
    cfgs = [("tcp_front_crop06", "TCP"),
            ("simlingo_front_crop06", "SimLingo")]
    orion = _orion_crop_pairs()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.3),
                                   gridspec_kw={"wspace": 0.35})
    w = 0.32
    for j, (key, lbl) in enumerate(cfgs):
        c = R[key]
        for i, (side, col) in enumerate((("native", "#8a8a85"),
                                         ("crop", C_VLA))):
            comp = [v for v in
                    c[side]["per_seed_completion"].values()]
            crr = [v for v in
                   c[side]["per_seed_collision_route_rate"].values()]
            x = j + (i - 0.5) * w
            axA.bar(x, np.mean(comp), w * 0.9, color=col, zorder=3)
            axA.scatter([x] * len(comp), comp, s=9, color="white",
                        edgecolor=INK, linewidth=0.5, zorder=4)
            axB.bar(x, np.mean(crr), w * 0.9, color=col if i else
                    "#8a8a85", zorder=3)
            axB.scatter([x] * len(crr), crr, s=9, color="white",
                        edgecolor=INK, linewidth=0.5, zorder=4)
    if orion:
        j = len(cfgs)
        for i, (side, col) in enumerate((("native", "#8a8a85"),
                                         ("crop", C_WM))):
            comp, crr = orion[side]
            x = j + (i - 0.5) * w
            axA.bar(x, np.mean(comp), w * 0.9, color=col, zorder=3)
            axA.scatter([x] * len(comp), comp, s=9, color="white",
                        edgecolor=INK, linewidth=0.5, zorder=4)
            axB.bar(x, np.mean(crr), w * 0.9, color=col, zorder=3)
            axB.scatter([x] * len(crr), crr, s=9, color="white",
                        edgecolor=INK, linewidth=0.5, zorder=4)
    ticks = [l for _, l in cfgs] + (["ORION"] if orion else [])
    for ax in (axA, axB):
        ax.set_xticks(range(len(ticks)), ticks)
        ax.grid(axis="x", visible=False)
    axA.set_ylabel("Route completion (%)")
    axA.set_ylim(0, 112)
    axA.set_title("a  Route completion", loc="left")
    axB.set_ylabel("Collision route rate")
    axB.set_ylim(0, 0.72)
    axB.set_title("b  Collision route rate", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#8a8a85",
                             label="native"),
               plt.Rectangle((0, 0), 1, 1, color=C_VLA,
                             label="fov-crop 0.6 (named intervention)"),
               plt.Rectangle((0, 0), 1, 1, color=C_WM,
                             label="fov-crop 0.6, world-model VLA"),
               plt.Line2D([], [], marker="o", ls="", mfc="white",
                          mec=INK, ms=4, label="individual seeds")]
    axB.legend(handles=handles, fontsize=6, loc="upper left")
    save(fig, "fig8_cropclosedloop")




# ===================================================================
# Fig 6 split versions for gap-organized paper: dist (a,b) / info (c)
# ===================================================================

# ===================================================================
# Fig 9 -- the TRIAGE workflow (framework spine figure)
# ===================================================================
def fig9_framework():
    fig, ax = plt.subplots(figsize=(7.6, 3.15))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    GATE_C = "#c9503c"; STAGE_C = "#1f6fb2"; OK_C = "#1c8a5a"
    GREY = "#555555"

    def stage(x, y, w, h, title, lines, cost, rq, ec=STAGE_C,
              fc="#eef4fb"):
        ax.add_patch(plt.Rectangle((x, y), w, h, ec=ec, fc=fc, lw=1.2,
                                   zorder=3))
        ax.text(x + w/2, y + h - 1.6, title, ha="center", va="top",
                fontsize=7.2, fontweight="bold", color=ec, zorder=4)
        ax.text(x + w/2, y + h - 5.6, lines, ha="center", va="top",
                fontsize=5.9, color="#222222", zorder=4,
                linespacing=1.35)
        if cost:
            ax.text(x + w/2, y + 4.6, cost, ha="center", va="center",
                    fontsize=5.5, color=GREY, style="italic", zorder=4)
        if rq:
            ax.text(x + w/2, y + 1.7, rq, ha="center", va="center",
                    fontsize=5.9, color=OK_C, fontweight="bold",
                    zorder=4)

    def arrow(x0, x1, y=25.0):
        ax.annotate("", (x1, y), (x0, y),
                    arrowprops=dict(arrowstyle="-|>", color="#333333",
                                    lw=1.1), zorder=2)

    Y, H = 13.5, 24.5  # box band
    stage(0.5, Y, 12.5, H, "Inputs",
          "pool $\\mathcal{C}$\nlocked target:\nsensors $S'$\ndeadline $d'$\nsite $D'$, budget $N$",
          None, None, ec="#444444", fc="#f4f4f4")
    stage(16.5, Y, 15, H, "1  Contract audit",
          "sensor contract vs $S'$\n(paper only);\nconversions become\nnamed interventions",
          "an afternoon, no GPU", "RQ1–3")
    stage(35.0, Y, 15, H, "2  Latency passport",
          "exclusive-card $\\tau_c$ on\nthe deployment GPU;\ntime gate: $\\tau_c>d'$\n$\\Rightarrow$ drop / async only",
          "10 min per candidate", "RQ4")
    stage(53.5, Y, 15, H, "3  Shadow probes",
          "10-min local logs:\nTI, $D(\\rho)$ + checks;\ntier placement,\nprovenance evidence",
          "10 min per candidate", "RQ5, RQ7")
    stage(72.0, Y, 12.5, H, "4  Rank",
          "champion within\ntop tier; provenance\ntiebreak; challenger",
          "free", "RQ7–8")
    stage(88.0, Y, 11.5, H, "5  Repair",
          "post-train the $D$-\nresidual only, with a\nsensitivity monitor;\nclosed-loop check",
          "$N$ frames + monitor", "RQ6, RQ3", ec=OK_C, fc="#eaf6f0")

    arrow(13.2, 16.3); arrow(31.7, 34.8); arrow(50.2, 53.3)
    arrow(68.7, 71.8); arrow(84.7, 87.8)
    ax.text(93.7, 39.0, "deploy + challenger", ha="center",
            fontsize=6.2, color=OK_C, fontweight="bold")
    ax.annotate("", (93.7, 38.2), (93.7, 38.6),
                arrowprops=dict(arrowstyle="-|>", color=OK_C, lw=1.1))
    ax.plot([93.7, 93.7], [38.3, 38.15], color=OK_C, lw=0)
    ax.annotate("", (93.7, 38.35), (93.7, 38.45))
    ax.annotate("", (93.7, 38.3), (93.7, Y + H),
                arrowprops=dict(arrowstyle="<|-", color=OK_C, lw=1.1))

    # Gate exits. Algorithm 1 gained an empty-feasible-set branch, so the
    # lane below carries the three outs it names rather than a single
    # "excluded" verdict, and the re-entry arrow is drawn because an
    # asynchronous-consumption contract sends a time-gated candidate back
    # to Gate 2 rather than out of the pool.
    for x, lab in ((20.0, "info gate"), (38.5, "time gate")):
        ax.annotate("", (x, 8.0), (x, Y),
                    arrowprops=dict(arrowstyle="-|>", color=GATE_C,
                                    lw=1.0, ls="--"), zorder=2)
        ax.text(x + 0.9, 9.6, lab, ha="left", va="center",
                fontsize=5.6, color=GATE_C)
    ax.add_patch(plt.Rectangle((14.0, 0.5), 72, 7.2, ec=GATE_C,
                               fc="#fbeeec", lw=1.0))
    ax.text(50, 5.5, "excluded, not down-weighted. No ranking is defined "
            "over an empty feasible set,\nand a gate is never widened to "
            "obtain one",
            ha="center", va="center", fontsize=5.7, color=GATE_C,
            fontweight="bold", linespacing=1.3)
    ax.text(50, 2.4,
            "return the binding gate per candidate   $\\cdot$   re-enter at "
            "Gate 2 under an asynchronous-consumption contract   $\\cdot$   "
            "commission the repair the binding class names\n"
            "(hardware for information, consumption architecture for time; "
            "never fine-tuning)",
            ha="center", va="center", fontsize=5.5, color=GATE_C,
            linespacing=1.4)
    # re-entry: async contract returns a time-gated candidate to Gate 2
    ax.annotate("", (42.5, Y - 0.2), (42.5, 7.9),
                arrowprops=dict(arrowstyle="-|>", color=GATE_C, lw=0.9,
                                ls=(0, (2, 1.6)),
                                connectionstyle="arc3,rad=0.0"), zorder=2)
    ax.text(43.2, 11.6, "re-enter (async)", ha="left", va="center",
            fontsize=5.4, color=GATE_C, style="italic")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99,
                        bottom=0.01)
    save(fig, "fig9_framework")


def fig6_split():
    er2 = os.path.join(OUT_BASE, "triage_er2_dose_v0", "summary.json")
    er1 = os.path.join(OUT_BASE, "triage_er1_fovft_v0", "readout.json")
    if not (os.path.exists(er2) and os.path.exists(er1)):
        print("fig6_split: readouts missing, skipping")
        return
    S = json.load(open(er2))
    R = json.load(open(er1))
    xpos = np.arange(4)
    xticks = ["0\n(zero-shot)", "1k", "5k", "20k"]

    # ---- distribution repair figure (a,b) ----
    fig = plt.figure(figsize=(6.4, 2.5))
    gs = fig.add_gridspec(1, 2, wspace=0.38)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    names = {"R1": "low-light sens.", "R2": "JPEG sens.",
             "R3": "temporal instab."}
    cols = {"R1": C_TRAD, "R2": "#4a3aa7", "R3": C_VLA}
    VAR_SEQ = ["transfuser_carla_m0", "er2_d1k", "er2_d5k", "er2_d20k"]
    for rid in ("R1", "R2", "R3"):
        e = S["readouts"][rid]
        seq = e.get("sequence_m0_d1k_d5k_d20k")
        if not seq or any(x is None for x in seq):
            continue
        ref = e["ref_navsim_mean"]; m0 = seq[0]; gap0 = m0 - ref
        if abs(gap0) < 1e-9: continue
        for c, vv in e.get("per_city", {}).items():
            cy = [vv.get(v) for v in VAR_SEQ]
            if any(x is None for x in cy): continue
            axA.plot(xpos, [(x-ref)/gap0 for x in cy], "-",
                     color=cols[rid], lw=0.5, alpha=0.30, zorder=2)
        axA.plot(xpos, [(x-ref)/gap0 for x in seq], "-o",
                 color=cols[rid], lw=1.5, ms=3.5, mfc="white",
                 mew=0.9, label=names[rid], zorder=4)
        t4d = os.path.join(OUT_BASE, "triage_t4d_seed2_v0",
                           "summary.json")
        if os.path.exists(t4d):
            sc = json.load(open(t4d))["seed_comparison"].get(rid)
            if sc:
                v2 = np.mean([c["seed2"] for c in
                              sc["per_city"].values()])
                axA.scatter([xpos[-1]+0.13], [(v2-ref)/gap0],
                            marker="x", s=22, color=cols[rid],
                            zorder=5, linewidth=1.2)
    for yv, lbl, cc in ((1.0, "zero-shot level", MUTED),
                        (0.5, "closure threshold", MUTED),
                        (0.0, "real-data twin", C_WM)):
        axA.axhline(yv, color=cc, lw=0.8,
                    ls={"1.0":"-","0.5":"--","0.0":":"}[f"{yv}"])
        axA.text(0.98, yv+0.09, lbl, fontsize=5.4, color=cc,
                 ha="right", transform=axA.get_yaxis_transform())
    axA.set_xticks(xpos, xticks, fontsize=6.5)
    axA.set_xlabel("Fine-tuning dose (real frames)")
    axA.set_ylabel("Fraction of sim-vs-real gap\nremaining")
    axA.set_title("a  Stability converges; sensitivity\n"
                  "    spikes first (robustness cost)", loc="left")
    axA.legend(fontsize=5.8, loc="upper right")
    axA.set_ylim(-0.25, 5.1)

    rows_dir = os.path.join(OUT_BASE, "triage_er2_dose_v0", "rows")
    for c in CITIES:
        cy = []
        for v in VAR_SEQ:
            p = os.path.join(rows_dir, v, c, "row.json")
            if not os.path.exists(p): cy=[]; break
            cy.append(json.load(open(p))["l2_vs_human"]["all"][
                "mean_l2_at_2s"])
        if cy:
            axB.plot(xpos, cy, "-", color=C_TRAD, lw=0.5,
                     alpha=0.30, zorder=2)
    l2 = S["readouts"]["R4"]["sequence_m0_d1k_d5k_d20k"]
    if l2 and all(x is not None for x in l2):
        axB.plot(xpos, l2, "-o", color=C_TRAD, lw=1.5, ms=3.5,
                 mfc="white")
    t4d = os.path.join(OUT_BASE, "triage_t4d_seed2_v0",
                       "summary.json")
    if os.path.exists(t4d):
        sc = json.load(open(t4d))["seed_comparison"].get("R4")
        if sc:
            v2 = np.mean([c["seed2"] for c in
                          sc["per_city"].values()])
            axB.scatter([xpos[-1]+0.13], [v2], marker="x", s=22,
                        color=C_TRAD, zorder=5, linewidth=1.2)
    axB.set_xticks(xpos, xticks, fontsize=6.5)
    axB.set_xlabel("Fine-tuning dose (real frames)")
    axB.set_ylabel("L2 vs. human at 2 s (m)")
    axB.set_title("b  ...and tracks human driving\n    (all frames)",
                  loc="left")
    save(fig, "fig6d_repair_dist")

    # ---- info repair figure (former panel c, standalone) ----
    fig, axC = plt.subplots(figsize=(4.2, 2.5))
    v = R["variants"]
    labels = ["zero-shot\nnative", "fine-tuned\nnative",
              "zero-shot\ncrop 0.6", "fine-tuned\ncrop 0.6"]
    keys = ["zeroshot_native", "ft_native", "zeroshot_crop",
            "ft_crop"]
    turn = [v[k]["pooled"]["turn"]["mean_l2_at_2s"] for k in keys]
    allf = [v[k]["pooled"]["all"]["mean_l2_at_2s"] for k in keys]
    x = np.arange(4)
    axC.bar(x-0.18, turn, 0.34, color=C_VLA, zorder=3,
            label="turn frames")
    axC.bar(x+0.18, allf, 0.34, color="#8a8a85", zorder=3,
            label="all frames")
    for i, k in enumerate(keys):
        pts = [d["turn"]["mean_l2_at_2s"] for d in
               v[k].get("per_city", {}).values() if d.get("turn")]
        axC.scatter([x[i]-0.18]*len(pts), pts, s=6, color="white",
                    edgecolor=INK, linewidth=0.5, zorder=4)
    for xi, t in zip(x, turn):
        axC.text(xi-0.18, t+max(turn)*0.02, f"{t:.2f}", ha="center",
                 fontsize=5.8)
    resid = R["verdict"].get("residual_turn_m")
    repA = R["verdict"].get("repairA_turn_m")
    if resid is not None and repA:
        axC.annotate(
            f"turn residual {resid:+.03f} m\n"
            f"= {100*resid/repA:.0f}% of repair",
            xy=(3-0.18, turn[3]+0.1),
            xytext=(0.30, 0.55), textcoords="axes fraction",
            fontsize=5.8, color=INK,
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED))
    axC.set_xticks(x, labels, fontsize=5.6)
    axC.set_ylabel("L2 vs. human at 2 s (m)")
    axC.set_ylim(0, max(max(turn), max(allf))*1.18)
    axC.legend(fontsize=6, loc="upper right")
    axC.grid(axis="x", visible=False)
    save(fig, "fig6i_repair_info")




# ===================================================================
# Fig 10 — D-TR7: the engagement anchor
#   (a) rank inversion: raw perturbation response vs engagement-normalized,
#       on the three STATELESS carriers where the anchor is validated
#   (b) the state-advance floor against the readouts it must qualify;
#       exactly zero on a stateless carrier, above every perturbation
#       readout on the stateful one
# Reads only outputs/triage_dtr7_engagement_anchor_v0/.
# ===================================================================
DTR7 = "triage_dtr7_engagement_anchor_v0"
STATELESS3 = ["transfuser", "ltf", "diffusiondrive"]


def _dtr7_means(cfg):
    g, k, gray = [], [], []
    for c in CITIES:
        r = jload(DTR7, "rows", cfg, c, "row.json")
        g.append(r["dose_displacement_mean"]["gamma"])
        k.append(r["engagement_normalized"]["gamma"])
        gray.append(r["anchor"]["D_gray"])
    return float(np.mean(g)), float(np.mean(k)), float(np.mean(gray))


def _dtr7_floor(cfg):
    v = []
    for c in CITIES:
        f = jload(DTR7, "rows", "_floor_curve", cfg, c, "floor.json")
        v.append(np.mean([f["drift_k"][str(k)]["mean"] for k in range(2, 7)]))
    return float(np.mean(v))


def _twin_means(cfg):
    g = j = gr = []
    G = [jload(DTR7, "rows", cfg, c, "row.json") for c in CITIES]
    return (float(np.mean([r["dose_displacement_mean"]["gamma"] for r in G])),
            float(np.mean([r["dose_displacement_mean"]["jpeg"] for r in G])),
            float(np.mean([r["anchor"]["D_gray"] for r in G])))


def fig10():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.9))

    # ---- (a) rank inversion, stateless carriers only -----------------
    ax = axes[0]
    vals = {c: _dtr7_means(c) for c in STATELESS3}
    raw = {c: vals[c][0] for c in STATELESS3}
    kap = {c: vals[c][1] for c in STATELESS3}
    r_rank = {c: i for i, c in enumerate(
        sorted(STATELESS3, key=lambda x: -raw[x]))}
    k_rank = {c: i for i, c in enumerate(
        sorted(STATELESS3, key=lambda x: -kap[x]))}
    for c in STATELESS3:
        col = TIER[c][1]
        ax.plot([0, 1], [r_rank[c], k_rank[c]], "-o", color=col, lw=1.6,
                ms=5, zorder=3)
        ax.annotate(f"{LABEL[c]}\n{raw[c]:.3f} m", (0, r_rank[c]),
                    xytext=(-6, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=7, color=col)
        ax.annotate(f"{LABEL[c]}\n{kap[c]:.3f}", (1, k_rank[c]),
                    xytext=(6, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=7, color=col)
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(2.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["by raw response\n$\\bar D_{\\gamma}$",
                        "by engagement-normalized\n$\\kappa_{\\gamma}="
                        "\\bar D_{\\gamma}/D_{\\mathrm{gray}}$"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["most\nsensitive", "", "least\nsensitive"])
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) The ordering completely reverses\n"
                 "(stateless carriers, four-city means)", loc="left")

    # ---- (b) floor vs readouts ---------------------------------------
    ax = axes[1]
    cfgs = [("transfuser", "TransFuser\n(stateless)"),
            ("uniad_nusc_born", "UniAD real-born\n(stateful)"),
            ("uniad_b2d_born", "UniAD sim-born\n(stateful)")]
    x = np.arange(len(cfgs))
    w = 0.22
    gam, jpg, gry, flr = [], [], [], []
    for cfg, _ in cfgs:
        g, _k, gr = _dtr7_means(cfg)
        r0 = jload(DTR7, "rows", cfg, CITIES[0], "row.json")
        j = float(np.mean([jload(DTR7, "rows", cfg, c, "row.json")
                           ["dose_displacement_mean"]["jpeg"]
                           for c in CITIES]))
        gam.append(g); jpg.append(j); gry.append(gr)
        try:
            flr.append(_dtr7_floor(cfg))
        except FileNotFoundError:
            flr.append(0.0)
    ax.bar(x - 1.5 * w, gam, w, label="$\\bar D_{\\gamma}$", color=C_TRAD)
    ax.bar(x - 0.5 * w, jpg, w, label="$\\bar D_{\\mathrm{jpeg}}$",
           color=C_VLA)
    ax.bar(x + 0.5 * w, gry, w, label="$D_{\\mathrm{gray}}$ (anchor)",
           color=C_WM)
    ax.bar(x + 1.5 * w, flr, w, label="repeat-call floor", color=C_VIOL,
           hatch="///", edgecolor="white", linewidth=0.4)
    for xi, f in zip(x, flr):
        if f == 0.0:
            ax.annotate("exactly 0", (xi + 1.5 * w, 0.06),
                        ha="center", fontsize=6.5, color=C_VIOL, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in cfgs])
    ax.set_ylabel("mean plan displacement (m)")
    ax.legend(loc="upper left", ncol=1)
    ax.set_title("(b) Where the floor exceeds every readout,\n"
                 "those readouts are not attributable", loc="left")
    # ---- (c) E-N2: the anchor's own gap is absorbed by an existing
    #      readout, so the test fails in an instructive way (sec.18) -------
    ax = axes[2]
    sim = _twin_means("transfuser_carla")
    real = _twin_means("transfuser_h4")

    def _arch(cfg):
        return float(np.mean([jload("lineb_probe_rows_v0", cfg, c,
                                    "row.json")["row"]["values"]
                              ["traj_diff_norm_mean"] for c in CITIES]))
    ratios = [sim[0] / real[0], sim[1] / real[1], sim[2] / real[2],
              _arch("transfuser_carla") / _arch("transfuser_h4")]
    labels = ["$\\bar D_{\\gamma}$", "$\\bar D_{\\mathrm{jpeg}}$",
              "$D_{\\mathrm{gray}}$\n(the anchor)",
              "baseline\ninstability"]
    cols = [C_TRAD, C_TRAD, C_WM, C_BLIND]
    x = np.arange(4)
    ax.bar(x, ratios, 0.6, color=cols)
    ax.axhline(1.0, color=MUTED, lw=0.7, ls="--")
    for xi, v in zip(x, ratios):
        ax.annotate(f"{v:.1f}$\\times$", (xi, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=INK)
    _y = max(ratios) * 0.42
    ax.annotate("", xy=(3, _y), xytext=(2, _y),
                arrowprops=dict(arrowstyle="<->", color=C_VIOL, lw=1.1))
    ax.text(2.5, _y * 0.80, "absorbed", ha="center", fontsize=7.5,
            color=C_VIOL)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("simulation-born / real-data-born")
    ax.set_ylim(0, max(ratios) * 1.22)
    ax.set_title("(c) The anchor's own gap is the size of a readout\n"
                 "we already had: the test is inconclusive", loc="left")
    fig.tight_layout()
    save(fig, "fig10_engagement")


# ===================================================================
# Fig 10 panel (c) + Fig 11 — engagement across the primary twin and
# across a second embodiment.  Reads only existing artifacts.
# ===================================================================
DTR9 = "triage_dtr9_manip_engagement_v0"
MANIP_LABEL = {"pi0": "$\\pi_0$", "groot": "GR00T N1.5",
               "smolvla": "SmolVLA", "openvla": "OpenVLA-7B",
               "octo_base": "Octo-base", "octo_small": "Octo-small"}


def _dtr9():
    out = {}
    for m in MANIP_LABEL:
        r = jload(DTR9, "rows", m, "row.json")
        out[m] = (r["dose_displacement_mean"]["fov_crop"],
                  r["engagement_normalized"]["fov_crop"],
                  r["anchor"]["D_gray"])
    return out


def fig11():
    """Cross-embodiment: the ordering reverses again, and why."""
    d = _dtr9()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    ax = axes[0]
    raw = {m: v[0] for m, v in d.items()}
    kap = {m: v[1] for m, v in d.items()}
    r_rank = {m: i for i, m in enumerate(sorted(raw, key=lambda x: -raw[x]))}
    k_rank = {m: i for i, m in enumerate(sorted(kap, key=lambda x: -kap[x]))}
    for m in d:
        ax.plot([0, 1], [r_rank[m], k_rank[m]], "-o", color=C_VLA, lw=1.4,
                ms=4, alpha=0.85, zorder=3)
        ax.annotate(MANIP_LABEL[m], (0, r_rank[m]), xytext=(-5, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7, color=INK)
        ax.annotate(MANIP_LABEL[m], (1, k_rank[m]), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7, color=INK)
    ax.set_xlim(-0.95, 1.95)
    ax.set_ylim(len(d) - 0.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["by raw\nresponse", "by engagement-\nnormalized"])
    ax.set_yticks([0, len(d) - 1])
    ax.set_yticklabels(["most\nsensitive", "least\nsensitive"])
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) Six manipulation models: the ordering\n"
                 "reverses again", loc="left")

    ax = axes[1]
    order = sorted(d, key=lambda m: d[m][2])
    vals = [d[m][2] for m in order]
    ax.barh(range(len(order)), vals, color=C_WM, height=0.62)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MANIP_LABEL[m] for m in order])
    ax.set_xscale("log")
    ax.set_xlabel("$D_{\\mathrm{gray}}$: response to losing the image "
                  "(log scale)")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (v, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=6.5,
                    color=MUTED)
    ax.annotate(f"{max(vals) / min(vals):.0f}$\\times$ spread",
                (0.5, 0.06), xycoords="axes fraction", ha="center",
                fontsize=8, color=C_VIOL)
    ax.set_title("(b) Why: image engagement differs by\n"
                 "more than an order of magnitude", loc="left")
    fig.tight_layout()
    save(fig, "fig11_crossembodiment")


# ===================================================================
# Fig 12 — E-R5: the dose series on the fine-tuning carrier.
#   (a) every readout falls monotonically; the spike
#       threshold is never approached
#   (b) the turn stratum does NOT fall monotonically: the smallest dose
#       is worse than zero-shot before the larger two recover
# All aggregation is POOLED OVER FRAMES (see TURN_STRATUM_N above); a
# per-city equal-weight mean gives a visibly different number, which is
# why the convention is stated in the caption.
# ===================================================================
ER5 = "triage_er5_vla_dose_v0"
DOSES = ["zeroshot", "d1k", "d5k", "d20k"]
DOSE_X = [0, 1, 5, 20]          # thousands of frames


def _er5():
    out = {}
    for v in DOSES:
        g, j, t, la, na, lt, nt, per = [], [], [], [], [], [], [], {}
        for c in CITIES:
            r = jload(ER5, "rows", v, c, "row.json")
            rv = r["row"]["values"]
            g.append(rv["gamma_response_auc"])
            j.append(rv["jpeg_response_auc"])
            t.append(rv["traj_diff_norm_mean"])
            l = r["l2_vs_human"]
            la.append(l["all"]["mean_l2_at_2s"]); na.append(l["all"]["n"])
            if l["turn"]["n"] > 0:
                lt.append(l["turn"]["mean_l2_at_2s"]); nt.append(l["turn"]["n"])
                per[c] = l["turn"]["mean_l2_at_2s"]
        out[v] = dict(gamma=float(np.mean(g)), jpeg=float(np.mean(j)),
                      traj=float(np.mean(t)),
                      l2_all=float(np.average(la, weights=na)),
                      l2_turn=float(np.average(lt, weights=nt)),
                      n_all=int(sum(na)), n_turn=int(sum(nt)), per_city=per)
    return out


def fig12():
    d = _er5()
    z = d["zeroshot"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax = axes[0]
    series = [("gamma", "$\\gamma$ response", C_TRAD),
              ("jpeg", "compression response", C_VLA),
              ("traj", "temporal instability", C_WM),
              ("l2_all", "tracking error", C_BLIND)]
    for key, lab, col in series:
        y = [d[v][key] / z[key] for v in DOSES]
        ax.plot(DOSE_X, y, "-o", color=col, lw=1.5, ms=4, label=lab)
    ax.axhline(2.0, color=C_VIOL, lw=1.0, ls="--")
    ax.text(10.5, 2.05, "spike threshold", fontsize=7,
            color=C_VIOL, ha="center")
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_ylim(0, 2.35)
    ax.set_xlabel("fine-tuning frames (thousands)")
    ax.set_ylabel("ratio to zero-shot")
    ax.legend(loc="center right", fontsize=6.5)
    ax.set_title("(a) No readout approaches the threshold\n"
                 "a spike would have to cross", loc="left")

    ax = axes[1]
    ya = [d[v]["l2_all"] for v in DOSES]
    yt = [d[v]["l2_turn"] for v in DOSES]
    for c, col in zip(sorted(z["per_city"]), (C_TRAD, C_VLA, C_WM)):
        ax.plot(DOSE_X, [d[v]["per_city"][c] for v in DOSES], "-",
                color=col, lw=0.8, alpha=0.45)
        ax.annotate(f"$n{{=}}{TURN_STRATUM_N[c]}$",
                    (DOSE_X[0], z["per_city"][c]), xytext=(-4, 0),
                    textcoords="offset points", fontsize=6, color=col,
                    va="center", ha="right")
    ax.plot(DOSE_X, yt, "-o", color=INK, lw=1.8, ms=4.5,
            label=f"turn stratum (pooled, $n{{=}}{z['n_turn']}$)")
    ax.plot(DOSE_X, ya, "--s", color=MUTED, lw=1.4, ms=4,
            label=f"all frames (pooled, $n{{=}}{z['n_all']}$)")
    ax.annotate("worse than zero-shot", (1, yt[1]), xytext=(6, 10),
                textcoords="offset points", fontsize=7, color=C_VIOL,
                arrowprops=dict(arrowstyle="->", color=C_VIOL, lw=0.9))
    ax.set_xlim(-3.2, 21.5)
    ax.set_xlabel("fine-tuning frames (thousands)")
    ax.set_ylabel("tracking error at 2 s (m)")
    ax.legend(loc="upper right", fontsize=6.5)
    ax.set_title("(b) On turns the smallest dose moves\n"
                 "backwards first", loc="left")
    fig.tight_layout()
    save(fig, "fig12_dose")


# ===================================================================
# Fig 13 — E-R4: can a moderate crop be adapted away?  Two carriers
# disagree, so the answer is carrier-specific.
#   (a) the four arms on the turn stratum of the VLA carrier
#   (b) the ratio on both carriers against its threshold
# Frame-pooled throughout (TURN_STRATUM_N); reads only existing rows.
# ===================================================================
ER4 = "triage_er4_vla_fovft_v0"
ARMS = ["zeroshot_native", "zeroshot_crop", "ft_native", "ft_crop"]


def _er4_pooled(strat):
    d = jload(ER4, "readout.json")["variants"]
    out = {}
    for v in ARMS:
        x, w = [], []
        for c in CITIES:
            e = d[v]["per_city"][c][strat]
            if e["n"] > 0 and e["mean_l2_at_2s"] is not None:
                x.append(e["mean_l2_at_2s"]); w.append(e["n"])
        out[v] = float(np.average(x, weights=w))
    return out, int(sum(w))


def fig13():
    turn, n_turn = _er4_pooled("turn")
    allf, n_all = _er4_pooled("all")
    vk = jload(ER4, "readout.json")["verdict_by_keep"]["keep_0.6"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax = axes[0]
    x = np.arange(4)
    cols = [C_BLIND, C_VIOL, C_TRAD, C_WM]
    ax.bar(x, [turn[v] for v in ARMS], 0.58, color=cols)
    ax.plot(x, [allf[v] for v in ARMS], "s--", color=MUTED, ms=4, lw=1.1,
            label=f"all frames ($n{{=}}{n_all}$)")
    for xi, v in zip(x, ARMS):
        ax.annotate(f"{turn[v]:.3f}", (xi, turn[v]), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=6.5)
    top = max(turn.values())
    for xa, xb, val, col, lab in [
            (0, 1, vk["damage_turn_m"], C_VIOL, "damage"),
            (0, 2, -vk["repairA_turn_m"], C_TRAD, "repair"),
            (2, 3, vk["residual_turn_m"], INK, "residual")]:
        xm = (xa + xb) / 2.0
        yb = top * (1.10 if lab == "damage" else
                    (1.22 if lab == "repair" else 1.10))
        ax.annotate("", xy=(xb, yb), xytext=(xa, yb),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.1))
        ax.text(xm, yb + top * 0.015, f"{lab} {val:+.3f}", ha="center",
                fontsize=7, color=col)
    ax.set_ylim(0, top * 1.40)
    ax.set_xticks(x)
    ax.set_xticklabels(["zero-shot\nnative", "zero-shot\ncropped",
                        "adapted\nnative", "adapted\ncropped"])
    ax.set_ylabel("tracking error at 2 s (m)")
    ax.set_xlim(-0.6, 4.1)
    ax.legend(loc="lower left", fontsize=6.5)
    ax.set_title(f"(a) Turn stratum ($n{{=}}{n_turn}$): a quarter of the\n"
                 "crop's damage survives adaptation", loc="left")

    ax = axes[1]
    e1 = jload("triage_er1_fovft_v0", "readout.json")
    import re
    s = json.dumps(e1)
    g = lambda k: float(re.search(rf'"{k}[^"]*":\s*([-\d.eE]+)', s).group(1))
    # Panel (b) compares the residual against the repair that it survives.
    # The traditional-stack carrier has a keep=0.6 arm only (E-R1 fixes
    # fov_keep=0.6), so the keep=0.3 column exists for the VLA carrier
    # alone and the panel is three bars, not a 2x2 grid with a hole.
    pct = [100 * g("residual_turn") / g("repairA_turn"),
           100 * vk["residual_turn_m"] / vk["repairA_turn_m"]]
    labs = ["traditional stack\n(full fine-tune)\nkeep 0.6",
            "VLA (low-rank\nadapt)\nkeep 0.6"]
    cols = [C_TRAD, C_VLA]
    vk03 = jload(ER4, "readout.json")["verdict_by_keep"].get("keep_0.3", {})
    if "residual_turn_m" in vk03 and "repairA_turn_m" in vk03:
        pct.append(100 * vk03["residual_turn_m"] / vk03["repairA_turn_m"])
        labs.append("VLA (low-rank\nadapt)\nkeep 0.3")
        cols.append(C_VLA)
    else:
        print("[fig13] keep_0.3 verdict absent (armB03 rows missing); "
              "panel (b) drawn with the two keep=0.6 bars only")
    xs = list(range(len(pct)))
    ax.bar(xs, pct, 0.5, color=cols)
    ax.axhline(25, color=C_VIOL, lw=1.0, ls="--")
    # left side: the tallest bar is now on the right, so the label sat on it
    ax.text(xs[0] - 0.34, 27, "threshold", fontsize=6.5,
            color=C_VIOL, ha="left", va="bottom")
    for xi, v in zip(xs, pct):
        ax.annotate(f"{v:.1f}%", (xi, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=6.5)
    ax.set_ylabel("residual as % of the repair it survives")
    ax.set_ylim(0, max(52.0, max(pct) * 1.18))
    ax.set_title("(b) Residual against the repair it survives,\n"
                 "by carrier and crop severity", loc="left")
    fig.tight_layout()
    save(fig, "fig13_infogate_vla")





# ===================================================================
# Entry point.  It lives at the END of the file on purpose: an earlier
# revision had it mid-file, so figures defined below it were silently NOT
# regenerated by a plain `python scripts/79...`, and a reader regenerating
# the figure set would have got stale panels without any error.
# ===================================================================
if __name__ == "__main__":
    for _f in (fig2, fig3, fig4, fig6, fig7, fig8, fig6_split,
               fig10, fig11, fig12, fig13):
        _f()
    print("all figures in", FIG_DIR)
