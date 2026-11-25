#!/usr/bin/env python3
"""Sensing-gate analysis on vehicle samples, with the offline definitions.

kappa (V3a): on the virtual CAM_F0 view, every perturb_stride-th window, all fed frames degraded:
  field of view  center crop keeping {0.9, 0.75, 0.6} of each dimension (perturb.fov_crop), resized back
                 (INTER_LINEAR)                                     [scripts/102 degrade_fn "fov"]
  resolution     downscale by {0.75, 0.5, 0.33} (INTER_AREA), upscaled back (INTER_LINEAR)   ["res"]
  displacement   mean waypoint distance to the clean plan; kappa = D_fov(0.75) / D_res(0.75)
r7: on synchronous windows, the stock forward frames fed unchanged as the front camera vs the virtual
  view; deviation = distance between the two plans' 2 s waypoints; turning = logged heading change
  >= 15 deg over the next 2 s, straight = all other windows; ratio = mean(turning) / mean(straight);
  fewer than 30 turning windows -> NOT JUDGED. Also each view's 2 s endpoint error against the driver.
Virtual plans can be reused from the probe output (--virtual-plans) so the same plans are compared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "src"))
from toolkit.probes.perturb import fov_crop                        # noqa: E402
from triage_vehicle.run_driveva_samples import windows                # noqa: E402
import probe_rows as v2                        # noqa: E402

FOV_KEEPS = [0.9, 0.75, 0.6]
RES_KEEPS = [0.75, 0.5, 0.33]
MIN_TURNING = 30
WP_2S = 3            # index of the 2.0 s waypoint in 8 x 0.5 s


def degrade(img, kind, keep):
    if kind == "fov":
        crop, _ = fov_crop(img, np.eye(3), keep)
        return cv2.resize(crop, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * keep)), max(1, int(h * keep))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class StockView:
    """Wraps a V2 adapter so it reads the stock forward frames instead of the virtual view."""

    def __init__(self, ad, sd):
        self.ad, self.sd = ad, sd

    def frames(self, w):
        return self.ad.frames(w, key="stock_front")        # the same frames this model consumes, stock view

    def plan(self, w):
        return self.ad.plan(w, self.frames(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["kappa", "r7"])
    ap.add_argument("model", choices=["driveva", "recogdrive", "alpamayo", "drivevlaw0", "wamflow", "cv"])
    ap.add_argument("samples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--virtual-plans", default=None, help="V2 probe output json to reuse its plans")
    ap.add_argument("--perturb-stride", type=int, default=10)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--min-turning", type=int, default=MIN_TURNING,
                    help=" minimum turning windows (30; 20 for the V4 shadow lap only)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refuse: {out} exists")
    sd = Path(a.samples)
    wins, skipped = windows([json.loads(l) for l in open(sd / "samples.jsonl")])
    if a.max_windows:
        wins = wins[: a.max_windows]
    rows_all = [json.loads(l) for l in open(sd / "samples.jsonl")]
    ad = v2.ADAPTERS[a.model](sd)
    if hasattr(ad, "set_samples"):
        ad.set_samples(rows_all)
    vplans = {}
    if a.virtual_plans:
        vplans = {int(k): np.asarray(v) for k, v in json.load(open(a.virtual_plans))["plans"].items()}
    t0 = time.time()
    res = {"what": a.what, "model": a.model, "samples": str(sd.resolve()), "n_windows": len(wins),
           "windows_skipped_missing": skipped, "settings": ad.settings,
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if a.what == "kappa":
        D = {"fov": {k: [] for k in FOV_KEEPS}, "res": {k: [] for k in RES_KEEPS}}
        for w in wins[:: a.perturb_stride]:
            k = w[-1]["k"]
            base = vplans.get(k)
            if base is None:
                base = ad.plan(w)
            for kind, keeps in (("fov", FOV_KEEPS), ("res", RES_KEEPS)):
                for kp in keeps:
                    D[kind][kp].append(v2.disp(base, v2.plan_perturbed(ad, w, lambda f, _k=kind, _p=kp: degrade(f, _k, _p))))
        mean = {kind: {str(kp): (float(np.mean(v)) if v else None) for kp, v in d.items()} for kind, d in D.items()}
        f75, r75 = mean["fov"]["0.75"], mean["res"]["0.75"]
        res.update(D_mean=mean, n_stride_windows=len(wins[:: a.perturb_stride]),
                   kappa=(f75 / r75) if (f75 is not None and r75) else None)
    else:
        rows = []
        for w in wins:
            need = (-3, -1) if a.model == "drivevlaw0" else range(-ad.views, 0)   # the frames the model reads
            if not all(w[i].get("stock_front") for i in need):
                rows.append({"k": w[-1]["k"], "status": "stock frame missing"})
                continue
            k = w[-1]["k"]
            pv = vplans.get(k)
            if pv is None:
                pv = ad.plan(w)
            ps = StockView(ad, sd).plan(w)
            drv = w[-1].get("driver_future_xy")
            rows.append({"k": k, "status": "ok", "turning": w[-1].get("turning"),
                         "dev_2s": float(np.linalg.norm(np.asarray(pv)[WP_2S] - np.asarray(ps)[WP_2S])),
                         "err_virtual_2s": None if not drv else float(np.linalg.norm(np.asarray(pv)[WP_2S] - np.asarray(drv[WP_2S]))),
                         "err_stock_2s": None if not drv else float(np.linalg.norm(np.asarray(ps)[WP_2S] - np.asarray(drv[WP_2S]))),
                         "plan_stock": np.round(ps, 4).tolist()})
        ok = [r for r in rows if r["status"] == "ok" and r["turning"] is not None]
        tr = [r["dev_2s"] for r in ok if r["turning"]]
        st = [r["dev_2s"] for r in ok if not r["turning"]]

        def m(x):
            return float(np.mean(x)) if x else None
        judged = len(tr) >= a.min_turning and bool(st)
        res.update(n_ok=len(ok), n_turning=len(tr), n_straight=len(st), min_turning=a.min_turning,
                   dev_turning_mean=m(tr), dev_straight_mean=m(st),
                   ratio=(m(tr) / m(st)) if (tr and st and m(st)) else None,
                   verdict=("NOT JUDGED (insufficient turning frames)" if not judged else
                            ("HELD" if m(tr) / m(st) > 1 else "NOT HELD")),
                   err_virtual_2s_turning=m([r["err_virtual_2s"] for r in ok if r["turning"] and r["err_virtual_2s"] is not None]),
                   err_stock_2s_turning=m([r["err_stock_2s"] for r in ok if r["turning"] and r["err_stock_2s"] is not None]),
                   err_virtual_2s_straight=m([r["err_virtual_2s"] for r in ok if not r["turning"] and r["err_virtual_2s"] is not None]),
                   err_stock_2s_straight=m([r["err_stock_2s"] for r in ok if not r["turning"] and r["err_stock_2s"] is not None]),
                   rows=rows)
    res["wall_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, out)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
