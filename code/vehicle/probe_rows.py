#!/usr/bin/env python3
"""Probe rows on vehicle samples: the local readings that the selection rule consumes.

Every readout reuses the offline definitions, so a local row is comparable with the
four-city rows (outputs/lineb_probe_rows_v0) the rule was defined on:

  temporal instability (TI)  scripts/68: consecutive plans 0.5 s apart; the earlier plan is moved
                             into the later ego frame (INS motion, response.se2_transform) and
                             compared on the overlapping waypoints; mean and p95 over pairs
  A_gamma, A_jpeg            scripts/68: every perturb_stride-th window, all fed frames degraded
                             (perturb.gamma_lowlight 1.8 / 2.6, seed 0; JPEG 50 / 20), displacement =
                             mean waypoint distance to the unperturbed plan; features.response_auc
                             with severities gamma-1 and 90-q; D_bar = plain mean of the two doses
  D_gray, D_noise            scripts/110: every fed view replaced by constant gray 128 / seeded uniform
                             noise (seed 20260823 + 97*window + view), same displacement
  repeat-call floor          appendix E: 6 consecutive calls on the same unperturbed input, mean
                             displacement of calls 2..6 from call 1 (first 5 stride windows)
  sign diagnostic            scripts/68: windows whose logged heading changes by > 2 deg over the
                             next 2 s and whose plan ends > 0.05 m off-axis; share with the same sign

Models: driveva (driveva env), recogdrive (recogdrive env), cv (constant velocity, blind anchor;
any env). Windows: 5 consecutive "ok" samples; missing samples break windows, never patched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from toolkit.probes import features as F          # noqa: E402
from toolkit.probes.perturb import gamma_lowlight  # noqa: E402
from toolkit.probes.response import se2_transform  # noqa: E402
from triage_vehicle.run_driveva_samples import windows  # noqa: E402

GAMMA_LEVELS = [1.0, 1.8, 2.6]
JPEG_LEVELS = [90, 50, 20]
GRAY_LEVEL = 128
NOISE_BASE_SEED = 20260823
REPEATS = 6


def degrade(rgb, kind, level):
    """scripts/68 degrade_file, in memory (the runtime JPEG-encodes at q95 as degrade_file wrote)."""
    if kind == "gamma":
        return gamma_lowlight(rgb, gamma=level, seed=0)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(level)])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def ablate(shape, kind, widx, view):
    h, w = shape[:2]
    if kind == "gray":
        return np.full((h, w, 3), GRAY_LEVEL, np.uint8)
    rng = np.random.default_rng(NOISE_BASE_SEED + 97 * widx + view)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def rel_pose(a, b):
    """pose b [x,y,yaw] expressed in frame a -> (dx, dy, dyaw)."""
    wx, wy = b[0] - a[0], b[1] - a[1]
    c, s = math.cos(-a[2]), math.sin(-a[2])
    d = b[2] - a[2]
    return wx * c - wy * s, wx * s + wy * c, math.atan2(math.sin(d), math.cos(d))


class DriveVAAdapter:
    views = 5

    def __init__(self, sd):
        from triage_vehicle.driveva_runtime import DriveVARuntime
        self.rt, self.sd = DriveVARuntime(), sd
        self.settings = self.rt.settings

    def frames(self, w, key="virtual_cam_f0"):
        return [cv2.cvtColor(cv2.imread(str(self.sd / s[key])), cv2.COLOR_BGR2RGB) for s in w]

    def plan(self, w, frames=None):
        fr = frames if frames is not None else self.frames(w)
        return self.rt.infer(fr, w[-1]["ego_vel"], w[-1]["cmd"] or [0, 1, 0, 0])[0][:, :2]


class ReCogDriveAdapter:
    views = 1          # cam_type single: only the current CAM_F0 image is read

    def __init__(self, sd):
        from triage_vehicle.recogdrive_runtime import ReCogDriveRuntime
        self.rt, self.sd = ReCogDriveRuntime(), sd
        self.settings = self.rt.settings
        self.tmp = tempfile.mkdtemp(prefix="triage_v2_recog_")

    def frames(self, w, key="virtual_cam_f0"):
        return [cv2.cvtColor(cv2.imread(str(self.sd / w[-1][key])), cv2.COLOR_BGR2RGB)]

    def plan(self, w, frames=None):
        cur = w[-1]
        if frames is None:
            path = self.sd / cur["virtual_cam_f0"]
        else:            # degraded view written as JPEG q95, as scripts/68 degrade_file did
            path = Path(self.tmp) / "cur.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        hist = []
        for s in w[-4:]:
            dx, dy, dyaw = rel_pose(cur["pose"], s["pose"])
            hist.append([dx, dy, dyaw])
        prev = w[-2]
        acc = [(cur["ego_vel"][0] - prev["ego_vel"][0]) / 0.5, (cur["ego_vel"][1] - prev["ego_vel"][1]) / 0.5]
        return self.rt.infer(path, hist, cur["ego_vel"], acc, cur["cmd"] or [0, 1, 0, 0])[0][:, :2]


class AlpamayoAdapter:
    """Alpamayo-1.5 via scripts/92's AlpamayoRunner (weights, processor, message format, sampling
    settings: seed 42 per call, top_p 0.98, temperature 0.6, max 256 generated tokens, front camera as
    index 1); only the inputs change: the last 4 virtual CAM_F0 frames of the window (0.5 s apart) and
    the ego history interpolated from the INS poses of those 4 samples to 16 points at 0.1 s, as
    scripts/92 build_ego_history does from the nuPlan poses (planar here: z = 0, rotation about z)."""
    views = 4

    def __init__(self, sd):
        import importlib.util
        spec = importlib.util.spec_from_file_location("p92", str(REPO / "scripts" / "92_triage_alpamayo15_probe.py"))
        self.h = importlib.util.module_from_spec(spec); sys.modules["p92"] = self.h; spec.loader.exec_module(self.h)
        self.runner = self.h.AlpamayoRunner(self.h.CONFIGS["front"])
        self.sd = sd
        self.settings = {"config": "front (camera index 1)", "seed": self.h.SEED, "top_p": 0.98, "temperature": 0.6,
                         "max_generation_length": 256, "attn": self.runner.attn_used, "frames": 4}

    def frames(self, w, key="virtual_cam_f0"):
        return [cv2.cvtColor(cv2.imread(str(self.sd / s[key])), cv2.COLOR_BGR2RGB) for s in w[-4:]]

    def _history(self, w4):
        import scipy.spatial.transform as spt
        import torch
        t_key = np.array([s["t"] for s in w4], float)
        xyz_key = np.array([[s["pose"][0], s["pose"][1], 0.0] for s in w4], float)
        rots_key = spt.Rotation.from_euler("z", np.array([s["pose"][2] for s in w4], float)[:, None])
        t_q = np.clip(t_key[-1] + np.arange(-(self.h.EGO_HIST_STEPS - 1), 1) * self.h.EGO_HIST_DT, t_key[0], t_key[-1])
        xyz_q = np.stack([np.interp(t_q, t_key, xyz_key[:, i]) for i in range(3)], axis=1)
        rot_q = spt.Slerp(t_key, rots_key)(t_q)
        r0_inv = rots_key[-1].inv()
        return (torch.from_numpy(r0_inv.apply(xyz_q - xyz_key[-1])).float()[None, None],
                torch.from_numpy((r0_inv * rot_q).as_matrix()).float()[None, None])

    def plan(self, w, frames=None):
        import torch
        r = self.runner
        fr = frames if frames is not None else self.frames(w)
        imgs = torch.stack([torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1) for f in fr])[None]
        messages = r.helper.create_message(frames=imgs.flatten(0, 1), camera_indices=torch.tensor([1], dtype=torch.int64),
                                           num_frames_per_camera=4)
        inputs = r.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                 continue_final_message=True, return_dict=True, return_tensors="pt")
        hx, hr = self._history(w[-4:])
        data = r.helper.to_device({"tokenized_data": inputs, "ego_history_xyz": hx, "ego_history_rot": hr}, "cuda")
        torch.cuda.manual_seed_all(self.h.SEED)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _ = r.model.sample_trajectories_from_data_with_vlm_rollout(
                data=data, top_p=0.98, temperature=0.6, num_traj_samples=1, max_generation_length=256)
        native = pred_xyz[0, 0, 0].float().cpu().numpy()
        return native[self.h.FROZEN_8PT_IDX, :2].copy()


class DriveVLAW0Adapter:
    """DriveVLA-W0 via scripts/102's Runtime (official model + dataset pipeline, the paper-90.2 checkpoint,
    sdpa, bf16) and its scene schema: VQ codes of the current frame (k) and the frame 1 s earlier (k-2),
    each resized to 256x144 before encoding as encode_jpg does; 12-step relative SE2 actions over k-3..k+8
    (and k-5..k+6 for the 1 s-earlier window) from the INS poses, normalised with the test stats; one text
    command per step from the turn-signal command. As offline, the future steps come from the log; a step
    whose pose is unavailable contributes a zero action, as an out-of-log index does in scripts/102."""
    views = 2

    def __init__(self, sd):
        import importlib.util
        spec = importlib.util.spec_from_file_location("p102", str(REPO / "scripts" / "102_lineb_probe_drivevlaw0.py"))
        self.h = importlib.util.module_from_spec(spec); sys.modules["p102"] = self.h; spec.loader.exec_module(self.h)
        # host paths (repo rule: from the environment; the lab values stay the default)
        self.h.CKPT = os.environ.get("TOOLKIT_DRIVEVLAW0_CKPT", self.h.CKPT)
        self.h.DV = os.environ.get("TOOLKIT_DRIVEVLAW0_REPO", self.h.DV)
        self.rt = self.h.Runtime()
        self.sd, self.by_k = sd, {}
        self.tmp = tempfile.mkdtemp(prefix="triage_v2_w0_")
        self.settings = {"checkpoint": "Emu3_Multi_Anchor_PDMS_90.3", "attn": "sdpa", "dtype": "bf16",
                         "loading_missing": self.rt.loading_missing}

    def set_samples(self, rows):
        self.by_k = {r["k"]: r for r in rows}

    def frames(self, w, key="virtual_cam_f0"):
        return [cv2.cvtColor(cv2.imread(str(self.sd / s[key])), cv2.COLOR_BGR2RGB) for s in (w[-3], w[-1])]

    def _encode(self, rgb, name, degrade=None):
        from PIL import Image
        vt, proc = self.rt.vision_tokenizer()
        img = Image.fromarray(rgb).resize((256, 144))
        if degrade is not None:          # scripts/102 encode_jpg: the degradation acts on the 256x144 image
            img = Image.fromarray(degrade(np.array(img.convert("RGB"))))
        t = proc(img, return_tensors="pt")["pixel_values"].to(self.rt.device)
        with self.rt.torch.no_grad():
            codes = vt.encode(t)
        out = os.path.join(self.tmp, name)
        np.save(out, codes.detach().cpu().numpy())
        return out

    def _window(self, center):
        acts, texts = [], []
        for j in range(center - 3, center + 9):
            a_, b_ = self.by_k.get(j), self.by_k.get(j + 1)
            if a_ and b_ and a_.get("pose") and b_.get("pose"):
                acts.append(self.h.rel_se2(a_["pose"], b_["pose"]))
            else:
                acts.append(np.zeros(3))
            c = (a_ or {}).get("cmd")
            texts.append(self.h.TEXT_NAMES[int(np.argmax(c))] if c else self.h.TEXT_NAMES[3])
        return np.array(acts), texts

    def plan_degraded(self, w, fn, frames=None):
        """Perturbation entry used by V2/V3 for this model: fn acts after the 256x144 resize, as offline."""
        return self.plan(w, frames, degrade=fn)

    def plan(self, w, frames=None, degrade=None):
        k = w[-1]["k"]
        fr = frames if frames is not None else self.frames(w)
        pre_npy, cur_npy = self._encode(fr[0], "pre.npy", degrade), self._encode(fr[1], "cur.npy", degrade)
        acts, texts = self._window(k)
        pacts, ptexts = self._window(k - 2)
        imgs = [None] * 12; pimgs = [None] * 12
        imgs[3], pimgs[3] = cur_npy, pre_npy
        sc = {"text": texts, "image": imgs, "action": acts, "pre_1s_text": ptexts, "pre_1s_image": pimgs,
              "pre_1s_action": pacts}
        return self.rt.plan(self.h.normalize_scene(self.rt, sc))[:, :2]


class WAMFlowAdapter:
    """WAM-Flow via scripts/64's build_agent("wamflow") (official agent yaml, discrete_fm_steps 2, seed 99,
    fp32) and its _WamFlowShim; the agent reads only the current CAM_F0 image path and the current ego
    status (command, velocity, acceleration), so the AgentInput carries exactly those."""
    views = 1

    def __init__(self, sd):
        import importlib.util
        spec = importlib.util.spec_from_file_location("p64", str(REPO / "scripts" / "64_lineb_shadow_probes_navsim.py"))
        self.h = importlib.util.module_from_spec(spec); sys.modules["p64"] = self.h; spec.loader.exec_module(self.h)
        repo = os.environ.get("TOOLKIT_WAMFLOW_REPO")
        ck = os.environ.get("TOOLKIT_WAMFLOW_CKPT")         # dir holding fudoki/ and wam-flow/
        if repo or ck:
            root, yaml_rel, over = self.h.AGENTS["wamflow"]
            over = dict(over)
            if ck:
                for key, val in list(over.items()):
                    if isinstance(val, str) and val.startswith("/work/checkpoints/wamflow"):
                        over[key] = val.replace("/work/checkpoints/wamflow", ck)
            self.h.AGENTS["wamflow"] = (repo or root, yaml_rel, over)
        self.agent = self.h.build_agent("wamflow")
        self.sd = sd
        self.tmp = tempfile.mkdtemp(prefix="triage_v2_wam_")
        self.settings = {"discrete_fm_steps": 2, "seed": 99, "dtype": "fp32", "agent": "scripts/64 build_agent(wamflow)"}

    def frames(self, w, key="virtual_cam_f0"):
        return [cv2.cvtColor(cv2.imread(str(self.sd / w[-1][key])), cv2.COLOR_BGR2RGB)]

    def plan(self, w, frames=None):
        from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus
        cur, prev = w[-1], w[-2]
        if frames is None:
            path = self.sd / cur["virtual_cam_f0"]
        else:
            path = Path(self.tmp) / "cur.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        acc = [(cur["ego_vel"][0] - prev["ego_vel"][0]) / 0.5, (cur["ego_vel"][1] - prev["ego_vel"][1]) / 0.5]
        es = EgoStatus(ego_pose=np.zeros(3), ego_velocity=np.asarray(cur["ego_vel"], np.float32),
                       ego_acceleration=np.asarray(acc, np.float32),
                       driving_command=np.asarray(cur["cmd"] or [0, 1, 0, 0], int), in_global_frame=False)
        empty = Camera()
        cams = Cameras(cam_f0=Camera(image=str(path)), cam_l0=empty, cam_l1=empty, cam_l2=empty,
                       cam_r0=empty, cam_r1=empty, cam_r2=empty, cam_b0=empty)
        traj = self.agent.compute_trajectory(AgentInput(ego_statuses=[es] * 4, cameras=[cams] * 4, lidars=[None] * 4))
        return np.asarray(traj.poses, float)[:, :2]


class CVAdapter:
    """Constant velocity: the blind anchor. Reads no image, so every image response must be 0."""
    views = 5

    def __init__(self, sd):
        self.sd, self.settings = sd, {"model": "constant velocity, 8 x 0.5 s"}

    def frames(self, w, key="virtual_cam_f0"):
        return [cv2.cvtColor(cv2.imread(str(self.sd / s[key])), cv2.COLOR_BGR2RGB) for s in w]

    def plan(self, w, frames=None):
        vx, vy = w[-1]["ego_vel"]
        t = 0.5 * np.arange(1, 9)
        return np.stack([vx * t, vy * t], axis=1)


ADAPTERS = {"driveva": DriveVAAdapter, "recogdrive": ReCogDriveAdapter, "alpamayo": AlpamayoAdapter,
            "drivevlaw0": DriveVLAW0Adapter, "wamflow": WAMFlowAdapter, "cv": CVAdapter}


def plan_perturbed(ad, w, fn):
    """Plan with every consumed frame degraded by fn, at the stage the offline probe degraded it
    (full resolution for all models except DriveVLA-W0, which degrades after its 256x144 resize)."""
    if hasattr(ad, "plan_degraded"):
        return ad.plan_degraded(w, fn)
    return ad.plan(w, [fn(f) for f in ad.frames(w)])


def disp(a, b):
    h = min(len(a), len(b))
    return float(np.linalg.norm(np.asarray(a)[:h] - np.asarray(b)[:h], axis=1).mean())


def compute(ad, wins, stride, floor_keys, own=None):
    """Per-window work in the unsharded order: every owned window's plan first, then the owned stride
    windows' perturbations, anchors and (first 5 stride windows) repeat-call floor, in stride order."""
    plans, per = {}, {}
    for w in wins:
        if own is None or w[-1]["k"] in own:
            plans[w[-1]["k"]] = ad.plan(w)
    by_k = {w[-1]["k"]: w for w in wins}
    pert = {"gamma": GAMMA_LEVELS, "jpeg": JPEG_LEVELS}
    for k in stride:
        if own is not None and k not in own:
            continue
        w, base = by_k[k], plans[k]
        r = {"D": {kind: {} for kind in pert}}
        for kind, levels in pert.items():
            for lv in levels[1:]:
                r["D"][kind][lv] = disp(base, plan_perturbed(ad, w, lambda f, _k=kind, _l=lv: degrade(f, _k, _l)))
        for kind in ("gray", "noise"):
            views = iter(range(1000))
            r[kind] = disp(base, plan_perturbed(ad, w, lambda f, _k=kind: ablate(f.shape, _k, k, next(views))))
        if k in floor_keys:
            first = ad.plan(w)
            r["floor"] = float(np.mean([disp(first, ad.plan(w)) for _ in range(REPEATS - 1)]))
        per[k] = r
    return plans, per


def aggregate(model, sd, settings, wins, skipped, plans, per, stride, wall_s):
    by_k = {w[-1]["k"]: w for w in wins}
    plans = {w[-1]["k"]: plans[w[-1]["k"]] for w in wins}          # window order, as unsharded
    # sign diagnostic (scripts/68: heading change over the next 2 s > 2 deg, plan end |y| > 0.05 m)
    curv, lat = [], []
    for k, p in plans.items():
        dh = by_k[k][-1].get("dheading_2s_deg")
        if dh is not None and abs(dh) > 2.0 and abs(p[-1, 1]) > 0.05:
            curv.append(np.sign(dh)); lat.append(np.sign(p[-1, 1]))
    sign_diag = float(np.mean(np.array(curv) == np.array(lat))) if curv else None
    # temporal instability over consecutive windows
    diffs = []
    for k in sorted(plans):
        if k + 1 in plans:
            dx, dy, dyaw = rel_pose(by_k[k][-1]["pose"], by_k[k + 1][-1]["pose"])
            pp = se2_transform(plans[k], dx, dy, dyaw)
            h = min(len(pp) - 1, len(plans[k + 1]))
            if h >= 1:
                diffs.append(float(np.linalg.norm(pp[1:h + 1] - plans[k + 1][:h], axis=1).mean()))
    # perturbations, anchors, repeat floor on stride windows, collected in stride order
    pert = {"gamma": GAMMA_LEVELS, "jpeg": JPEG_LEVELS}
    D = {k: {lv: [] for lv in v[1:]} for k, v in pert.items()}
    anchor = {"gray": [], "noise": []}
    floor = []
    for k in stride:
        r = per[k]
        for kind, levels in pert.items():
            for lv in levels[1:]:
                D[kind][lv].append(r["D"][kind][lv])
        for kind in ("gray", "noise"):
            anchor[kind].append(r[kind])
        if "floor" in r:
            floor.append(r["floor"])

    def auc(kind):
        lv = pert[kind]
        sev = [g - 1.0 for g in lv] if kind == "gamma" else [lv[0] - q for q in lv]
        return F.response_auc(sev, [0.0] + [float(np.mean(D[kind][l])) if D[kind][l] else float("nan") for l in lv[1:]])

    def dbar(kind):
        v = [float(np.mean(D[kind][l])) for l in pert[kind][1:] if D[kind][l]]
        return float(np.mean(v)) if v else float("nan")

    row = {
        "model": model, "samples": str(sd.resolve()), "settings": settings,
        "n_windows": len(wins), "windows_skipped_missing": skipped, "n_stride_windows": len(stride),
        "ti": float(np.mean(diffs)) if diffs else float("nan"),
        "ti_p95": float(np.percentile(diffs, 95)) if diffs else float("nan"), "n_ti_pairs": len(diffs),
        "a_gamma": auc("gamma"), "a_jpeg": auc("jpeg"), "dbar_gamma": dbar("gamma"), "dbar_jpeg": dbar("jpeg"),
        "d_gray": float(np.mean(anchor["gray"])) if anchor["gray"] else float("nan"),
        "d_noise": float(np.mean(anchor["noise"])) if anchor["noise"] else float("nan"),
        "repeat_floor": float(np.mean(floor)) if floor else float("nan"), "n_floor_windows": len(floor),
        "sign_diag": sign_diag, "n_sign_frames": len(curv),
        "per_level_D": {k: {str(l): v for l, v in d.items()} for k, d in D.items()},
        "wall_s": wall_s,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return row, plans


def shard_owner(wins, stride, n):
    """Balanced split: stride windows (the expensive ones) round-robin by stride index, the other
    windows round-robin by their index among the non-stride windows."""
    sset = {k: i for i, k in enumerate(stride)}
    rest = [w[-1]["k"] for w in wins if w[-1]["k"] not in sset]
    owner = {k: i % n for i, k in enumerate(stride)}
    owner.update({k: i % n for i, k in enumerate(rest)})
    return owner


def write_json(out, obj):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["driveva", "recogdrive", "alpamayo", "drivevlaw0", "wamflow", "cv"])
    ap.add_argument("samples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--perturb-stride", type=int, default=10)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all (a dry run may cap)")
    ap.add_argument("--shard", default=None, help="i/n: compute only this share of the windows and write a partial file "
                    "(only for models whose calls do not depend on call order, i.e. not recogdrive)")
    ap.add_argument("--merge", nargs="+", default=None, help="partial files of all n shards -> the unsharded output")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refuse: {out} exists")
    if a.shard and a.merge:
        sys.exit("refuse: --shard and --merge together")
    if a.shard and a.model == "recogdrive":
        sys.exit("refuse: recogdrive is seeded once and run in order; sharding would change its calls")
    sd = Path(a.samples)
    rows = [json.loads(l) for l in open(sd / "samples.jsonl")]
    wins, skipped = windows(rows)
    if a.max_windows:
        wins = wins[: a.max_windows]
    all_k = sorted(w[-1]["k"] for w in wins)
    stride = all_k[:: a.perturb_stride]
    floor_keys = set(stride[:5])
    ident = {"model": a.model, "samples": str(sd.resolve()), "n_windows": len(wins), "perturb_stride": a.perturb_stride,
             "max_windows": a.max_windows, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if a.merge:
        parts = [json.load(open(p)) for p in a.merge]
        n = parts[0]["shard_n"]
        if sorted(p["shard_i"] for p in parts) != list(range(n)) or any(p["shard_n"] != n for p in parts):
            sys.exit(f"refuse: shards {[p['shard_i'] for p in parts]} of n={n} do not form a complete set")
        for p in parts:
            if p["ident"] != ident:
                sys.exit(f"refuse: shard {p['shard_i']} identity {p['ident']} != {ident}")
        plans, per = {}, {}
        for p in parts:
            plans.update({int(k): np.asarray(v, dtype=p["plan_dtype"]) for k, v in p["plans"].items()})
            per.update({int(k): {**r, "D": {kind: {float(lv) if kind == "gamma" else int(lv): x for lv, x in d.items()}
                                               for kind, d in r["D"].items()}} for k, r in p["per"].items()})
        if set(plans) != set(all_k) or set(per) != set(stride):
            sys.exit("refuse: merged shards do not cover every window and stride window exactly")
        row, plans = aggregate(a.model, sd, parts[0]["settings"], wins, skipped, plans, per, stride,
                               max(p["wall_s"] for p in parts))
        row["shards"] = [{"file": str(Path(q).resolve()), "wall_s": p["wall_s"]} for q, p in zip(a.merge, parts)]
    else:
        ad = ADAPTERS[a.model](sd)
        if hasattr(ad, "set_samples"):
            ad.set_samples(rows)
        t0 = time.time()
        own = None
        if a.shard:
            i, n = (int(x) for x in a.shard.split("/"))
            owner = shard_owner(wins, stride, n)
            own = {k for k, s in owner.items() if s == i}
        plans, per = compute(ad, wins, stride, floor_keys, own)
        if a.shard:
            dt = {str(np.asarray(v).dtype) for v in plans.values()}
            write_json(out, {"shard_i": i, "shard_n": n, "ident": ident, "settings": ad.settings,
                             "plan_dtype": dt.pop() if len(dt) == 1 else "float64",
                             "plans": {str(k): np.asarray(v).tolist() for k, v in plans.items()},
                             "per": {str(k): r for k, r in per.items()}, "wall_s": time.time() - t0})
            print(json.dumps({"model": a.model, "shard": a.shard, "windows": len(plans), "stride_windows": len(per),
                              "wall_s": time.time() - t0}))
            return
        row, plans = aggregate(a.model, sd, ad.settings, wins, skipped, plans, per, stride, time.time() - t0)
    write_json(out, {"row": row, "plans": {str(k): np.round(v, 4).tolist() for k, v in plans.items()}})
    print(json.dumps({k: row[k] for k in ("model", "n_windows", "ti", "a_gamma", "a_jpeg", "d_gray",
                                           "repeat_floor", "sign_diag", "n_sign_frames", "wall_s")}))


if __name__ == "__main__":
    main()
