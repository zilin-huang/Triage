"""Trajectory format shared with the vehicle's existing controller.

The controller (path_tracker_client_local.py, sha 5d9b379f) consumes
/alpamayo/trajectory as JSON {seq, stamp, dt, ego_pose, xy}: points in the ego
frame at inference time (x forward, y left, rear axle), dt apart, and the global
pose at inference time for latency compensation. It accepts any number of points
>= 2 and estimates speed from the first 5 segments (0.5 s at dt 0.1).

DriveVA emits 8 poses at 0.5 s (t = 0.5 .. 4.0 s). They are resampled to dt 0.1
over the same 4.0 s (40 points); nothing is extrapolated beyond the model's horizon.
"""
from __future__ import annotations

import json
import time

import numpy as np

DT = 0.1


def resample_driveva(traj_8x3: np.ndarray, native_dt: float = 0.5, dt: float = DT) -> np.ndarray:
    """(8,3) x,y,heading at t = native_dt*(1..8) -> (N,2) x,y at t = dt*(1..N), same horizon.

    The origin (t = 0) is the ego position, so the first interval is interpolated from
    (0, 0). Linear interpolation in x and y separately.
    """
    traj = np.asarray(traj_8x3, float)
    if traj.ndim != 2 or traj.shape[1] < 2 or traj.shape[0] < 1:
        raise ValueError(f"bad trajectory shape {traj.shape}")
    if not np.all(np.isfinite(traj[:, :2])):
        raise ValueError("non-finite trajectory")
    t_src = native_dt * np.arange(0, traj.shape[0] + 1)
    x_src = np.concatenate([[0.0], traj[:, 0]])
    y_src = np.concatenate([[0.0], traj[:, 1]])
    n = int(round(t_src[-1] / dt))
    t_dst = dt * np.arange(1, n + 1)
    return np.stack([np.interp(t_dst, t_src, x_src), np.interp(t_dst, t_src, y_src)], axis=1)


def payload(seq: int, xy: np.ndarray, ego_pose, dt: float = DT, stamp: float | None = None,
            extra: dict | None = None) -> str:
    d = {
        "seq": int(seq),
        "stamp": time.time() if stamp is None else float(stamp),
        "dt": float(dt),
        "ego_pose": [float(v) for v in ego_pose],
        "xy": [[round(float(x), 4), round(float(y), 4)] for x, y in np.asarray(xy, float)],
    }
    if extra:
        d.update(extra)   # ignored by the controller (it reads seq, xy, ego_pose only)
    return json.dumps(d)
