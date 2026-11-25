"""Virtual camera: reproject a wide DJI frame into another camera's geometry.

The NAVSIM candidates consume nuPlan's CAM_F0 image as recorded: 1920x1080,
fx = fy = 1545 px, principal point (960, 560), with nuPlan's lens distortion
still in the image (DriveVA resizes the raw distorted JPEG to 832x480 without
undistorting). The virtual view therefore reproduces that distortion.

Only a rotation between the two cameras is modelled; the DJI and the nuPlan
camera sit at different heights and positions, a viewpoint shift that is
reported, not corrected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# nuPlan CAM_F0 (test-log metadata; OpenCV order k1, k2, p1, p2, k3).
NUPLAN_CAM_F0_K = np.array([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])
NUPLAN_CAM_F0_D = np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231])
NUPLAN_CAM_F0_SIZE = (1920, 1080)  # (width, height)


@dataclass(frozen=True)
class Pinhole:
    K: np.ndarray            # 3x3
    D: np.ndarray            # OpenCV plumb_bob (k1, k2, p1, p2, k3); zeros for a rectified image
    size: tuple              # (width, height)


def nuplan_cam_f0() -> Pinhole:
    return Pinhole(NUPLAN_CAM_F0_K.copy(), NUPLAN_CAM_F0_D.copy(), NUPLAN_CAM_F0_SIZE)


def rotation_from_ypr_deg(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Rotation taking a ray in the virtual camera frame to the source camera frame.

    Camera frame: x right, y down, z forward. yaw > 0 turns the virtual view to the
    right of the source optical axis, pitch > 0 tilts it down, roll about z.
    """
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])      # about camera y (down)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])      # about camera x (right)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def _undistort_to_rays(cam: Pinhole, uv: np.ndarray) -> np.ndarray:
    """Pixels (N,2) of a (possibly distorted) camera -> unit-z rays (N,3)."""
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
    xy = cv2.undistortPointsIter(uv.reshape(-1, 1, 2).astype(np.float64), cam.K, cam.D,
                                 None, None, crit).reshape(-1, 2)
    return np.concatenate([xy, np.ones((xy.shape[0], 1))], axis=1)


def horizontal_fov_deg(cam: Pinhole) -> float:
    """True horizontal coverage along the principal row, distortion included."""
    w = cam.size[0]
    r = _undistort_to_rays(cam, np.array([[0.0, cam.K[1, 2]], [w - 1.0, cam.K[1, 2]]]))
    return math.degrees(math.atan(-r[0, 0]) + math.atan(r[1, 0]))


def build_remap(src: Pinhole, dst: Pinhole, R_src_from_dst: np.ndarray | None = None):
    """Maps for cv2.remap producing a dst-geometry image from a src image.

    Returns (map_x, map_y, valid_fraction). Pixels whose ray falls outside the source
    image (or behind it) map to -1 and render black; valid_fraction reports how much
    of the virtual image the source actually covers.
    """
    R = np.eye(3) if R_src_from_dst is None else np.asarray(R_src_from_dst, float)
    w, h = dst.size
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays = _undistort_to_rays(dst, np.stack([uu.ravel(), vv.ravel()], axis=1))
    rs = rays @ R.T
    front = rs[:, 2] > 1e-6
    pts = np.full((rs.shape[0], 2), -1.0)
    if front.any():
        proj, _ = cv2.projectPoints(rs[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), src.K, src.D)
        pts[front] = proj.reshape(-1, 2)
    sw, sh = src.size
    inside = front & (pts[:, 0] >= 0) & (pts[:, 0] <= sw - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= sh - 1)
    pts[~inside] = -1.0
    map_x = pts[:, 0].reshape(h, w).astype(np.float32)
    map_y = pts[:, 1].reshape(h, w).astype(np.float32)
    return map_x, map_y, float(inside.mean())


def render(src_img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(src_img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---- orientation from extrinsics -------------------------------------------------
# Vehicle base frame: x forward, y left, z up. Camera frame: x right, y down, z forward.
# R_base_cam has the camera axes as columns, expressed in the base frame (the rotation
# block of a camera-to-base transform, as in the rig_dji_final_*.json files).

# nuPlan CAM_F0 orientation relative to the ego frame: median over NAVSIM test logs
# (sensor2lidar_rotation, lidar2ego = identity): pitch -1.45 deg (down), yaw 0 deg,
# roll -0.5 deg; mounted 1.52 m high.
NUPLAN_CAM_F0_PITCH_DEG = -1.45
NUPLAN_CAM_F0_YAW_DEG = 0.0
NUPLAN_CAM_F0_ROLL_DEG = -0.5


def r_base_cam(pitch_deg: float, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Camera looking along base x, then yawed left (+), pitched up (+), rolled (+ = clockwise seen from behind)."""
    level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # columns: x_cam, y_cam, z_cam in base
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])        # yaw about base z
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])        # pitch: optical axis up for +pitch
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    Rroll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])     # roll about the camera's own z
    return Rz @ Ry @ level @ Rroll


def optical_axis_angles(R_base_cam_: np.ndarray):
    """(pitch_up_deg, yaw_left_deg) of a camera's optical axis in the base frame."""
    z = R_base_cam_[:, 2]
    return math.degrees(math.asin(z[2])), math.degrees(math.atan2(z[1], z[0]))


def rotation_virtual_to_source(R_base_src: np.ndarray, R_base_virtual: np.ndarray) -> np.ndarray:
    """R such that a ray r in the virtual camera frame is R @ r in the source camera frame."""
    return np.asarray(R_base_src, float).T @ np.asarray(R_base_virtual, float)
