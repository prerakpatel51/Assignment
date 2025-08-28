# calibration_utils.py
# Core helpers for OpenCV camera calibration with clean, stateless design.

from __future__ import annotations
import os, glob, json, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# ---------- IO helpers ----------
class IO:
    @staticmethod
    def ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def list_images(dir_path: str, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")) -> List[str]:
        files: List[str] = []
        for e in exts:
            files.extend(glob.glob(os.path.join(dir_path, f"*{e}")))
        return sorted(files)

    @staticmethod
    def imread_rgb(path: str):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def imwrite_rgb(path: str, rgb: np.ndarray) -> None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)

    @staticmethod
    def save_json(obj: Dict[str, Any], path: str) -> None:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)


# ---------- Board model ----------
class Board:
    """
    Chessboard object model.
    pattern_size = (cols, rows) are INNER corners e.g. (9,6)
    square_size = side length in meters e.g. 0.025
    """
    @staticmethod
    def object_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
        cols, rows = pattern_size
        objp = np.zeros((rows * cols, 3), np.float32)
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp[:, :2] = grid * square_size
        return objp


# ---------- Image helpers ----------
class Img:
    @staticmethod
    def to_gray(rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def find_corners(gray: np.ndarray, pattern_size: Tuple[int, int]):
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not ok:
            return False, None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        return True, corners_refined


# ---------- Calibration pipeline ----------
class Calib:
    @staticmethod
    def calibrate(image_paths: List[str], pattern_size=(9,6), square_size=0.025) -> Dict[str, Any]:
        """
        Returns a dict with K, D, extrinsics (R,t), per-view errors, etc.
        """
        obj_pts: List[np.ndarray] = []
        img_pts: List[np.ndarray] = []
        imsize = None
        objp = Board.object_points(pattern_size, square_size)

        valid_paths: List[str] = []
        for p in image_paths:
            rgb = IO.imread_rgb(p)
            if rgb is None:
                continue
            gray = Img.to_gray(rgb)
            if imsize is None:
                imsize = (gray.shape[1], gray.shape[0])
            ok, corners = Img.find_corners(gray, pattern_size)
            if ok:
                obj_pts.append(objp.copy())
                img_pts.append(corners)
                valid_paths.append(p)

        if imsize is None or len(obj_pts) < 5:
            raise ValueError("Not enough valid detections (need at least ~5).")

        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            imageSize=imsize,
            cameraMatrix=None,
            distCoeffs=None,
            flags=0
        )

        per_view_errors = []
        for i in range(len(obj_pts)):
            proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, D)
            err = cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj)
            per_view_errors.append(float(err))
        mean_err = float(np.mean(per_view_errors))

        # Convert extrinsics to R,t
        extrinsics = []
        for rv, tv in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rv)
            extrinsics.append({"R": R.tolist(), "t": tv.flatten().tolist()})

        result = {
            "ret_rms": float(ret),
            "mean_reproj_error": mean_err,
            "image_size": {"width": imsize[0], "height": imsize[1]},
            "K": np.asarray(K).tolist(),
            "D": np.asarray(D).flatten().tolist(),
            "pattern_size": {"cols": pattern_size[0], "rows": pattern_size[1]},
            "square_size_m": float(square_size),
            "num_views": len(valid_paths),
            "per_view_errors": per_view_errors,
            "valid_paths": valid_paths,
            "extrinsics": extrinsics
        }
        return result

    @staticmethod
    def undistort(rgb: np.ndarray, K, D, keep_size=True):
        h, w = rgb.shape[:2]
        K = np.array(K, dtype=np.float64)
        D = np.array(D, dtype=np.float64)
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
        und = cv2.undistort(rgb, K, D, None, newK)
        if keep_size:
            und = cv2.resize(und, (w, h), interpolation=cv2.INTER_AREA)
        return und


# ---------- Overlay helpers ----------
class Overlay:
    @staticmethod
    def draw_axes(rgb: np.ndarray, K, D, rvec, tvec, axis_len=0.05):
        pts3d = np.float32([[0,0,0],[axis_len,0,0],[0,axis_len,0],[0,0,axis_len]]).reshape(-1,3)
        pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, D)
        pts2d = pts2d.reshape(-1,2).astype(int)

        img = rgb.copy()
        origin = tuple(pts2d[0]); xpt = tuple(pts2d[1]); ypt = tuple(pts2d[2]); zpt = tuple(pts2d[3])
        # OpenCV uses BGR; our rgb is RGB, but drawing with RGB colors is fine for visualization
        cv2.line(img, origin, xpt, (255,0,0), 3)   # X red
        cv2.line(img, origin, ypt, (0,255,0), 3)   # Y green
        cv2.line(img, origin, zpt, (0,0,255), 3)   # Z blue
        return img

    @staticmethod
    def make_sample_overlays(image_paths: List[str], K, D, pattern_size, square_size, max_images=8):
        K = np.array(K, np.float64)
        D = np.array(D, np.float64)
        cols, rows = pattern_size
        objp = Board.object_points(pattern_size, square_size)

        chosen = image_paths[:max_images]
        out = []
        for p in chosen:
            rgb = IO.imread_rgb(p)
            if rgb is None: 
                continue
            gray = Img.to_gray(rgb)
            ok, corners = Img.find_corners(gray, pattern_size)
            if not ok:
                continue
            ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            with_axes = Overlay.draw_axes(rgb, K, D, rvec, tvec, axis_len=square_size*3)
            out.append((p, with_axes))
        return out


# ---------- 3D camera pose visualization ----------
class Render:
    @staticmethod
    def camera_centers_and_axes(extrinsics: List[Dict[str, Any]]):
        centers = []
        z_axes = []
        for ex in extrinsics:
            R = np.array(ex["R"], dtype=np.float64)
            t = np.array(ex["t"], dtype=np.float64).reshape(3,1)
            C = (-R.T @ t).flatten()                  # camera center in world
            z_axis = (R.T @ np.array([0,0,1.0])).flatten()
            centers.append(C); z_axes.append(z_axis)
        return np.array(centers), np.array(z_axes)

    @staticmethod
    def plot_poses_plotly(extrinsics: List[Dict[str, Any]], square_size=0.025, board_size=(9,6)):
        centers, z_axes = Render.camera_centers_and_axes(extrinsics)
        cols, rows = board_size
        w = (cols-1)*square_size; h = (rows-1)*square_size

        bx = [0, w, w, 0, 0]; by = [0, 0, h, h, 0]; bz = [0, 0, 0, 0, 0]

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode='lines', name='Board'))
        if len(centers) > 0:
            fig.add_trace(go.Scatter3d(x=centers[:,0], y=centers[:,1], z=centers[:,2],
                                       mode='markers', name='Cameras', marker=dict(size=5)))
            scale = square_size*3
            for c, z in zip(centers, z_axes):
                p2 = c + z*scale
                fig.add_trace(go.Scatter3d(x=[c[0], p2[0]], y=[c[1], p2[1]], z=[c[2], p2[2]],
                                           mode='lines', line=dict(width=4), name='optical_axis'))
        fig.update_layout(width=800, height=500,
                          scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                                     aspectmode="data"),
                          title="Estimated Camera Poses w.r.t. Chessboard")
        return fig

    @staticmethod
    def plot_poses_matplotlib(extrinsics: List[Dict[str, Any]], square_size=0.025, board_size=(9,6)):
        centers, z_axes = Render.camera_centers_and_axes(extrinsics)
        cols, rows = board_size
        w = (cols-1)*square_size; h = (rows-1)*square_size

        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111, projection='3d')
        bx = [0, w, w, 0, 0]; by = [0, 0, h, h, 0]; bz = [0,0,0,0,0]
        ax.plot(bx, by, bz, lw=2, label="Board")
        if len(centers) > 0:
            ax.scatter(centers[:,0], centers[:,1], centers[:,2], s=10, label="Cameras")
            scale = square_size*3
            for c, z in zip(centers, z_axes):
                p2 = c + z*scale
                ax.plot([c[0], p2[0]], [c[1], p2[1]], [c[2], p2[2]], lw=2)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.legend(); ax.set_title("Estimated Camera Poses w.r.t. Chessboard")
        plt.tight_layout()
        return fig
