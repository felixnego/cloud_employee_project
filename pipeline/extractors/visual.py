"""Feature 4 - per-frame brightness, colour and motion (OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .. import config
from ..prepare import Prepared

NAME = "visual"
VERSION = "1.0"
FEATURE = "Brightness, saturation, contrast, colourfulness, sharpness, edge density and frame-to-frame motion"
TOOL = "OpenCV"
GRAIN = f"one row per sampled frame ({config.VISUAL_HZ} Hz)"
REQUIRES: tuple[str, ...] = ()

_COLS = ["frame_index", "t_start_s", "t_end_s", "brightness", "brightness_std",
         "saturation", "contrast", "colourfulness", "sharpness", "edge_density",
         "dominant_hue_deg", "motion_abs_diff", "hist_distance", "is_near_black"]

# Stats are computed on a fixed-width thumbnail so that a 1080x1920 UGC clip
# and a 1920x1080 broadcast spot produce comparable numbers.
_STAT_WIDTH = 256


def _colourfulness(bgr: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric."""
    b, g, r = (c.astype(np.float32) for c in cv2.split(bgr))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def extract(p: Prepared) -> dict[str, pd.DataFrame]:
    rows: list[dict] = []
    prev_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None

    for frame_index, t_start, path in p.sample(config.VISUAL_HZ):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        small = cv2.resize(img, (_STAT_WIDTH, max(1, round(h * _STAT_WIDTH / w))),
                           interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hue, sat, val = cv2.split(hsv)

        # Motion: mean absolute luma change, plus a histogram-correlation
        # distance that spikes on hard cuts rather than on camera movement.
        if prev_gray is None:
            motion_abs = None
            hist_distance = None
        else:
            motion_abs = float(np.abs(gray.astype(np.int16)
                                      - prev_gray.astype(np.int16)).mean() / 255.0)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist)
            hist_distance = float(1.0 - cv2.compareHist(prev_hist, hist,
                                                        cv2.HISTCMP_CORREL))
        hist_now = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist_now, hist_now)

        rows.append({
            "frame_index": frame_index,
            "t_start_s": round(t_start, 3),
            "t_end_s": round(t_start + 1.0 / config.VISUAL_HZ, 3),
            "brightness": round(float(val.mean()) / 255.0, 5),
            "brightness_std": round(float(val.std()) / 255.0, 5),
            "saturation": round(float(sat.mean()) / 255.0, 5),
            "contrast": round(float(gray.std()) / 255.0, 5),
            "colourfulness": round(_colourfulness(small), 4),
            "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4),
            "edge_density": round(float((cv2.Canny(gray, 100, 200) > 0).mean()), 5),
            "dominant_hue_deg": round(float(np.median(hue)) * 2.0, 2),  # OpenCV hue is 0-179
            "motion_abs_diff": round(motion_abs, 5) if motion_abs is not None else None,
            "hist_distance": round(hist_distance, 5) if hist_distance is not None else None,
            "is_near_black": bool(val.mean() < 16),
        })
        prev_gray, prev_hist = gray, hist_now

    return {"visual_frames": pd.DataFrame(rows, columns=_COLS)}
