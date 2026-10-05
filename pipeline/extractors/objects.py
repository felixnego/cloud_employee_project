"""Feature 8 - objects and people on screen (YOLOX-nano, COCO-80)."""
from __future__ import annotations

import cv2
import pandas as pd

from .. import config
from ..models.yolox import ObjectDetector
from ..prepare import Prepared

NAME = "objects"
VERSION = "1.0"
FEATURE = "COCO-80 object and person detections with on-screen size and position"
TOOL = "YOLOX-nano (ONNX)"
GRAIN = f"one row per detection per sampled frame ({config.OBJECT_HZ} Hz)"
REQUIRES: tuple[str, ...] = ()

_COLS = ["frame_index", "t_start_s", "detection_index", "class_id",
         "class_name", "score", "x0", "y0", "x1", "y1", "area_frac"]

_detector = None


def _get_detector() -> ObjectDetector:
    global _detector
    if _detector is None:
        _detector = ObjectDetector()
    return _detector


def extract(p: Prepared) -> dict[str, pd.DataFrame]:
    detector = _get_detector()
    rows: list[dict] = []

    for frame_index, t_start, path in p.sample(config.OBJECT_HZ):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for di, det in enumerate(detector.detect(img)):
            rows.append({
                "frame_index": frame_index,
                "t_start_s": round(t_start, 3),
                "detection_index": di,
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "score": round(det["score"], 4),
                "x0": round(det["x0"], 5), "y0": round(det["y0"], 5),
                "x1": round(det["x1"], 5), "y1": round(det["y1"], 5),
                "area_frac": round(det["area_frac"], 6),
            })

    return {"object_detections": pd.DataFrame(rows, columns=_COLS)}
