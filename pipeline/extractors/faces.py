"""Feature 7 - faces on screen and their expressions (YuNet + FER+)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .. import config
from ..models.faces import ExpressionClassifier, FaceDetector
from ..prepare import Prepared

NAME = "faces"
VERSION = "1.0"
FEATURE = "Face count, face size/position on screen, and an 8-class facial expression per face"
TOOL = "YuNet (detection) + FER+ (expression), both ONNX"
GRAIN = f"one row per detected face per sampled frame ({config.FACE_HZ} Hz)"
REQUIRES: tuple[str, ...] = ()

_CROP_PAD = 0.15        # FER+ was trained on loosely cropped faces
_MIN_CROP_PX = 24

_COLS = (["frame_index", "t_start_s", "face_index", "score",
          "x0", "y0", "x1", "y1", "area_frac", "centre_x", "centre_y",
          "is_largest", "emotion_top", "emotion_top_prob"]
         + [f"p_{label}" for label in config.EMOTION_LABELS])

_detector = None
_classifier = None


def _models() -> tuple[FaceDetector, ExpressionClassifier]:
    global _detector, _classifier
    if _detector is None:
        _detector = FaceDetector()
        _classifier = ExpressionClassifier()
    return _detector, _classifier


def extract(p: Prepared) -> dict[str, pd.DataFrame]:
    detector, classifier = _models()
    labels = config.EMOTION_LABELS
    rows: list[dict] = []

    for frame_index, t_start, path in p.sample(config.FACE_HZ):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        faces = detector.detect(img)
        if len(faces) == 0:
            continue

        areas = [float(f[2] * f[3]) for f in faces]
        largest = int(np.argmax(areas))

        for fi, face in enumerate(faces):
            x, y, bw, bh = (float(v) for v in face[:4])
            score = float(face[14])

            # Pad, then clamp to the frame before cropping.
            px, py = bw * _CROP_PAD, bh * _CROP_PAD
            cx0 = int(max(0, x - px)); cy0 = int(max(0, y - py))
            cx1 = int(min(w, x + bw + px)); cy1 = int(min(h, y + bh + py))
            crop = img[cy0:cy1, cx0:cx1]

            if crop.size and min(crop.shape[:2]) >= _MIN_CROP_PX:
                probs = classifier.predict(crop)
                top = int(np.argmax(probs))
                emotion_top, top_prob = labels[top], float(probs[top])
            else:
                # Face too small to classify reliably -- recorded as present
                # but unlabelled rather than guessed at.
                probs = np.full(len(labels), np.nan)
                emotion_top, top_prob = None, None

            row = {
                "frame_index": frame_index,
                "t_start_s": round(t_start, 3),
                "face_index": fi,
                "score": round(score, 4),
                "x0": round(max(x, 0) / w, 5),
                "y0": round(max(y, 0) / h, 5),
                "x1": round(min(x + bw, w) / w, 5),
                "y1": round(min(y + bh, h) / h, 5),
                "area_frac": round(min(bw * bh / (w * h), 1.0), 6),
                "centre_x": round((x + bw / 2) / w, 5),
                "centre_y": round((y + bh / 2) / h, 5),
                "is_largest": fi == largest,
                "emotion_top": emotion_top,
                "emotion_top_prob": round(top_prob, 4) if top_prob is not None else None,
            }
            row.update({f"p_{lab}": (None if np.isnan(pr) else round(float(pr), 4))
                        for lab, pr in zip(labels, probs)})
            rows.append(row)

    return {"face_detections": pd.DataFrame(rows, columns=_COLS)}
