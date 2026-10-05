"""Feature 6 - on-screen text: taglines, offers, logos, captions (RapidOCR)."""
from __future__ import annotations

import pandas as pd
from PIL import Image

from .. import config
from ..prepare import Prepared

NAME = "ocr"
VERSION = "1.0"
FEATURE = "On-screen text detections (taglines, offers, captions, logotype)"
TOOL = "RapidOCR (PP-OCRv4 ONNX)"
GRAIN = f"one row per text detection per sampled frame ({config.OCR_HZ} Hz)"
REQUIRES: tuple[str, ...] = ()

_COLS = ["frame_index", "t_start_s", "detection_index", "text", "n_chars",
         "score", "x0", "y0", "x1", "y1", "area_frac", "height_frac",
         "centre_y"]

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def extract(p: Prepared) -> dict[str, pd.DataFrame]:
    engine = _get_engine()
    rows: list[dict] = []

    for frame_index, t_start, path in p.sample(config.OCR_HZ):
        with Image.open(path) as im:          # header-only read for dimensions
            width, height = im.size

        result, _ = engine(str(path))
        if not result:
            continue

        for di, det in enumerate(result):
            box, text, score = det[0], det[1], float(det[2])
            text = (text or "").strip()
            if not text or score < config.OCR_SCORE_THRESHOLD:
                continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x0, x1 = min(xs) / width, max(xs) / width
            y0, y1 = min(ys) / height, max(ys) / height
            rows.append({
                "frame_index": frame_index,
                "t_start_s": round(t_start, 3),
                "detection_index": di,
                "text": text,
                "n_chars": len(text),
                "score": round(score, 4),
                "x0": round(x0, 5), "y0": round(y0, 5),
                "x1": round(x1, 5), "y1": round(y1, 5),
                "area_frac": round(max(x1 - x0, 0) * max(y1 - y0, 0), 6),
                # Text height as a share of frame height separates a headline
                # from a legal disclaimer without anyone watching the ad.
                "height_frac": round(max(y1 - y0, 0), 5),
                "centre_y": round((y0 + y1) / 2, 5),
            })

    return {"screen_text_detections": pd.DataFrame(rows, columns=_COLS)}
