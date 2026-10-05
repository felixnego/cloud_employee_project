"""Feature 9 - free-text semantic labels per shot (CLIP, zero-shot).

This is the layer that makes the dataset readable by someone who has never
watched the ads: "shot 7 is a product close-up in a studio, shot 8 is a crowd
outdoors". It is also the one extractor with a declared dependency -- it
labels *shots*, so `shots` must have run first.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .. import config
from ..models.clip import ZeroShotLabeller
from ..prepare import Prepared

NAME = "labels"
VERSION = "1.1"
FEATURE = "Zero-shot semantic labels per shot across four axes (shot scale, setting, subject, style)"
TOOL = "CLIP ViT-B/32 (int8 ONNX), zero-shot"
GRAIN = "one row per (shot, axis, label)"
REQUIRES: tuple[str, ...] = ("shots",)

_COLS = ["shot_index", "axis", "label", "score", "rank", "is_top", "n_frames_scored"]

_labeller = None


def _get_labeller() -> ZeroShotLabeller:
    global _labeller
    if _labeller is None:
        _labeller = ZeroShotLabeller()
    return _labeller


def _shot_frames(p: Prepared, start_s: float, end_s: float) -> list[np.ndarray]:
    """Up to CLIP_FRAMES_PER_SHOT evenly spaced frames from inside a shot."""
    n = config.CLIP_FRAMES_PER_SHOT
    # Sample at the 1/(n+1) quantiles so we avoid the frames straddling a cut.
    targets = [start_s + (end_s - start_s) * (k + 1) / (n + 1) for k in range(n)]
    indices = sorted({
        int(np.clip(round(t * config.FRAME_HZ), 0, p.n_frames - 1))
        for t in targets
    })
    frames = []
    for idx in indices:
        img = cv2.imread(str(p.frame_paths[idx]), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def extract(p: Prepared) -> dict[str, pd.DataFrame]:
    shots_path = config.RAW_DIR / "shot_boundaries" / f"{p.ref.creative_id}.parquet"
    if not shots_path.exists():
        raise FileNotFoundError(
            f"'{NAME}' requires the 'shots' extractor to have run first "
            f"(missing {shots_path})"
        )
    shots = pd.read_parquet(shots_path)
    if p.n_frames == 0:
        return {"shot_label_scores": pd.DataFrame(columns=_COLS)}

    labeller = _get_labeller()
    rows: list[dict] = []

    for shot in shots.itertuples():
        frames = _shot_frames(p, float(shot.start_s), float(shot.end_s))
        if not frames:
            continue
        for axis, probs in labeller.score_axes(frames).items():
            names = [label for label, _ in labeller.axes[axis]]
            order = np.argsort(-probs)
            for rank, pos in enumerate(order):
                rows.append({
                    "shot_index": int(shot.shot_index),
                    "axis": axis,
                    "label": names[pos],
                    "score": round(float(probs[pos]), 5),
                    "rank": rank,
                    "is_top": rank == 0,
                    "n_frames_scored": len(frames),
                })

    return {"shot_label_scores": pd.DataFrame(rows, columns=_COLS)}
