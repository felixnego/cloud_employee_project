"""Feature 2 - shot boundaries and shot lengths (PySceneDetect)."""
from __future__ import annotations

import pandas as pd
from scenedetect import ContentDetector, detect

from .. import config
from ..prepare import Prepared

NAME = "shots"
VERSION = "1.0"
FEATURE = "Shot boundaries, shot count and shot durations (pacing)"
TOOL = "PySceneDetect ContentDetector"
GRAIN = "one row per shot"
REQUIRES: tuple[str, ...] = ()


def extract(p: Prepared) -> dict[str, pd.DataFrame]:
    # Detection runs on the original file, not the 4 Hz cache: a cut between
    # two cached frames would otherwise be rounded to the sampling grid.
    scenes = detect(str(p.ref.path), ContentDetector(threshold=config.SCENE_THRESHOLD))

    if not scenes:
        # Single continuous take (common for UGC) -- still emit one shot so
        # downstream joins never produce NULL shot_index.
        rows = [{
            "shot_index": 0, "start_s": 0.0, "end_s": round(p.duration_s, 3),
            "duration_s": round(p.duration_s, 3), "start_frame": 0,
            "end_frame": int(p.duration_s * p.fps) if p.fps else None,
        }]
    else:
        rows = [{
            "shot_index": i,
            "start_s": round(start.get_seconds(), 3),
            "end_s": round(end.get_seconds(), 3),
            "duration_s": round(end.get_seconds() - start.get_seconds(), 3),
            "start_frame": int(start.get_frames()),
            "end_frame": int(end.get_frames()),
        } for i, (start, end) in enumerate(scenes)]

    return {"shot_boundaries": pd.DataFrame(rows)}
