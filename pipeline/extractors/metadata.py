"""Feature 1 - container, stream and codec metadata (ffprobe)."""
from __future__ import annotations

import pandas as pd

from .. import config
from ..prepare import Prepared, _audio_stream, _parse_fps, _video_stream

NAME = "metadata"
VERSION = "1.0"
FEATURE = "Container, codec, resolution, frame-rate and audio-stream metadata"
TOOL = "ffprobe (FFmpeg)"
GRAIN = "one row per creative; one row per sampled extractor"
REQUIRES: tuple[str, ...] = ()


def _orientation(width: int, height: int) -> str:
    if not width or not height:
        return "unknown"
    ratio = width / height
    if ratio < 0.95:
        return "vertical"
    if ratio > 1.05:
        return "horizontal"
    return "square"


def extract(p: Prepared) -> dict[str, pd.DataFrame]:
    v = _video_stream(p.probe)
    a = _audio_stream(p.probe)
    fmt = p.probe.get("format", {})

    row = {
        "title": p.ref.title,
        "source_label": p.ref.source_label,
        "file_name": p.ref.file_name,
        "file_sha256": p.ref.sha256,
        "file_size_bytes": p.ref.size_bytes,
        "container": fmt.get("format_name"),
        "duration_s": round(p.duration_s, 3),
        "width": p.width,
        "height": p.height,
        "aspect_ratio": round(p.width / p.height, 4) if p.height else None,
        "orientation": _orientation(p.width, p.height),
        "fps": round(p.fps, 4),
        "n_video_frames": int(v.get("nb_frames") or 0) or None,
        "video_codec": v.get("codec_name"),
        "pix_fmt": v.get("pix_fmt"),
        "video_bitrate_kbps": (int(v["bit_rate"]) // 1000) if v.get("bit_rate") else None,
        "total_bitrate_kbps": (int(fmt["bit_rate"]) // 1000) if fmt.get("bit_rate") else None,
        "has_audio": a is not None,
        "audio_codec": a.get("codec_name") if a else None,
        "audio_sample_rate": int(a["sample_rate"]) if a and a.get("sample_rate") else None,
        "audio_channels": int(a["channels"]) if a and a.get("channels") else None,
        # Sampling actually applied, so a row can always be interpreted later.
        "sampled_frame_hz": config.FRAME_HZ,
        "n_sampled_frames": p.n_frames,
    }
    # Published alongside the metadata so the warehouse is self-describing:
    # downstream SQL uses step_frames to distinguish "this frame was never
    # looked at by that model" from "it was looked at and found nothing".
    sampling = pd.DataFrame([
        {"extractor": "visual", "sample_hz": config.VISUAL_HZ},
        {"extractor": "audio", "sample_hz": config.FRAME_HZ},
        {"extractor": "faces", "sample_hz": config.FACE_HZ},
        {"extractor": "objects", "sample_hz": config.OBJECT_HZ},
        {"extractor": "ocr", "sample_hz": config.OCR_HZ},
    ])
    sampling["frame_hz"] = config.FRAME_HZ
    sampling["step_frames"] = (config.FRAME_HZ / sampling["sample_hz"]).round().astype(int)

    return {
        "creative_metadata": pd.DataFrame([row]),
        "sampling_config": sampling,
    }
