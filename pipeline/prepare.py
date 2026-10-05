"""Decode each video exactly once into a reusable cache.

Five of the nine extractors need decoded frames and two need a mono waveform.
Decoding per extractor would mean demuxing each video seven times. Instead a
single `prepare` step writes a frame cache (JPEG at FRAME_HZ) and a 16 kHz mono
WAV, keyed by the video's content hash, and every extractor reads from that.

Frame-time convention, used everywhere downstream:
    frame_index i (1-based)  ->  t_start_s = (i - 1) / FRAME_HZ
                                 t_end_s   = i / FRAME_HZ
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .io import CreativeRef, ffprobe

_CACHE_VERSION = 1


@dataclass
class Prepared:
    """Everything an extractor needs, decoded once."""
    ref: CreativeRef
    probe: dict
    frames_dir: Path
    frame_paths: list[Path]
    frame_times: np.ndarray       # t_start_s per frame, shape (n_frames,)
    audio_path: Path | None
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool

    @property
    def n_frames(self) -> int:
        return len(self.frame_paths)

    def sample(self, hz: float) -> list[tuple[int, float, Path]]:
        """Subsample the frame cache down to `hz`, shared by every extractor.

        Returns (frame_index, t_start_s, path) so a row written by a 1 Hz
        extractor still joins cleanly onto the 4 Hz spine.
        """
        step = max(1, round(config.FRAME_HZ / hz))
        return [
            (i + 1, float(self.frame_times[i]), self.frame_paths[i])
            for i in range(0, self.n_frames, step)
        ]


def _video_stream(probe: dict) -> dict:
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    raise ValueError("no video stream found")


def _audio_stream(probe: dict) -> dict | None:
    for s in probe.get("streams", []):
        if s.get("codec_type") == "audio":
            return s
    return None


def _parse_fps(stream: dict) -> float:
    raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    num, _, den = raw.partition("/")
    try:
        den_f = float(den or 1)
        return float(num) / den_f if den_f else 0.0
    except ValueError:
        return 0.0


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{proc.stderr[-2000:]}")


def prepare(ref: CreativeRef, force: bool = False) -> Prepared:
    """Build (or reuse) the decode cache for one creative."""
    probe = ffprobe(ref.path)
    vstream = _video_stream(probe)
    astream = _audio_stream(probe)

    duration = float(probe.get("format", {}).get("duration") or 0.0)
    width, height = int(vstream.get("width", 0)), int(vstream.get("height", 0))
    fps = _parse_fps(vstream)

    # Content-addressed cache: different bytes or different sampling params
    # produce a different cache, so stale frames can't leak into a run.
    cache_root = config.CACHE_DIR / ref.sha256[:16]
    frames_dir = cache_root / "frames"
    audio_path = cache_root / "audio.wav"
    stamp_path = cache_root / "cache.json"
    stamp = {
        "cache_version": _CACHE_VERSION,
        "frame_hz": config.FRAME_HZ,
        "frame_height": config.FRAME_HEIGHT,
        "audio_sr": config.AUDIO_SR,
        "sha256": ref.sha256,
    }

    valid = (stamp_path.exists()
             and json.loads(stamp_path.read_text()) == stamp
             and frames_dir.exists()
             and any(frames_dir.iterdir()))

    if force or not valid:
        if cache_root.exists():
            shutil.rmtree(cache_root)
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Never upscale: min(target, ih). -2 keeps width even for the encoder.
        _run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(ref.path),
            "-vf", f"fps={config.FRAME_HZ},scale=-2:'min({config.FRAME_HEIGHT},ih)'",
            "-q:v", "3", str(frames_dir / "%06d.jpg"),
        ])

        if astream is not None:
            _run([
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(ref.path),
                "-vn", "-ac", "1", "-ar", str(config.AUDIO_SR),
                "-c:a", "pcm_s16le", str(audio_path),
            ])

        stamp_path.write_text(json.dumps(stamp, indent=2))

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    frame_times = np.arange(len(frame_paths), dtype=float) / config.FRAME_HZ

    return Prepared(
        ref=ref,
        probe=probe,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        frame_times=frame_times,
        audio_path=audio_path if audio_path.exists() else None,
        duration_s=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=astream is not None,
    )
