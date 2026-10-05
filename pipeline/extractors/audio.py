"""Feature 5 - loudness, tempo and the music-vs-speech signal (librosa).

A note on "music vs speech", because it is the one feature on the list that
cannot be read straight off a model: there is no free, licence-clean
music/speech classifier in this stack. Rather than ship a black box, this
extractor emits the *evidence* -- harmonic/percussive balance, spectral
flatness, zero-crossing rate, loudness stability -- and the transform layer
combines it with Whisper's VAD-backed speech spans to label each second
(`audio_class` in `creative_seconds`). Speech detection is therefore
model-grounded; music vs ambient is an explicit, inspectable heuristic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..prepare import Prepared

NAME = "audio"
VERSION = "1.0"
FEATURE = "Loudness (dBFS), tempo, spectral shape and harmonic/percussive balance"
TOOL = "librosa"
GRAIN = f"one row per sampled frame ({config.FRAME_HZ} Hz); one summary row per creative"
REQUIRES: tuple[str, ...] = ()

_SILENCE_DBFS = -50.0

_FRAME_COLS = ["frame_index", "t_start_s", "t_end_s", "rms", "dbfs",
               "spectral_centroid_hz", "spectral_flatness",
               "spectral_rolloff_hz", "zero_crossing_rate", "onset_strength",
               "harmonic_ratio", "is_silent"]
_SUMMARY_COLS = ["tempo_bpm", "n_beats", "mean_dbfs", "peak_dbfs",
                 "dynamic_range_db", "silence_share", "mean_harmonic_ratio",
                 "sample_rate"]


def _empty() -> dict[str, pd.DataFrame]:
    return {
        "audio_frames": pd.DataFrame(columns=_FRAME_COLS),
        "audio_summary": pd.DataFrame([{c: None for c in _SUMMARY_COLS}]),
    }


def extract(p: Prepared) -> dict[str, pd.DataFrame]:
    if p.audio_path is None:
        return _empty()

    import librosa

    y, sr = librosa.load(str(p.audio_path), sr=config.AUDIO_SR, mono=True)
    if y.size == 0:
        return _empty()

    # hop == one sampling period, so librosa frame j lands exactly on video
    # frame j+1. No interpolation anywhere downstream.
    hop = int(round(sr / config.FRAME_HZ))
    n_fft = 4096

    def feat(fn, **kw) -> np.ndarray:
        return fn(y=y, sr=sr, hop_length=hop, n_fft=n_fft, **kw).reshape(-1)

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).reshape(-1)
    centroid = feat(librosa.feature.spectral_centroid)
    rolloff = feat(librosa.feature.spectral_rolloff)
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop).reshape(-1)
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=n_fft, hop_length=hop).reshape(-1)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop).reshape(-1)

    # Harmonic/percussive separation: tonal content (music beds, sung vocals)
    # sits in the harmonic part, transients (drums, consonants) in the percussive.
    y_harm, y_perc = librosa.effects.hpss(y)
    rms_h = librosa.feature.rms(y=y_harm, frame_length=n_fft, hop_length=hop).reshape(-1)
    rms_p = librosa.feature.rms(y=y_perc, frame_length=n_fft, hop_length=hop).reshape(-1)
    harmonic_ratio = rms_h / np.maximum(rms_h + rms_p, 1e-9)

    n = min(len(rms), len(centroid), len(rolloff), len(flatness),
            len(zcr), len(onset), len(harmonic_ratio))
    dbfs = 20.0 * np.log10(np.maximum(rms[:n], 1e-9))

    frames = pd.DataFrame({
        "frame_index": np.arange(1, n + 1),
        "t_start_s": np.round(np.arange(n) / config.FRAME_HZ, 3),
        "t_end_s": np.round((np.arange(n) + 1) / config.FRAME_HZ, 3),
        "rms": np.round(rms[:n], 6),
        "dbfs": np.round(dbfs, 3),
        "spectral_centroid_hz": np.round(centroid[:n], 2),
        "spectral_flatness": np.round(flatness[:n], 6),
        "spectral_rolloff_hz": np.round(rolloff[:n], 2),
        "zero_crossing_rate": np.round(zcr[:n], 5),
        "onset_strength": np.round(onset[:n], 4),
        "harmonic_ratio": np.round(harmonic_ratio[:n], 5),
        "is_silent": dbfs < _SILENCE_DBFS,
    })[_FRAME_COLS]

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    audible = dbfs[dbfs >= _SILENCE_DBFS]
    summary = pd.DataFrame([{
        "tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 2),
        "n_beats": int(len(beats)),
        "mean_dbfs": round(float(audible.mean()), 3) if audible.size else None,
        "peak_dbfs": round(float(dbfs.max()), 3),
        "dynamic_range_db": round(float(audible.max() - audible.min()), 3) if audible.size else None,
        "silence_share": round(float((dbfs < _SILENCE_DBFS).mean()), 4),
        "mean_harmonic_ratio": round(float(harmonic_ratio[:n].mean()), 5),
        "sample_rate": sr,
    }], columns=_SUMMARY_COLS)

    return {"audio_frames": frames, "audio_summary": summary}
