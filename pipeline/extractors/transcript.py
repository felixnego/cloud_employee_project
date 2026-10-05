"""Feature 3 - word-level transcript with timestamps (faster-whisper)."""
from __future__ import annotations

import pandas as pd

from .. import config
from ..prepare import Prepared

NAME = "transcript"
VERSION = "1.0"
FEATURE = "Spoken-word transcript with segment and word-level timestamps"
TOOL = f"faster-whisper '{config.WHISPER_MODEL}' (CTranslate2, int8 CPU)"
GRAIN = "one row per utterance; one row per word"
REQUIRES: tuple[str, ...] = ()

_model = None


def _get_model():
    """Loaded once per process -- it is ~250 MB and five creatives share it."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            config.WHISPER_MODEL, device="cpu",
            compute_type=config.WHISPER_COMPUTE,
            download_root=str(config.WHISPER_CACHE),
        )
    return _model


_SEGMENT_COLS = ["segment_index", "start_s", "end_s", "text", "n_words",
                 "words_per_s", "avg_logprob", "no_speech_prob",
                 "compression_ratio"]
_WORD_COLS = ["segment_index", "word_index", "word", "start_s", "end_s",
              "probability"]
_SUMMARY_COLS = ["language", "language_probability", "model", "n_segments",
                 "n_words", "speech_s", "speech_share", "words_per_minute"]


def extract(p: Prepared) -> dict[str, pd.DataFrame]:
    if p.audio_path is None:
        return {
            "transcript_segments": pd.DataFrame(columns=_SEGMENT_COLS),
            "transcript_words": pd.DataFrame(columns=_WORD_COLS),
            "transcript_summary": pd.DataFrame([{
                "language": None, "language_probability": None,
                "model": config.WHISPER_MODEL, "n_segments": 0, "n_words": 0,
                "speech_s": 0.0, "speech_share": 0.0, "words_per_minute": 0.0,
            }]),
        }

    # vad_filter uses the Silero VAD bundled with faster-whisper, which keeps
    # the model from hallucinating lyrics over instrumental music beds.
    segments, info = _get_model().transcribe(
        str(p.audio_path), beam_size=5, word_timestamps=True, vad_filter=True,
    )

    seg_rows, word_rows = [], []
    for si, seg in enumerate(segments):          # generator: consumes lazily
        words = list(seg.words or [])
        span = max(seg.end - seg.start, 1e-6)
        seg_rows.append({
            "segment_index": si,
            "start_s": round(seg.start, 3),
            "end_s": round(seg.end, 3),
            "text": (seg.text or "").strip(),
            "n_words": len(words),
            "words_per_s": round(len(words) / span, 3),
            "avg_logprob": round(seg.avg_logprob, 4) if seg.avg_logprob is not None else None,
            "no_speech_prob": round(seg.no_speech_prob, 4) if seg.no_speech_prob is not None else None,
            "compression_ratio": round(seg.compression_ratio, 4) if seg.compression_ratio is not None else None,
        })
        for wi, w in enumerate(words):
            word_rows.append({
                "segment_index": si,
                "word_index": wi,
                "word": (w.word or "").strip(),
                "start_s": round(w.start, 3) if w.start is not None else None,
                "end_s": round(w.end, 3) if w.end is not None else None,
                "probability": round(w.probability, 4) if w.probability is not None else None,
            })

    speech_s = sum(r["end_s"] - r["start_s"] for r in seg_rows)
    summary = {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "model": config.WHISPER_MODEL,
        "n_segments": len(seg_rows),
        "n_words": len(word_rows),
        "speech_s": round(speech_s, 3),
        "speech_share": round(speech_s / p.duration_s, 4) if p.duration_s else None,
        "words_per_minute": round(60 * len(word_rows) / p.duration_s, 2) if p.duration_s else None,
    }

    return {
        "transcript_segments": pd.DataFrame(seg_rows, columns=_SEGMENT_COLS),
        "transcript_words": pd.DataFrame(word_rows, columns=_WORD_COLS),
        "transcript_summary": pd.DataFrame([summary], columns=_SUMMARY_COLS),
    }
