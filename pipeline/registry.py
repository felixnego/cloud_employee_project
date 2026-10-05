"""The ordered list of extractors -- the only place the nine features are enumerated.

Order matters only for declared dependencies (`REQUIRES`); everything else is
independent and could be parallelised. `validate_order` asserts the invariant
rather than trusting the author of the list.
"""
from __future__ import annotations

from types import ModuleType

from .extractors import (audio, faces, labels, metadata, objects, ocr, shots,
                         transcript, visual)

EXTRACTORS: list[ModuleType] = [
    metadata,     # 1  ffprobe
    shots,        # 2  PySceneDetect   (labels depends on this)
    transcript,   # 3  faster-whisper
    visual,       # 4  OpenCV
    audio,        # 5  librosa
    ocr,          # 6  RapidOCR
    faces,        # 7  YuNet + FER+
    objects,      # 8  YOLOX-nano
    labels,       # 9  CLIP zero-shot
]

BY_NAME: dict[str, ModuleType] = {m.NAME: m for m in EXTRACTORS}


def validate_order(mods: list[ModuleType]) -> None:
    seen: set[str] = set()
    for mod in mods:
        missing = [r for r in mod.REQUIRES if r not in seen]
        if missing:
            raise ValueError(
                f"extractor '{mod.NAME}' requires {missing}, which do not run before it"
            )
        seen.add(mod.NAME)


def resolve(only: list[str] | None = None) -> list[ModuleType]:
    """Select extractors by name, keeping registry order. Pulls in dependencies."""
    if not only:
        validate_order(EXTRACTORS)
        return list(EXTRACTORS)

    unknown = [n for n in only if n not in BY_NAME]
    if unknown:
        raise ValueError(f"unknown extractor(s) {unknown}; known: {list(BY_NAME)}")

    wanted = set(only)
    frontier = list(only)
    while frontier:                      # close over REQUIRES transitively
        for req in BY_NAME[frontier.pop()].REQUIRES:
            if req not in wanted:
                wanted.add(req)
                frontier.append(req)

    selected = [m for m in EXTRACTORS if m.NAME in wanted]
    validate_order(selected)
    return selected


def describe() -> list[dict]:
    """Registry as plain rows -- feeds `run list` and docs/schema.md."""
    return [{
        "n": i + 1,
        "name": m.NAME,
        "version": m.VERSION,
        "feature": m.FEATURE,
        "tool": m.TOOL,
        "grain": m.GRAIN,
        "requires": ", ".join(m.REQUIRES) or "-",
    } for i, m in enumerate(EXTRACTORS)]
