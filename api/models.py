"""Request and response shapes. These are the API's contract."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from pipeline import config as creative_config

EMOTIONS = set(creative_config.EMOTION_LABELS)


# ------------------------------------------------------------------ ingest --
class CreativeFromURL(BaseModel):
    """Register a video by URL.

    URL rather than multipart upload on purpose: creatives already live in
    object storage or a CDN in every setup worth integrating with, the request
    stays small, and the service needs no upload handling or temp-file cleanup.
    """
    url: str
    title: str | None = Field(None, description="Overrides the filename-derived title")

    @field_validator("url")
    @classmethod
    def _http_only(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must be http or https")
        return v


class CreativeAccepted(BaseModel):
    creative_id: str
    title: str
    sha256: str
    size_bytes: int
    path: str
    next: str


class BiometricRow(BaseModel):
    """One viewer-second. Mirrors `audience.biometric_seconds`."""
    second: int = Field(ge=0)
    attention_index: float | None = Field(None, ge=0, le=1)
    gaze_on_screen: bool | None = None
    face_detected: bool | None = None
    valence: float | None = Field(None, ge=-1, le=1)
    arousal: float | None = Field(None, ge=0, le=1)
    # Flattened to p_<label> columns on write. A dict keeps the wire format
    # small and lets a client send only the classes its model produces.
    emotions: dict[str, float] | None = None

    @field_validator("emotions")
    @classmethod
    def _known_labels(cls, v):
        if v and (unknown := set(v) - EMOTIONS):
            raise ValueError(f"unknown emotion labels {sorted(unknown)}; "
                             f"expected a subset of {sorted(EMOTIONS)}")
        return v


class BiometricBatch(BaseModel):
    session_id: str
    viewer_id: str
    creative_id: str
    rows: list[BiometricRow] = Field(min_length=1, max_length=10_000)


class BatchAccepted(BaseModel):
    batch_id: str
    rows_accepted: int
    landed_at: str
    note: str


# -------------------------------------------------------------- processing --
class JobRequest(BaseModel):
    """Process a video. Either a URL to fetch, or a creative already on disk."""
    url: str | None = None
    creative_id: str | None = None
    extractors: list[str] | None = Field(
        None, description="Subset of extractor names; dependencies are pulled in. "
                          "Omit to run all nine.")
    force: bool = False

    @field_validator("url")
    @classmethod
    def _http_only(cls, v):
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("url must be http or https")
        return v


class Job(BaseModel):
    job_id: str
    status: str                      # queued | running | succeeded | failed
    creative_id: str | None = None
    extractors: list[str] = []
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    results: list[dict] = []         # one entry per extractor, from the manifest
    output_dir: str | None = None
    error: str | None = None
