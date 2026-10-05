"""Ingestion API — getting new data in.

Two things arrive from outside: new creatives, and new audience biometrics.

The rule that shapes this service: **it never writes to warehouse.duckdb.**
DuckDB is single-writer, so an API holding the write lock would deadlock the
pipeline. Instead:

  * a new creative is downloaded into VIDEO_DIR, where `discover_creatives()`
    already finds it — no registry table, no migration, nothing to keep in sync;
  * a biometrics batch is appended as Parquet to a landing zone, and
    `python -m api.load` folds it into the warehouse later.

Landing zone then batch load is how ingest works anyway, so the constraint cost
nothing and kept the single-writer property honest.

    uvicorn api.ingest:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from pipeline import config
from pipeline.io import CreativeRef, discover_creatives

from . import download
from .models import (BatchAccepted, BiometricBatch, CreativeAccepted,
                     CreativeFromURL)

LANDING_DIR = config.WAREHOUSE_DIR / "landing" / "biometrics"

app = FastAPI(
    title="Creative pipeline — ingestion API",
    version="1.0",
    description=__doc__,
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    try:
        creatives = len(discover_creatives())
    except FileNotFoundError:
        creatives = 0
    pending = len(list(LANDING_DIR.glob("*.parquet"))) if LANDING_DIR.exists() else 0
    return {
        "status": "ok",
        "video_dir": str(config.VIDEO_DIR),
        "creatives_on_disk": creatives,
        "biometric_batches_pending_load": pending,
    }


@app.get("/creatives", tags=["creatives"])
def list_creatives() -> list[dict]:
    """Whatever is currently on disk. Cheap: no warehouse read."""
    try:
        refs = discover_creatives()
    except FileNotFoundError:
        return []
    return [{"creative_id": r.creative_id, "title": r.title,
             "source_label": r.source_label, "file_name": r.file_name,
             "sha256": r.sha256, "size_bytes": r.size_bytes} for r in refs]


@app.post("/creatives", response_model=CreativeAccepted, status_code=201,
          tags=["creatives"])
def add_creative(body: CreativeFromURL) -> CreativeAccepted:
    """Register a new creative by URL.

    Downloads it into VIDEO_DIR and returns its content-addressed identity.
    Does not extract — that is the processing API's job, deliberately, so a
    slow model run never blocks an ingest request.
    """
    try:
        path = download.fetch(body.url, config.VIDEO_DIR, body.title)
    except ValueError as exc:                       # size/scheme rejections
        raise HTTPException(status_code=413, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"could not fetch url: {exc}")

    ref = CreativeRef.from_path(path)
    return CreativeAccepted(
        creative_id=ref.creative_id, title=body.title or ref.title,
        sha256=ref.sha256, size_bytes=ref.size_bytes, path=str(path),
        next=f"POST /jobs on the processing API with "
             f'{{"creative_id": "{ref.creative_id}"}} to extract features',
    )


@app.post("/biometrics", response_model=BatchAccepted, status_code=202,
          tags=["audience"])
def add_biometrics(batch: BiometricBatch) -> BatchAccepted:
    """Append a batch of viewer-seconds to the landing zone.

    Shaped for a browser client posting `getUserMedia`-derived expression data
    every few seconds: small batches, no transaction, 202 rather than 200
    because the rows are not queryable until the next load.
    """
    received_at = datetime.now(timezone.utc)
    batch_id = uuid.uuid4().hex[:12]

    rows = []
    for row in batch.rows:
        record = {
            "session_id": batch.session_id, "viewer_id": batch.viewer_id,
            "creative_id": batch.creative_id, "second": row.second,
            "attention_index": row.attention_index,
            "gaze_on_screen": row.gaze_on_screen,
            "face_detected": row.face_detected,
            "valence": row.valence, "arousal": row.arousal,
        }
        # Same p_<label> columns as audience.biometric_seconds, so ingested and
        # synthetic rows share one shape.
        for label in config.EMOTION_LABELS:
            record[f"p_{label}"] = (row.emotions or {}).get(label)
        rows.append(record)

    LANDING_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["data_source"] = "ingested"
    df["batch_id"] = batch_id
    df["received_at"] = received_at.isoformat(timespec="seconds")
    df.to_parquet(LANDING_DIR / f"{received_at:%Y%m%dT%H%M%S}_{batch_id}.parquet",
                  index=False, compression="zstd")

    return BatchAccepted(
        batch_id=batch_id, rows_accepted=len(rows),
        landed_at=received_at.isoformat(timespec="seconds"),
        note="Landed as Parquet. Run `python -m api.load` to make it queryable "
             "in audience.ingested_biometric_seconds.",
    )


@app.get("/landing", tags=["audience"])
def landing_status() -> dict:
    """What is waiting to be loaded."""
    files = sorted(LANDING_DIR.glob("*.parquet")) if LANDING_DIR.exists() else []
    return {
        "pending_batches": len(files),
        "pending_bytes": sum(f.stat().st_size for f in files),
        "oldest": files[0].name if files else None,
        "load_with": "python -m api.load",
    }
