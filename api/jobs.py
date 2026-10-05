"""A minimal in-process job store for the processing API.

One worker thread, so jobs run in submission order and never contend over the
ONNX sessions. That is the whole queue: no Redis, no Celery, no broker.

The limits are real and stated rather than hidden: jobs live in memory and are
lost on restart, and a second replica would have its own store. The seam for a
real queue is `pipeline.run.extract_one` -- a Celery or RQ worker calls exactly
the function `_run` calls below, and nothing else changes.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from pipeline import config, registry
from pipeline.io import CreativeRef, discover_creatives
from pipeline.run import extract_one

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="extract")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()

MAX_JOBS_RETAINED = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _set(job_id: str, **fields) -> None:
    with _lock:
        _jobs[job_id].update(fields)


def get(job_id: str) -> dict | None:
    with _lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def list_jobs(limit: int = 50) -> list[dict]:
    with _lock:
        return [dict(j) for j in list(_jobs.values())[-limit:]][::-1]


def submit(url: str | None, creative_id: str | None,
           extractors: list[str] | None, force: bool) -> dict:
    job_id = uuid.uuid4().hex[:12]
    record = {
        "job_id": job_id, "status": "queued", "creative_id": creative_id,
        "extractors": extractors or [m.NAME for m in registry.EXTRACTORS],
        "created_at": _now(), "started_at": None, "finished_at": None,
        "duration_s": None, "results": [], "output_dir": None, "error": None,
    }
    with _lock:
        _jobs[job_id] = record
        # Bounded memory: drop the oldest records once past the cap.
        while len(_jobs) > MAX_JOBS_RETAINED:
            _jobs.pop(next(iter(_jobs)))
    _executor.submit(_run, job_id, url, creative_id, extractors, force)
    return dict(record)


def _resolve_ref(url: str | None, creative_id: str | None) -> CreativeRef:
    if url:
        from .download import fetch
        path = fetch(url, config.VIDEO_DIR)
        return CreativeRef.from_path(path)
    matches = [r for r in discover_creatives()
               if creative_id in (r.creative_id, r.file_name)
               or creative_id.lower() in r.creative_id.lower()]
    if not matches:
        raise FileNotFoundError(f"no creative matching {creative_id!r}")
    return matches[0]


def _run(job_id: str, url, creative_id, extractors, force) -> None:
    started = time.perf_counter()
    _set(job_id, status="running", started_at=_now())
    try:
        ref = _resolve_ref(url, creative_id)
        mods = registry.resolve(extractors)
        _set(job_id, creative_id=ref.creative_id,
             extractors=[m.NAME for m in mods])

        rows = extract_one(ref, mods, force=force)
        # extract_one records only what it ran; cached extractors return no row.
        failed = [r for r in rows if r["status"] != "ok"]
        _set(job_id,
             status="failed" if failed else "succeeded",
             results=[{k: r[k] for k in
                       ("extractor", "status", "rows_written", "tables", "duration_s", "error")}
                      for r in rows],
             output_dir=str(config.RAW_DIR),
             error="; ".join(f"{r['extractor']}: {r['error']}" for r in failed) or None,
             finished_at=_now(), duration_s=round(time.perf_counter() - started, 2))
    except Exception as exc:
        _set(job_id, status="failed", error=f"{type(exc).__name__}: {exc}",
             finished_at=_now(), duration_s=round(time.perf_counter() - started, 2))
        traceback.print_exc()
