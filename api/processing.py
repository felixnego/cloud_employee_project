"""Processing API — the video pipeline as an async service.

For engineering stakeholders who want the feature extraction but not the rest
of this project: a step inside someone else's pipeline, or a worker behind a
job queue.

Extraction takes seconds to minutes per video, so every call is asynchronous:
POST a job, get a 202 and a job_id, poll it. There is no synchronous endpoint
on purpose — an HTTP request that holds a connection open for four minutes is
a request that times out somewhere in the middle.

It is usable standalone. Point WAREHOUSE_DIR and VIDEO_DIR anywhere and this
service becomes a self-contained feature extractor that knows nothing about
the audience layer or the marts.

    uvicorn api.processing:app --host 0.0.0.0 --port 8002
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from pipeline import config, registry

from . import jobs
from .models import Job, JobRequest

app = FastAPI(
    title="Creative pipeline — processing API",
    version="1.0",
    description=__doc__,
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "extractors": len(registry.EXTRACTORS),
        "warehouse_dir": str(config.WAREHOUSE_DIR),
        "video_dir": str(config.VIDEO_DIR),
        "frame_hz": config.FRAME_HZ,
    }


@app.get("/extractors", tags=["meta"])
def list_extractors() -> list[dict]:
    """The nine features, their versions and their output grain.

    Discoverability for someone choosing a subset: the same registry the CLI
    and the docs read, so it cannot drift.
    """
    return registry.describe()


@app.post("/jobs", response_model=Job, status_code=202, tags=["jobs"])
def create_job(body: JobRequest) -> Job:
    """Queue an extraction.

    Supply either `url` (fetched first) or `creative_id` (already on disk).
    `extractors` selects a subset; declared dependencies are pulled in
    automatically, so asking for `labels` also runs `shots`.
    """
    if not body.url and not body.creative_id:
        raise HTTPException(422, "provide either url or creative_id")
    if body.url and body.creative_id:
        raise HTTPException(422, "provide url or creative_id, not both")
    if body.extractors:
        try:
            registry.resolve(body.extractors)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    return Job(**jobs.submit(body.url, body.creative_id,
                             body.extractors, body.force))


@app.get("/jobs", response_model=list[Job], tags=["jobs"])
def list_jobs(limit: int = 50) -> list[Job]:
    return [Job(**j) for j in jobs.list_jobs(limit)]


@app.get("/jobs/{job_id}", response_model=Job, tags=["jobs"])
def get_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return Job(**job)
