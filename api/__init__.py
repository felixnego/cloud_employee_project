"""Two small HTTP services, for engineering stakeholders.

`api.ingest`      -- get new data in: a video by URL, or a batch of biometrics.
`api.processing`  -- run the extraction pipeline as an async job, so it can sit
                     inside someone else's pipeline or job queue.

They are deliberately separate apps: ingestion is about landing data in this
project's warehouse, processing is about using the video pipeline as a service
with no opinion about where results go. Different consumers, different lifecycles.

Both are thin. Neither contains analysis logic, and neither writes to the DuckDB
file -- see `api.ingest` for why that matters.
"""
