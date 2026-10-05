"""Discovery, content addressing, Parquet writes and the run manifest.

Two ideas carry most of the weight here:

1. A creative is identified by a slug of its filename but *versioned* by the
   sha256 of its bytes. Re-running is therefore free and safe: if the bytes and
   the extractor version haven't changed, there is nothing to do.
2. Every extractor writes its own Parquet file under `warehouse/raw/<table>/`.
   Nothing writes to a shared table, so extractors never collide and can be
   run, re-run or parallelised independently.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

# ------------------------------------------------------------------ helpers --


def slugify(value: str) -> str:
    """Filename -> stable, URL-safe creative_id."""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^\w\s-]", " ", value).strip().lower()
    return re.sub(r"[\s_-]+", "-", value).strip("-")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ffprobe(path: Path) -> dict:
    """Raw ffprobe JSON for one file. The only place we shell out for metadata."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


# -------------------------------------------------------------- the creative --


@dataclass(frozen=True)
class CreativeRef:
    """A video file on disk, content-addressed."""
    creative_id: str
    title: str
    source_label: str | None
    file_name: str
    path: Path
    sha256: str
    size_bytes: int

    @classmethod
    def from_path(cls, path: Path) -> "CreativeRef":
        stem = path.stem
        # Convention in the sample set: "<Creative title> - <Advertiser or creator>"
        title, _, source = stem.rpartition(" - ")
        if not title:                      # no separator present
            title, source = stem, ""
        return cls(
            creative_id=slugify(stem),
            title=title.strip(),
            source_label=source.strip() or None,
            file_name=path.name,
            path=path,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )


def discover_creatives(video_dir: Path | None = None) -> list[CreativeRef]:
    """Every video in the input folder, in a deterministic order."""
    video_dir = video_dir or config.VIDEO_DIR
    paths = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.VIDEO_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no video files found in {video_dir}")
    return [CreativeRef.from_path(p) for p in paths]


# ----------------------------------------------------------- parquet writing --


def write_table(df: pd.DataFrame, table: str, creative_id: str) -> Path:
    """One Parquet file per (table, creative). Globbed back together in SQL."""
    out_dir = config.RAW_DIR / table
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{creative_id}.parquet"
    if "creative_id" not in df.columns:
        df = df.assign(creative_id=creative_id)
    cols = ["creative_id"] + [c for c in df.columns if c != "creative_id"]
    df[cols].to_parquet(out_path, index=False, compression="zstd")
    return out_path


# ------------------------------------------------------------- run manifest --

_MANIFEST_DIR = config.RAW_DIR / "_manifest"
_MANIFEST_COLUMNS = [
    "creative_id", "creative_sha256", "extractor", "extractor_version",
    "status", "rows_written", "tables", "run_id", "finished_at",
    "duration_s", "error",
]


def read_manifest(creative_id: str | None = None) -> pd.DataFrame:
    """Manifest rows for one creative, or all of them. Latest run per key wins."""
    if not _MANIFEST_DIR.exists():
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)
    pattern = f"{creative_id}.parquet" if creative_id else "*.parquet"
    files = sorted(_MANIFEST_DIR.glob(pattern))
    if not files:
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return (df.sort_values("finished_at")
              .drop_duplicates(["creative_id", "extractor"], keep="last")
              .reset_index(drop=True))


def append_manifest(rows: list[dict], creative_id: str) -> None:
    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = _MANIFEST_DIR / f"{creative_id}.parquet"
    new = pd.DataFrame(rows, columns=_MANIFEST_COLUMNS)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    new.to_parquet(path, index=False)


def is_fresh(ref: CreativeRef, extractor: str, version: str) -> bool:
    """True when this exact (bytes, extractor, version) already succeeded."""
    man = read_manifest(ref.creative_id)
    if man.empty:
        return False
    hit = man[(man.extractor == extractor)
              & (man.extractor_version == version)
              & (man.creative_sha256 == ref.sha256)
              & (man.status == "ok")]
    return not hit.empty


@dataclass
class RunRecorder:
    """Collects one manifest row per extractor invocation."""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    rows: list[dict] = field(default_factory=list)

    def record(self, ref: CreativeRef, extractor: str, version: str,
               status: str, rows_written: int, tables: list[str],
               started: float, error: str | None = None) -> None:
        self.rows.append({
            "creative_id": ref.creative_id,
            "creative_sha256": ref.sha256,
            "extractor": extractor,
            "extractor_version": version,
            "status": status,
            "rows_written": int(rows_written),
            "tables": ",".join(tables),
            "run_id": self.run_id,
            "finished_at": utcnow(),
            "duration_s": round(time.perf_counter() - started, 3),
            "error": error,
        })

    def flush(self, creative_id: str) -> None:
        if self.rows:
            append_manifest(self.rows, creative_id)
            self.rows = []
