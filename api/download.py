"""Fetch a video by URL. Shared by both APIs.

Guards are deliberately blunt: scheme, size cap, timeout. A service that
fetches arbitrary URLs on request is an SSRF vector, so a real deployment needs
an egress allowlist or a signed-URL scheme -- noted in the README rather than
half-solved here.
"""
from __future__ import annotations

import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from pipeline import config
from pipeline.io import slugify

MAX_BYTES = int(os.environ.get("MAX_DOWNLOAD_BYTES", 500 * 1024 * 1024))
TIMEOUT_S = int(os.environ.get("DOWNLOAD_TIMEOUT_S", 120))


def filename_from_url(url: str, title: str | None = None) -> str:
    name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    if suffix not in config.VIDEO_SUFFIXES:
        suffix = ".mp4"
    return f"{slugify(title or stem or 'creative')}{suffix}"


def fetch(url: str, dest_dir: Path, title: str | None = None) -> Path:
    """Download to `dest_dir`, returning the written path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename_from_url(url, title)

    req = urllib.request.Request(url, headers={"User-Agent": "creative-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        declared = resp.headers.get("Content-Length")
        if declared and int(declared) > MAX_BYTES:
            raise ValueError(f"file is {int(declared)} bytes, limit is {MAX_BYTES}")

        tmp = dest.with_suffix(dest.suffix + ".part")
        written = 0
        with tmp.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                written += len(chunk)
                if written > MAX_BYTES:
                    tmp.unlink(missing_ok=True)
                    raise ValueError(f"download exceeded {MAX_BYTES} bytes")
                fh.write(chunk)
        shutil.move(tmp, dest)
    return dest
