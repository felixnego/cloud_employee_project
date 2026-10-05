"""Command line entry point for the whole pipeline.

    python -m pipeline.run list                      # the nine features
    python -m pipeline.run extract                   # all creatives, all features
    python -m pipeline.run extract --only visual,audio --creative coordown
    python -m pipeline.run extract --force           # ignore the manifest
    python -m pipeline.run transform                 # build the DuckDB marts
    python -m pipeline.run docs                      # regenerate docs/schema.md
    python -m pipeline.run all                       # extract -> transform -> docs

Design notes worth knowing before you extend this:

* One extractor failing never aborts the run. The failure is recorded in the
  manifest with its traceback and the remaining extractors continue, because
  a broken OCR model should not cost you the transcript.
* Re-running is free. The manifest is keyed on (video sha256, extractor,
  extractor version), so unchanged work is skipped; bump a VERSION to
  invalidate just that extractor's output.
* `extract_one` is the integration seam. An API handler or a queue worker
  that receives a newly uploaded video calls exactly this function; nothing
  about the core needs to change to move from batch to streaming.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from types import ModuleType

from . import config, registry, transform
from .io import CreativeRef, RunRecorder, discover_creatives, is_fresh, write_table
from .prepare import Prepared, prepare


# ------------------------------------------------------------------ helpers --
def _select(refs: list[CreativeRef], patterns: list[str] | None) -> list[CreativeRef]:
    if not patterns:
        return refs
    chosen, missing = [], []
    for pat in patterns:
        needle = pat.lower()
        hits = [r for r in refs
                if needle in r.creative_id.lower() or needle in r.file_name.lower()]
        if not hits:
            missing.append(pat)
        chosen.extend(h for h in hits if h not in chosen)
    if missing:
        known = "\n  ".join(r.creative_id for r in refs)
        raise SystemExit(f"no creative matches {missing}. Known creatives:\n  {known}")
    return chosen


def _table(rows: list[dict], columns: list[str]) -> str:
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    line = "  ".join(c.upper().ljust(widths[c]) for c in columns)
    out = [line, "  ".join("-" * widths[c] for c in columns)]
    for r in rows:
        out.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(out)


# --------------------------------------------------------- the extraction seam --
def extract_one(ref: CreativeRef,
                extractors: list[ModuleType],
                force: bool = False,
                force_cache: bool = False) -> list[dict]:
    """Extract every requested feature for one creative. The integration point.

    Returns the manifest rows it wrote, so a caller (CLI, API, worker) can
    report on the run without re-reading Parquet.
    """
    print(f"\n=== {ref.creative_id}")
    print(f"    {ref.file_name}  sha256={ref.sha256[:12]}  "
          f"{ref.size_bytes / 1e6:.1f} MB")

    t0 = time.perf_counter()
    prepared: Prepared = prepare(ref, force=force_cache)
    print(f"    cache: {prepared.n_frames} frames @ {config.FRAME_HZ} Hz, "
          f"audio={'yes' if prepared.audio_path else 'no'}, "
          f"{prepared.width}x{prepared.height}, {prepared.duration_s:.1f}s "
          f"({time.perf_counter() - t0:.1f}s)")

    recorder = RunRecorder()
    for mod in extractors:
        if not force and is_fresh(ref, mod.NAME, mod.VERSION):
            print(f"    - {mod.NAME:<11} cached")
            continue

        started = time.perf_counter()
        try:
            tables = mod.extract(prepared)
            rows = 0
            for table_name, df in tables.items():
                write_table(df, table_name, ref.creative_id)
                rows += len(df)
            recorder.record(ref, mod.NAME, mod.VERSION, "ok", rows,
                            list(tables), started)
            print(f"    + {mod.NAME:<11} {rows:>6} rows  "
                  f"{time.perf_counter() - started:>6.1f}s  "
                  f"-> {', '.join(tables)}")
        except Exception as exc:                      # keep going; record why
            recorder.record(ref, mod.NAME, mod.VERSION, "error", 0, [],
                            started, error=f"{type(exc).__name__}: {exc}")
            print(f"    ! {mod.NAME:<11} FAILED after "
                  f"{time.perf_counter() - started:.1f}s: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            traceback.print_exc(limit=4)

    written = list(recorder.rows)
    recorder.flush(ref.creative_id)
    return written


# -------------------------------------------------------------- subcommands --
def cmd_list(args) -> int:
    rows = registry.describe()
    print(f"\n{len(rows)} extractors registered:\n")
    print(_table(rows, ["n", "name", "version", "tool", "grain", "requires"]))
    print("\nFeatures:")
    for r in rows:
        print(f"  {r['n']}. {r['name']:<11} {r['feature']}")
    print()
    try:
        refs = discover_creatives()
    except FileNotFoundError as exc:
        print(f"(no creatives: {exc})\n")
        return 0
    print(f"{len(refs)} creatives in {config.VIDEO_DIR}:\n")
    print(_table([{"creative_id": r.creative_id, "title": r.title,
                   "source": r.source_label or "-",
                   "MB": f"{r.size_bytes / 1e6:.1f}",
                   "sha256": r.sha256[:12]} for r in refs],
                 ["creative_id", "title", "source", "MB", "sha256"]))
    print()
    return 0


def cmd_prepare(args) -> int:
    for ref in _select(discover_creatives(), args.creative):
        p = prepare(ref, force=args.force)
        print(f"{ref.creative_id:<40} {p.n_frames:>5} frames  "
              f"audio={'yes' if p.audio_path else 'no'}")
    return 0


def cmd_extract(args) -> int:
    extractors = registry.resolve(args.only.split(",") if args.only else None)
    refs = _select(discover_creatives(), args.creative)
    print(f"{len(refs)} creative(s) x {len(extractors)} extractor(s): "
          f"{', '.join(m.NAME for m in extractors)}")

    failures: list[dict] = []
    t0 = time.perf_counter()
    for ref in refs:
        failures += [r for r in extract_one(ref, extractors,
                                            force=args.force,
                                            force_cache=args.force_cache)
                     if r["status"] != "ok"]

    print(f"\nextraction finished in {time.perf_counter() - t0:.1f}s")
    if failures:
        print(f"\n{len(failures)} extractor run(s) failed:", file=sys.stderr)
        for f in failures:
            print(f"  {f['creative_id']} / {f['extractor']}: {f['error']}",
                  file=sys.stderr)
        return 1
    return 0


def cmd_transform(args) -> int:
    print(f"building marts in {config.DUCKDB_PATH}")
    transform.run()
    print("\ntables:")
    for name, size in transform.table_summary():
        print(f"  {name:<28} {size:>9,} rows")
    return 0


def cmd_docs(args) -> int:
    from .docgen import write_schema_doc
    path = write_schema_doc()
    print(f"wrote {path}")
    return 0


def cmd_all(args) -> int:
    rc = cmd_extract(args)
    cmd_transform(args)
    cmd_docs(args)
    return rc


# -------------------------------------------------------------------- parser --
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pipeline.run",
                                 description="Creative feature extraction pipeline")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_extract_flags(p):
        p.add_argument("--creative", action="append", metavar="MATCH",
                       help="creative_id or filename substring; repeatable")
        p.add_argument("--only", metavar="a,b",
                       help="comma-separated extractor names (dependencies pulled in)")
        p.add_argument("--force", action="store_true",
                       help="re-run extractors even if the manifest says they are fresh")
        p.add_argument("--force-cache", action="store_true",
                       help="also re-decode frames and audio")

    sub.add_parser("list", help="show extractors and discovered creatives").set_defaults(fn=cmd_list)

    p = sub.add_parser("prepare", help="decode frames + audio into the cache")
    p.add_argument("--creative", action="append", metavar="MATCH")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("extract", help="run extractors and write raw Parquet")
    add_extract_flags(p)
    p.set_defaults(fn=cmd_extract)

    sub.add_parser("transform", help="build DuckDB marts from raw Parquet").set_defaults(fn=cmd_transform)
    sub.add_parser("docs", help="regenerate docs/schema.md from the warehouse").set_defaults(fn=cmd_docs)

    p = sub.add_parser("all", help="extract, then transform, then docs")
    add_extract_flags(p)
    p.set_defaults(fn=cmd_all)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
