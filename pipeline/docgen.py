"""Generates docs/schema.md from the registry and the live warehouse.

Hand-written schema docs drift the moment someone adds a column. Here the
*prose* is hand-written (grain, purpose, gotchas -- things a tool cannot know)
and the *structure* is introspected from DuckDB at generation time, so the
column lists and types are true by construction.

Regenerate with:  python -m pipeline.run docs
"""
from __future__ import annotations

from pathlib import Path

from . import config, registry
from .transform import connect

# --------------------------------------------------------------- hand-written --
LAYERS = [
    ("Dimensions", ["creatives", "shots"]),
    ("Spine (time series)", ["creative_seconds", "creative_frames"]),
    ("Language and text", ["transcript_segments", "transcript_words",
                           "screen_text", "screen_text_detections"]),
    ("Detections", ["face_detections", "object_detections", "shot_label_scores"]),
    ("Summary", ["creative_summary"]),
    ("Provenance", ["run_manifest", "sampling_config"]),
]

TABLE_DOCS: dict[str, dict] = {
    "creatives": dict(
        grain="one row per creative",
        key="creative_id",
        purpose="The creative dimension: file and codec facts plus whole-ad "
                "rollups of pacing, voice, sound and look.",
        notes=["`creative_id` is a slug of the filename; `file_sha256` is the "
               "real version key. Re-encoding a video gives a new sha256 and "
               "re-triggers extraction.",
               "Use `display_name`, not `title`, for any chart, pivot or GROUP BY "
               "a human will read: titles are not unique (two sample creatives "
               "are the same campaign by different creators)."],
    ),
    "creative_summary": dict(
        grain="one row per creative",
        key="creative_id",
        purpose="The table to open first. Everything here is derivable from the "
                "others; it exists so cross-creative comparison is a SELECT *.",
        notes=["`*_second_share` columns are fractions of the ad's seconds, so "
               "they are comparable across ads of different lengths."],
    ),
    "shots": dict(
        grain="one row per (creative, shot)",
        key="creative_id, shot_index",
        purpose="The unit a creative team thinks in. Carries CLIP's semantic "
                "labels plus what was visible and audible inside the shot.",
        notes=["`label_*` is the winning label on that axis and `conf_*` its "
               "within-axis probability. A low `conf_*` means CLIP was "
               "genuinely unsure -- filter on it before trusting a label.",
               "`position_in_ad` lets you compare openings and end-frames "
               "across creatives of different lengths."],
    ),
    "creative_seconds": dict(
        grain="one row per (creative, second)",
        key="creative_id, second",
        purpose="THE JOIN TARGET for audience data. Everything that varies over "
                "time, aggregated to whole seconds.",
        notes=["Biometric traces, dial tests and survey timestamps almost always "
               "arrive at 1 Hz. Join on (creative_id, second) -- no "
               "interpolation, no window functions.",
               "`audio_class` is the one field with judgement in it: speech is "
               "model-grounded (Whisper + Silero VAD), music vs ambient is a "
               "threshold stated in `transforms/07_creative_seconds.sql`.",
               "`max_objects` and `max_text_detections` are NULL for seconds "
               "whose frames were never sampled by that model -- see "
               "`sampling_config`."],
    ),
    "creative_frames": dict(
        grain=f"one row per (creative, frame) at {config.FRAME_HZ} Hz",
        key="creative_id, frame_index",
        purpose="The native grain. Use it when a sub-second event matters -- a "
                "400 ms product reveal is invisible at 1 Hz.",
        notes=["`frame_index` is 1-based; `t_start_s = (frame_index - 1) / "
               f"{config.FRAME_HZ}`.",
               "NULL vs zero is meaningful. `n_objects` is NULL on frames YOLOX "
               "never looked at and 0 on frames it looked at and found nothing. "
               "The `objects_sampled` / `ocr_sampled` flags make that explicit.",
               "`is_cut` marks the first frame of every shot except the first, "
               "so `SUM(is_cut)` is a cut count."],
    ),
    "transcript_segments": dict(
        grain="one row per spoken utterance",
        key="creative_id, segment_index",
        purpose="The script as Whisper heard it, with timings.",
        notes=["`is_low_confidence` flags lines where Whisper's own "
               "`avg_logprob` or `no_speech_prob` suggests a hallucination -- "
               "common over instrumental music beds."],
    ),
    "transcript_words": dict(
        grain="one row per spoken word",
        key="creative_id, segment_index, word_index",
        purpose="Word-level timings, for locating a brand mention to the "
                "half-second.",
        notes=["`word_norm` is lowercased and stripped of punctuation, so "
               "joining a brand-term list needs no cleaning."],
    ),
    "screen_text": dict(
        grain="one row per on-screen text span",
        key="creative_id, screen_text_id",
        purpose="On-screen copy as time spans: a tagline held for four seconds "
                "is one row, not four detections.",
        notes=["`is_large_text` and `vertical_position` approximate the role of "
               "the copy -- headline versus legal line -- without anyone "
               "watching the ad.",
               "OCR is imperfect on stylised type. `max_score` is the "
               "recogniser's confidence; filter on it for clean text."],
    ),
    "screen_text_detections": dict(
        grain="one row per OCR detection per sampled frame",
        key="creative_id, frame_index, detection_index",
        purpose="Unconsolidated OCR output, kept for anyone who needs raw boxes.",
        notes=[],
    ),
    "face_detections": dict(
        grain="one row per detected face per sampled frame",
        key="creative_id, frame_index, face_index",
        purpose="Every face on screen, with its size, position and an 8-class "
                "expression.",
        notes=["`p_*` columns are the full expression distribution, not just "
               "the winner. They are NULL when the face was too small to "
               "classify (recorded as present but unlabelled rather than "
               "guessed at).",
               "This measures the expressions of actors *in the ad*, which is "
               "not the same construct as a viewer's expression. See the "
               "README note on where viewer-side measurement plugs in."],
    ),
    "object_detections": dict(
        grain="one row per detection per sampled frame",
        key="creative_id, frame_index, detection_index",
        purpose="COCO-80 objects and people, with on-screen size and position.",
        notes=["COCO has no class for 'lip oil' or 'football shirt'. Treat this "
               "as coarse scene composition (how many people, is there a car, a "
               "phone, a bottle), not product recognition.",
               "`area_frac` is the share of the frame the box covers, so "
               "'product fills the screen' is a filter, not an eyeball judgement."],
    ),
    "shot_label_scores": dict(
        grain="one row per (creative, shot, axis, label)",
        key="creative_id, shot_index, axis, label",
        purpose="Every CLIP label with its score, not just the winner.",
        notes=["Scores are softmaxed *within* an axis, so they sum to 1 per "
               "(shot, axis) and are comparable only inside an axis.",
               "`shots` holds the pivoted winners; use this table when the "
               "runner-up or the margin matters."],
    ),
    "run_manifest": dict(
        grain="one row per (creative, extractor) run",
        key="creative_id, extractor",
        purpose="Provenance. Which extractor version produced which rows, when, "
                "how long it took, and whether it failed.",
        notes=["A failed extractor does not abort the run, so this is where you "
               "check that the data you are about to analyse actually exists.",
               "`creative_sha256` ties rows to exact video bytes."],
    ),
    "sampling_config": dict(
        grain="one row per (creative, subsampled extractor)",
        key="creative_id, extractor",
        purpose="The sampling rate each model actually ran at, so the warehouse "
                "describes its own resolution.",
        notes=["`step_frames = frame_hz / sample_hz`. Used by the SQL to tell "
               "'never sampled' apart from 'sampled, found nothing'."],
    ),
}


def _extractor_section() -> str:
    rows = registry.describe()
    out = ["| # | Extractor | Feature | Tool | Grain | Writes |",
           "|---|-----------|---------|------|-------|--------|"]
    # Map each extractor to the raw tables it writes, read from the manifest.
    try:
        with connect(read_only=True) as con:
            written = dict(con.execute(
                "SELECT extractor, string_agg(DISTINCT tables, ', ') "
                "FROM run_manifest GROUP BY extractor").fetchall())
    except Exception:
        written = {}
    for r in rows:
        out.append(f"| {r['n']} | `{r['name']}` v{r['version']} | {r['feature']} "
                   f"| {r['tool']} | {r['grain']} "
                   f"| {written.get(r['name'], '-')} |")
    return "\n".join(out)


def _table_section(con, table: str) -> str:
    doc = TABLE_DOCS.get(table, {})
    cols = con.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
    """, [table]).fetchall()
    if not cols:
        return ""
    n_rows = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]

    parts = [f"### `{table}`", ""]
    parts.append(f"**Grain** {doc.get('grain', '?')}  ·  "
                 f"**Key** `{doc.get('key', '?')}`  ·  "
                 f"**Rows** {n_rows:,}")
    parts.append("")
    if doc.get("purpose"):
        parts += [doc["purpose"], ""]
    for note in doc.get("notes", []):
        parts.append(f"> {note}")
        parts.append("")
    parts.append("| Column | Type |")
    parts.append("|--------|------|")
    parts += [f"| `{name}` | {dtype} |" for name, dtype in cols]
    parts.append("")
    return "\n".join(parts)


def write_schema_doc(path: Path | None = None) -> Path:
    path = path or Path("docs/schema.md")
    path.parent.mkdir(parents=True, exist_ok=True)

    head = f"""# Data layer schema

*Generated by `python -m pipeline.run docs`. The prose is hand-written; every
column list, type and row count is introspected from the warehouse, so this
file cannot drift from the data.*

The warehouse has four layers. Each one is a directory or a file you can point
a tool at, and each is built by SQL you can read in `transforms/`.

```
video_data/*.mp4
      |
      |  pipeline/prepare.py          decode once -> cache/ (JPEG @ {config.FRAME_HZ} Hz + 16 kHz WAV)
      v
 [1] warehouse/raw/<table>/<creative_id>.parquet
      |                               one file per extractor per creative
      |  transforms/01_raw_views.sql   views, so marts never touch file paths
      v
 [2] raw_* views in warehouse.duckdb
      |
      |  transforms/02..09             dimensions, spine, rollups
      v
 [3] tables in warehouse.duckdb       <- analysts query this
      |
      |  transforms/10_export_marts.sql
      v
 [4] warehouse/marts/*.parquet + .csv <- everything else reads this
```

## The nine extractors

{_extractor_section()}

Each extractor is one file in `pipeline/extractors/` exposing `NAME`,
`VERSION`, `FEATURE`, `TOOL`, `GRAIN`, `REQUIRES` and a single
`extract(prepared) -> dict[table_name, DataFrame]`. Adding a tenth signal is
one file plus one line in `pipeline/registry.py`.

## How to query it

```python
import duckdb
con = duckdb.connect("warehouse/warehouse.duckdb", read_only=True)
con.sql("SELECT * FROM creative_summary").df()
```

or without this repo at all:

```python
import pandas as pd
pd.read_parquet("warehouse/marts/creative_seconds.parquet")
```

## Tables
"""
    body = [head]
    with connect(read_only=True) as con:
        existing = {r[0] for r in con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
        ).fetchall()}
        for layer, tables in LAYERS:
            present = [t for t in tables if t in existing]
            if not present:
                continue
            body.append(f"\n## {layer}\n")
            for table in present:
                body.append(_table_section(con, table))

    path.write_text("\n".join(body))
    return path
