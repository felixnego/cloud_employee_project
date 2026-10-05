"""Generates docs/architecture.svg.

A script rather than a hand-drawn file so the diagram stays editable and
diffable, like docs/schema.md and the OpenAPI specs.

    python docs/make_architecture.py

Tints use fill-opacity rather than 8-digit hex: the latter is fine in browsers
but not in every SVG rasteriser, and a diagram that only renders in some
viewers is worse than no diagram.
"""
from __future__ import annotations

from pathlib import Path

W, H = 1180, 946

INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#9b9a95"
SURFACE, CARD, BORDER = "#fcfcfb", "#ffffff", "#dcdbd6"
BLUE, AQUA, ORANGE, VIOLET = "#2a78d6", "#0f9e6e", "#e2651f", "#4a3aa7"

MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"

out: list[str] = []
add = out.append


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, *, fill=CARD, fo=1.0, stroke=BORDER, so=1.0, rx=9, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'fill-opacity="{fo}" stroke="{stroke}" stroke-opacity="{so}" '
        f'stroke-width="{sw}"{d}/>')


def text(x, y, t, *, size=13, fill=INK, anchor="start", family=SANS, weight="400"):
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(t)}</text>')


def line(d, *, colour=FAINT, sw=1.8, dash=None, marker="arrow"):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{sw}"{ds} '
        f'marker-end="url(#{marker})"/>')


def vline(x, y1, y2, **kw):
    line(f"M {x} {y1} V {y2}", **kw)


def elbow(x1, y1, x2, y2, **kw):
    line(f"M {x1} {y1} V {(y1 + y2) / 2} H {x2} V {y2}", **kw)


def band(y, label, colour):
    add(f'<rect x="0" y="{y}" width="5" height="24" rx="2.5" fill="{colour}"/>')
    text(16, y + 17, label, size=10.5, fill=colour, weight="700")


def chip(x, y, label, colour):
    w = 7.1 * len(label) + 20
    box(x, y, w, 24, fill=colour, fo=0.10, stroke=colour, so=0.45, rx=12, sw=1.1)
    text(x + w / 2, y + 16.5, label, size=11, anchor="middle", family=MONO, fill=colour)
    return w


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" font-family="{SANS}">')
add("<defs>" + "".join(
    f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{col}"/></marker>'
    for mid, col in (("arrow", FAINT), ("arrowAqua", AQUA),
                     ("arrowOrange", ORANGE), ("arrowViolet", VIOLET))) + "</defs>")
add(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')

text(28, 40, "Creative feature extraction — architecture", size=19, weight="700")
text(28, 62, "Five MP4s in; a queryable warehouse out. Measured creative features, "
             "a synthetic audience panel, and two HTTP services.", size=12.5, fill=MUTED)

# ----------------------------------------------------------------- sources --
band(88, "SOURCES", BLUE)
box(150, 82, 268, 56)
text(166, 104, "video_data/*.mp4", size=13, family=MONO, weight="600")
text(166, 123, "five creatives, mounted read-only", size=11, fill=MUTED)

box(444, 82, 296, 56, stroke=ORANGE, so=0.55)
text(460, 104, "POST /creatives {url}", size=12.5, family=MONO, weight="600", fill=ORANGE)
text(460, 123, "ingest API :8001 — lands in video_data/", size=11, fill=MUTED)

box(766, 82, 386, 56, stroke=ORANGE, so=0.55)
text(782, 104, "POST /biometrics {session, rows[]}", size=12.5, family=MONO,
     weight="600", fill=ORANGE)
text(782, 123, "ingest API :8001 — getUserMedia-shaped batches", size=11, fill=MUTED)

# -------------------------------------------------------------- extraction --
band(186, "EXTRACTION", BLUE)
box(150, 180, 590, 56, fill=BLUE, fo=0.06, stroke=BLUE, so=0.40)
text(166, 202, "prepare — decode once", size=13, weight="600", fill=BLUE)
text(166, 221, "ffmpeg → cache/<sha256>/ : frames @ 4 Hz + 16 kHz mono WAV",
     size=11, fill=MUTED, family=MONO)

box(150, 256, 590, 104, fill=BLUE, fo=0.06, stroke=BLUE, so=0.40)
text(166, 278, "nine extractors", size=13, weight="600", fill=BLUE)
text(166, 296, "one interface · independent · manifest skips unchanged work",
     size=10.5, fill=MUTED)
cx = 166
for name in ["metadata", "shots", "transcript", "visual", "audio"]:
    cx += chip(cx, 306, name, BLUE) + 6
cx = 166
for name in ["ocr", "faces", "objects", "labels"]:
    cx += chip(cx, 332, name, BLUE) + 6

box(766, 256, 290, 104, stroke=ORANGE, so=0.55)
text(782, 278, "processing API :8002", size=12.5, weight="700", fill=ORANGE, family=MONO)
text(782, 298, "POST /jobs → 202 → poll", size=11, family=MONO, fill=MUTED)
text(782, 317, "async · one worker · no broker", size=11, fill=MUTED)
text(782, 336, "calls extract_one() — the", size=11, fill=MUTED)
text(782, 351, "seam for a real queue", size=11, fill=MUTED)

# ------------------------------------------------------------------ landed --
band(406, "LANDED", AQUA)
box(150, 400, 286, 60, fill=AQUA, fo=0.07, stroke=AQUA, so=0.45)
text(166, 422, "warehouse/raw/", size=12.5, family=MONO, weight="600", fill=AQUA)
text(166, 441, "one Parquet per extractor, per creative", size=11, fill=MUTED)

box(458, 400, 282, 60, fill=VIOLET, fo=0.07, stroke=VIOLET, so=0.45)
text(474, 422, "warehouse/audience/", size=12.5, family=MONO, weight="600", fill=VIOLET)
text(474, 441, "synthetic panel — generated", size=11, fill=MUTED)

box(766, 400, 386, 60, fill=ORANGE, fo=0.07, stroke=ORANGE, so=0.45)
text(782, 422, "warehouse/landing/", size=12.5, family=MONO, weight="600", fill=ORANGE)
text(782, 441, "ingested batches, awaiting make load-ingested", size=11, fill=MUTED)

# --------------------------------------------------------------- warehouse --
band(508, "WAREHOUSE", AQUA)
box(150, 500, 1002, 152, stroke=AQUA, so=0.55, sw=1.8)
text(170, 526, "warehouse.duckdb", size=14, family=MONO, weight="700", fill=AQUA)
# Right-aligned: the left half of this row is a connector channel.
text(1134, 526, "built by transforms/*.sql — raw views → dimensions → spine → marts",
     size=11, fill=MUTED, anchor="end")

for x, w, name, tag, tagcol, rows, col in (
        (170, 300, "main", "MEASURED from real video", AQUA,
         ["creatives · shots · transcripts", "creative_seconds  ← the spine"], AQUA),
        (490, 300, "audience", "SYNTHETIC — every row stamped", VIOLET,
         ["viewers · sessions · survey · IAT", "biometric_seconds"], VIOLET),
        (810, 324, "analysis", "MIXED provenance — labelled", ORANGE,
         ["viewer_seconds  ← the join", "creative_performance"], ORANGE)):
    box(x, 542, w, 94, fill=col, fo=0.06, stroke=col, so=0.35)
    text(x + 16, 564, name, size=13, family=MONO, weight="700")
    text(x + 16, 582, tag, size=10, fill=tagcol, weight="700")
    text(x + 16, 604, rows[0], size=11, fill=MUTED, family=MONO)
    text(x + 16, 622, rows[1], size=11, fill=MUTED, family=MONO)

# --------------------------------------------------------------- published --
band(708, "PUBLISHED", AQUA)
box(150, 702, 490, 56, fill=AQUA, fo=0.07, stroke=AQUA, so=0.45)
text(166, 724, "warehouse/marts/*.parquet + .csv", size=12.5, family=MONO,
     weight="600", fill=AQUA)
text(166, 743, "committed — readable with no Docker, no build", size=11, fill=MUTED)

box(662, 702, 438, 56)
text(678, 724, "docs/schema.md · openapi-*.json", size=12.5, family=MONO, weight="600")
text(678, 743, "generated from the warehouse and the apps", size=11, fill=MUTED)

# --------------------------------------------------------------- consumers --
band(810, "CONSUMERS", BLUE)
for i, (label, sub) in enumerate([
        ("notebook", "11-check acceptance + first read"),
        ("DuckDB / MotherDuck SQL", "the analyst surface"),
        ("pandas · Polars · BI", "straight off Parquet")]):
    x = 150 + i * 336
    box(x, 804, 318, 54)
    text(x + 16, 826, label, size=12.5, weight="600")
    text(x + 16, 844, sub, size=11, fill=MUTED)

# ------------------------------------------------------------------- flows --
vline(284, 138, 180)                                              # videos → prepare
elbow(592, 138, 400, 180, colour=ORANGE, marker="arrowOrange")    # POST /creatives → prepare
vline(445, 236, 256)                                              # prepare → extractors
line("M 766 308 H 744", colour=ORANGE, marker="arrowOrange")      # processing API → extractors
vline(293, 360, 400)                                              # extractors → raw
vline(293, 460, 500, colour=AQUA, marker="arrowAqua")             # raw → duckdb
vline(1115, 138, 400, colour=ORANGE, marker="arrowOrange")        # biometrics → landing
elbow(900, 460, 700, 542, colour=ORANGE, dash="5 4", marker="arrowOrange")  # landing → audience
vline(646, 460, 542, colour=VIOLET, marker="arrowViolet")         # audience parquet → schema

# the generator reads the measured spine and writes the synthetic panel
line("M 320 636 V 678 H 480 V 462", colour=VIOLET, sw=1.6, dash="5 4",
     marker="arrowViolet")
text(496, 682, "audience/generate.py — the response is a function of the measured creative",
     size=10.5, fill=VIOLET)

vline(395, 652, 702, colour=AQUA, marker="arrowAqua")             # duckdb → marts
vline(395, 758, 804)                                              # marts → consumers
vline(651, 652, 804)                                              # duckdb → SQL consumer

# -------------------------------------------------------------------- note --
box(662, 882, 490, 46, fill=ORANGE, fo=0.07, stroke=ORANGE, so=0.55, dash="4 3")
text(678, 901, "Single-writer rule", size=11.5, weight="700", fill=ORANGE)
text(678, 917, "No API ever writes to warehouse.duckdb — ingest lands files, "
               "a batch load folds them in.", size=10.5, fill=MUTED)

text(28, 917, "generated by docs/make_architecture.py", size=10, fill=FAINT)
add("</svg>")

path = Path(__file__).with_name("architecture.svg")
path.write_text("\n".join(out))
print(f"wrote {path} ({path.stat().st_size:,} bytes)")
