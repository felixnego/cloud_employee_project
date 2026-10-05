# Creative feature extraction — a data layer for ad testing

Turns video creatives into a queryable dataset.

Five ads go in as MP4 files; what comes out is a DuckDB warehouse plus Parquet
files describing what happens in each ad, second by second — what is on screen,
who is speaking, what is being said, how fast it is cutting, what the soundtrack
is doing. An analyst with SQL and a bit of Python can answer *"what was on screen
at 0:14?"* without opening a video player.

This is the **creative side** of an ad-testing dataset. A second, clearly
separated layer adds a **synthetic audience panel** — biometrics, demographics,
survey responses and an Implicit Association Test — generated from the measured
creative features, so the join the schema was shaped around can actually be
exercised. See [The audience layer](#the-audience-layer-synthetic).

```bash
# the five source MP4s are committed in video_data/
make build       # build the image (~5 min first time: deps + 681MB of models)
make all         # extract -> transform -> docs  (~4.5 min cold, seconds when cached)
make audience    # generate + load + validate the synthetic panel (~20s)
make notebook    # JupyterLab on http://localhost:8888
```

Nothing to install but Docker. No API keys, no paid services, no network access
after the build.

**All of that is skippable.** `warehouse/marts/` is committed — 1.9MB of
Parquet and CSV covering every table, creative and audience alike. To read the
dataset without Docker or a build:

```python
import pandas as pd
pd.read_parquet("warehouse/marts/creative_seconds.parquet")   # the creative spine
pd.read_csv("warehouse/marts/creative_summary.csv")           # one row per ad
pd.read_parquet("warehouse/marts/viewer_seconds.parquet")     # the joined table
```

| Where to look | What is there |
|---|---|
| [`notebooks/01_creative_data_overview.ipynb`](notebooks/01_creative_data_overview.ipynb) | **Start here.** Verification that it ran, plus a first read of the data. Rendered with outputs. |
| [`docs/schema.md`](docs/schema.md) | Every table, grain, key and column — generated from the warehouse, so it cannot drift |
| [`transforms/*.sql`](transforms/) | The SQL that builds each mart, in order |
| [`pipeline/extractors/`](pipeline/extractors/) | One file per feature, all behind the same interface |
| [`audience/contract.py`](audience/contract.py) | The audience data contract a real panel ingest would satisfy |
| `warehouse/marts/*.parquet` | The output, readable without this repo |

---

## The nine features

Each is one file in `pipeline/extractors/`. All models are ONNX or CTranslate2,
run on CPU, and are baked into the image.

| # | Feature | What it provides | Tool |
|---|---------|-------------------|------|
| 1 | **Metadata** | Duration, resolution, orientation, frame rate, codecs, audio stream presence. Establishes that a vertical 576×1024 UGC clip and a 1280×720 broadcast film are different objects before any comparison. | ffprobe |
| 2 | **Shots** | Boundaries of every cut, and the length of each shot. This is *pacing* — the difference between a 57-shot film and a 2-shot talking head. | PySceneDetect |
| 3 | **Transcript** | Everything spoken, with timestamps at both utterance and individual-word level. Locates a brand mention or a price to the half-second. | faster-whisper `small` |
| 4 | **Visual** | Per-frame brightness, saturation, contrast, colourfulness, sharpness, edge density, and frame-to-frame motion. The raw texture of the picture, and the basis for "when did it get visually busy?". | OpenCV |
| 5 | **Audio** | Per-frame loudness (dBFS), spectral shape, onset strength, harmonic/percussive balance, plus global tempo. Feeds the speech/music/silence classification. | librosa |
| 6 | **On-screen text** | Every piece of text burned into the picture — taglines, prices, captions, logotype — with its size and position, consolidated into time spans. Text height separates a headline from a legal line. | RapidOCR (PP-OCRv4) |
| 7 | **Faces & expressions** | Every face on screen, its size and position, and an 8-class expression (neutral, happiness, surprise, sadness, anger, disgust, fear, contempt). Face size is a decent proxy for intimacy of framing. | YuNet + FER+ |
| 8 | **Objects & people** | COCO-80 detections with on-screen size — how many people are in frame, is there a product, a phone, a car. | YOLOX-nano |
| 9 | **Semantic shot labels** | Zero-shot labels per shot on four axes: shot scale (close-up → wide), setting (indoor/outdoor/studio), subject (product, single person, crowd, hands, place), style (UGC selfie, cinematic, documentary, animation). This is what makes the dataset readable by someone who has never watched the ads. | CLIP ViT-B/32 |

Adding a tenth is one file plus one line in `pipeline/registry.py`. The list of
features is configuration, not architecture.

---

## Architecture

```
video_data/*.mp4
     │
     │  prepare: decode ONCE into a content-hashed cache
     ▼
  cache/<sha>/frames/*.jpg  (4 Hz)  +  audio.wav (16 kHz mono)
     │
     │  nine extractors, same interface, independent
     ▼
  warehouse/raw/<table>/<creative_id>.parquet          ← one file per extractor per creative
     │
     │  transforms/*.sql, in filename order
     ▼
  warehouse/warehouse.duckdb                           ← analysts query this
     │
     ▼
  warehouse/marts/*.parquet + .csv                     ← everything else reads this
```

**There is no database server.** The workload is columnar, read-heavy,
single-writer, and five files wide. Parquet on disk plus DuckDB as the engine
gives zero ops, no migrations, no connection strings — and DuckDB is the local
twin of MotherDuck, so hosting this has a one-line answer. The
trade-off is real and stated: DuckDB is single-writer and not multi-tenant. The
moment concurrent writers or a shared BI layer are needed, it is MotherDuck (same
SQL, same files) or Postgres.

**Decode once.** Five of the nine extractors need frames and two need audio.
Decoding per extractor would demux each video seven times. `prepare` writes a
frame cache and a WAV keyed by the video's sha256, and every extractor reads from
it. Cold run: 263s for five creatives.

**Re-running is free.** A manifest keyed on `(video sha256, extractor name,
extractor version)` means unchanged work is skipped. Bumping one extractor's
`VERSION` invalidates only that extractor — when the CLIP taxonomy changed, eight
extractors stayed cached and `labels` re-ran in 6 seconds instead of 263.

**One failure does not cost the run.** An extractor that throws is recorded
in the manifest with its error and the rest continue. A broken OCR model should
not lose the transcript.

---

## The data layer


| Table | Grain | Why it exists |
|---|---|---|
| `creative_summary` | one row per ad | The table to open first. Cross-ad comparison as a `SELECT *`. |
| `creatives` | one row per ad | File facts plus whole-ad rollups. |
| `shots` | one row per shot | The unit a creative team thinks in, carrying CLIP labels and what was in the shot. |
| **`creative_seconds`** | **one row per (ad, second)** | **The join target.** Everything time-varying, at 1 Hz. |
| `creative_frames` | one row per (ad, frame), 4 Hz | The native grain, for sub-second events. |
| `transcript_segments` / `_words` | utterance / word | The script, and word-level timings. |
| `screen_text` | one text span | On-screen copy as spans, not per-frame detections. |
| `face_detections` / `object_detections` / `shot_label_scores` | one detection / one label | Full detail, including non-winning labels and full probability distributions. |
| `run_manifest` / `sampling_config` | one run / one extractor | Provenance and the sampling rates actually used. |

Full column-level documentation is in [`docs/schema.md`](docs/schema.md).

### Two details that matter

**4 Hz is the native grain, seconds are the published one.** These are fast-cut
social ads; at 1 Hz a 400 ms product reveal disappears. Aggregating up is always
possible, aggregating down is not — so extraction happens at 4 Hz and `creative_seconds` is the
friendly rollup.

**NULL and zero mean different things.** YOLOX samples at 2 Hz and OCR at 1 Hz,
so `n_objects` is `NULL` on frames the model never looked at and `0` on frames it
looked at and found nothing. The `objects_sampled` / `ocr_sampled` flags make that
explicit rather than leaving it to be rediscovered by whoever writes the first
`AVG()`. `sampling_config` records the rates, so the warehouse describes its own
resolution.

---

## The audience layer (synthetic)

The creative layer is measured from real files. The audience layer is
**generated**, and the two are kept structurally apart:

| Schema | Contents | Provenance |
|---|---|---|
| `main` | creatives, shots, `creative_seconds`, transcripts … | measured from real video |
| `audience` | viewers, sessions, biometrics, survey, IAT | **synthetic** |
| `analysis` | the joined marts | mixed, and labelled as such |

Separation is structural rather than a naming convention, because the worst
failure mode for a project like this is a synthetic number being read as a real
one. Every audience row additionally carries `data_source = 'synthetic_v1'`.

### A contract, not just a script

`audience/contract.py` defines seven tables — grain, key, purpose, and the places
each will bite. The generator is one *producer* of that contract; a real ingest
from a biometric vendor, a survey platform and an IAT tool would be another,
writing the same tables. Nothing downstream imports the generator, so migrating
to real data means deleting `audience/` and pointing an ingest at the same
tables. The dependency runs one way — `audience` imports `pipeline`, never the
reverse.

### The response is caused by the creative

Random numbers would exercise nothing. Attention at each second is a logistic
function of what the ad is doing at that second, read from `creative_seconds`:

```
attention ~ logistic( β_pos·position_in_ad + β_cut·n_cuts + β_face·has_face
                    + β_speech·has_speech + β_text·has_screen_text
                    + β_motion·motion_z + viewer_effect + AR(1) noise )
```

Expression valence follows on-screen affect and the viewer's implicit preference;
abandonment is a hazard that rises as attention falls; survey answers follow from
what happened during the exposure. The IAT runs in the correct causal direction —
a latent preference causes the reaction times, and the D-score recovers it
imperfectly, as a real IAT would.

Some behaviour emerges rather than being written down: completion rate falls with
ad duration (65% on the 33-second clip, 14% on the 147-second film), and attention
declines monotonically across every runtime.

### Ground truth is published

`audience.generator_params` holds the coefficients that produced the panel. This
is the advantage synthetic data has over real data: an analysis can be *checked*.
`python -m audience.run validate` refits the attention model by OLS and confirms
the planted coefficients come back out, alongside integrity checks:

```
PASS  recovers beta[cut_density]                   true +0.45  estimated +0.44
PASS  recovers beta[has_face]                      true +0.35  estimated +0.36
PASS  IAT D-score tracks the latent trait          corr = 0.568
PASS  emotions NULL exactly when no face detected
14/14 checks passed
```

Runs are reproducible: a fixed seed reproduces the panel exactly.

### The join

`analysis.viewer_seconds` is the table the schema was designed to produce — one
row per viewer-second, every predictor attached:

```sql
SELECT v.second, v.attention_index,
       s.audio_class, s.n_cuts, s.max_faces, s.modal_face_emotion
FROM audience.biometric_seconds v
JOIN main.creative_seconds s USING (creative_id, second);
```

No interpolation, no window functions, no per-analyst resampling. *"Attention
dipped at 0:14 — what was on screen, who was speaking, how fast was it cutting?"*
is one query.

### What is deliberately modelled

- **Consent.** A few panelists withhold biometric consent and have no biometric
  rows. The real pipeline must honour that, so the synthetic one does.
- **Missing signal.** When the webcam loses the face, emotion and valence are
  `NULL`, not zero — the same discipline the creative layer uses.
- **Incomplete design.** Not every viewer sees every creative, as in a real panel.
- **Response style.** Some panelists answer consistently high; acquiescence is a
  viewer trait, not noise.

---

## Tested and rejected

Measured decisions, not preferences.

**PyTorch.** The obvious stack (`ultralytics` for YOLO, Py-Feat or DeepFace for
expressions, EasyOCR for text) pulls in PyTorch: ~2.5GB on top of the image, and
several pieces have no native arm64 wheels. Every one of those has an ONNX
equivalent — YOLOX-nano, FER+, RapidOCR — so the whole stack runs on ONNX Runtime
at 2.84GB total, natively on Apple Silicon.

**YOLOv8 → YOLOX-nano.** Every public YOLOv8 ONNX export sits behind an
auth-gated Hugging Face repo (verified: 401). YOLOX-nano from the OpenCV Zoo is
COCO-80, 34MB, and downloads from a stable URL. The decode (grid/stride, NMS) is
hand-written in `pipeline/models/yolox.py` because the export ships raw tensors.

**MediaPipe → YuNet + FER+.** MediaPipe publishes no linux/aarch64 wheel, and its
blendshapes are muscle activations rather than emotions. YuNet + FER+ is ~35MB,
arm64-native, and returns the eight *named* categories an analyst can reason about.

**A "title card" CLIP label — removed.** Any prompt mentioning a text or title
card absorbs almost every frame regardless of wording (tested four phrasings;
a stadium crowd scored 0.85 as "a screenshot of text on a plain background").
CLIP ViT-B/32 has a very strong prior toward text-card prompts. "Is this a title
card?" is answered far better by the OCR feature.

**A CLIP "mood" axis — removed.** It never identified an upbeat beauty ad as
upbeat under any wording tried, and scored it 0.76 "sombre". Single-frame affect
is beyond this model. A confidently-wrong mood label in a dataset meant to be
correlated against biometrics is worse than no mood label. Mood is better
approached through signals that *are* measured well — `audio_class`, tempo, face
expression, cut rate.

**CLIP prompt wording, generally.** Concrete "a photo taken outside in a street,
field or stadium" beats abstract "an outdoor scene in the open air". Rewording the
`setting` axis moved Norwich City from *studio* (16/23 shots, conf 0.53) to
*outdoor* (20/23, conf 0.75) — the correct answer, confirmed by looking at the
frames.

**dbt.** For eight models with no cross-project lineage needs, a 30-line runner
(`pipeline/transform.py`) keeps the stack to one dependency and the SQL plain
enough to paste into a notebook. dbt-duckdb is the upgrade path once there are
enough models to need tests and docs; the SQL ports as-is.

---

## Known limitations

Things that are wrong or weak based on cursory research. Points to bring up to the Subject Matter Experts in video analysis. 

- **FER+ is heavily neutral-biased** — 1,421 of 1,789 face detections read
  `neutral`. Treat it as "nothing strong detected", not as a finding.
- **These are the expressions of actors in the ad**, which is a different
  construct from a viewer's expression. The same model class would run on webcam
  footage; that is the viewer-side pipeline, not this one.
- **COCO has no class for "lip oil"** — a lip-oil wand is confidently detected as
  `toothbrush`. Object detection here is coarse scene composition, not product
  recognition.
- **OCR struggles on stylised type over moving video.** "TikTok" scores 0.96;
  decorative overlay text degrades to `erOM SEAIAEK`. Filter on `max_score`.
- **Letterboxing is not detected.** One creative carries black bars, which drags
  its brightness and contrast down relative to the others. Cross-creative
  comparisons of those columns are mildly unfair; `ffmpeg cropdetect` would fix it.
- **`audio_class` music-vs-ambient is a threshold, not a model.** Speech is
  model-grounded (Whisper + Silero VAD); the music boundary is two constants at
  the top of `transforms/07_creative_seconds.sql`, visible so they can be argued
  with. The classes separate cleanly on this sample (music: harmonic ratio 0.71,
  flatness 0.008; speech: 0.32).
- **Whisper transcribes five English-language ads.** Nothing here is tuned for
  other languages, though `language` and `language_probability` are recorded.
- **Five creatives is not a sample.** Every cross-creative number is a
  description of these five files, not evidence about advertising.
- **The audience panel is synthetic and proves nothing about these ads.** It
  demonstrates that the schema, the join and the analysis path work. Any
  difference between creatives in a response column is a generator artefact.
- **The generator cannot validate its own assumptions.** It confirms that an
  analysis recovers the coefficients planted in it. Whether those coefficients
  resemble real human attention is a question only real panel data can answer.
- **Expression inference is scientifically contested.** That discrete emotions
  are reliably readable from faces is disputed (Barrett et al., 2019). Both the
  creative-side `face_detections` and the synthetic viewer expressions inherit
  that caveat, and real deployment needs explicit consent and retention limits
  beyond the consent flag modelled here.

---

## Repo layout

```
video_data/              the five source MP4s (inputs, mounted read-only)
pipeline/
  config.py              every path, sampling rate and threshold
  io.py                  discovery, content hashing, Parquet writes, manifest
  prepare.py             decode-once cache (frames + audio)
  registry.py            the ordered list of nine extractors
  run.py                 CLI; `extract_one` is the integration seam
  transform.py           runs transforms/*.sql against DuckDB
  docgen.py              generates docs/schema.md from the live warehouse
  extractors/            one file per feature
  models/                ONNX wrappers (YuNet, FER+, YOLOX, CLIP)
audience/
  config.py              panel size, seed, and the ground-truth coefficients
  contract.py            the seven-table contract a real ingest would satisfy
  generate.py            the generator (creative features -> viewer response)
  validate.py            integrity + coefficient-recovery checks
  run.py                 CLI
transforms/              numbered SQL, raw views -> dims -> spine -> marts
transforms_audience/     SQL building the audience and analysis schemas
notebooks/               the overview notebook
docs/schema.md           generated schema reference
warehouse/               generated: raw/, marts/, warehouse.duckdb
cache/                   generated: decoded frames and audio
```

### CLI

```bash
docker compose run --rm pipeline list                          # features + creatives
docker compose run --rm pipeline extract --only visual,audio   # a subset
docker compose run --rm pipeline extract --creative norwich    # one creative
docker compose run --rm pipeline extract --force               # ignore the manifest
docker compose run --rm pipeline transform                     # rebuild marts only

docker compose run --rm --entrypoint python pipeline -m audience.run all
docker compose run --rm --entrypoint python pipeline -m audience.run generate --seed 7 --n-viewers 800
docker compose run --rm --entrypoint python pipeline -m audience.run validate
```

