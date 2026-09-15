# ANPR Vision AI

**Automatic Number Plate Recognition with vehicle tracking and temporal OCR aggregation.**

A complete, local-first ANPR pipeline: it detects vehicles in an image or video,
tracks each one across frames, locates and rectifies its number plate, reads the
plate through several preprocessing variants, validates the result against Indian
registration formats, and fuses readings from many frames into one answer per
vehicle before storing it.

Runs on a normal laptop CPU. Uses CUDA automatically when it is available.

---

## Contents

1. [What makes this more than a tutorial](#what-makes-this-more-than-a-tutorial)
2. [Demo](#demo)
3. [Architecture](#architecture)
4. [The vision pipeline, stage by stage](#the-vision-pipeline-stage-by-stage)
5. [Quick start](#quick-start)
6. [Modes](#modes)
7. [Configuration](#configuration)
8. [Project structure](#project-structure)
9. [Database](#database)
10. [Performance](#performance)
11. [Testing](#testing)
12. [Limitations and honest caveats](#limitations-and-honest-caveats)

---

## What makes this more than a tutorial

Most ANPR demos run OCR on every frame and print whatever comes back. Four things
here address the problems that appear the moment you point this at real traffic:

**Temporal aggregation.** A single frame's OCR is a noisy measurement. Each
tracked vehicle accumulates several readings, which are fused by weighted voting
(confidence × format plausibility). When no single string wins, the aggregator
votes character by character and can reconstruct a plate that *no individual
frame read correctly*.

**Multi-variant preprocessing.** There is no preprocessing recipe that wins on
every plate — CLAHE helps a shadowed plate and hurts a clean one. The pipeline
builds several variants and scores them, turning a fragile assumption into a
cheap search.

**Position-aware character correction.** `O↔0`, `I↔1`, `B↔8` are the standard OCR
confusions, but replacing them globally destroys valid plates. Correction is
applied only where the expected plate layout says a position must be a letter or
a digit, only when it upgrades the match, and every change is recorded and shown.

**OCR is throttled, not run per frame.** OCR is the most expensive stage by an
order of magnitude. Tracking means one vehicle needs a handful of readings across
its time on screen — and those few, aggregated, beat ninety independent ones.

---

## Demo

> Add a short screen recording here once you have run it on your own footage.
>
> ```
> docs/demo.gif
> ```

Annotated output shows, per vehicle:

```
CAR #17
Plate: UP32AB1234
OCR: 94%
```

with the plate box colour-coded by validation status (green = valid format,
amber = suspicious, red = low confidence).

---

## Architecture

```
                         ┌──────────────────────────────────────┐
                         │        Streamlit application         │
                         │  Image · Video · Live · Dashboard    │
                         └───────────────────┬──────────────────┘
                                             │  (UI holds no CV logic)
                         ┌───────────────────▼──────────────────┐
                         │          ANPRPipeline                │
                         │      src/pipeline.py — orchestrator  │
                         └──┬─────────┬──────────┬───────────┬──┘
                            │         │          │           │
          ┌─────────────────▼──┐  ┌───▼──────┐  ┌▼─────────┐ ┌▼──────────────┐
          │    detection/      │  │ tracking/│  │   ocr/   │ │  analytics/   │
          │ vehicle_detector   │  │ ByteTrack│  │ engine   │ │ confidence    │
          │ plate_detector     │  │ Simple   │  │ preproc  │ │ aggregation   │
          └──────────┬─────────┘  └────┬─────┘  │ validator│ │ statistics    │
                     │                 │        └────┬─────┘ └───────┬───────┘
                     └─────────────────┴─────────────┴───────────────┘
                                             │
                         ┌───────────────────▼──────────────────┐
                         │   database/  (SQLite + dedup rules)  │
                         └──────────────────────────────────────┘

          utils/geometry.py   IoU · containment · homography
          utils/video.py      readers, writers, annotation
          config/config.py    every threshold, one place
```

The UI calls the pipeline; the pipeline calls the components. Nothing in `src/`
imports Streamlit, which is why the same code runs from `cli.py` and from tests.

---

## The vision pipeline, stage by stage

```
frame
  → vehicle detection        YOLO, classes {car, motorcycle, bus, truck}
  → tracking                 ByteTrack — persistent IDs across frames
  → plate detection          YOLO, searched inside each vehicle box
  → crop + padding           coordinates clamped to image bounds
  → perspective correction   homography, with fallback to the raw crop
  → preprocessing variants   grayscale · CLAHE · sharpen · threshold · morph
  → OCR                      PaddleOCR on each variant, best one wins
  → normalisation            uppercase, strip punctuation, drop stray glyphs
  → validation               Indian formats, state codes, plausibility score
  → temporal aggregation     weighted voting per track ID
  → storage                  SQLite, duplicate-controlled
  → annotation               boxes, IDs, plate text, confidence, warnings
```

### 1. Vehicle detection

YOLO restricted to four COCO classes. The detector emits a box, a class and a
confidence; overlapping duplicates are removed by **NMS** using an IoU threshold.

- **Confidence** — the detector's score that a box holds an object of that class.
  Raising it trades recall for precision.
- **IoU** — `area(A∩B) / area(A∪B)`, scale-invariant, in `[0,1]`.
- **NMS** — sort by confidence, keep the best box, discard same-class boxes whose
  IoU with it exceeds the threshold. Too low and a genuinely occluded second car
  gets deleted; too high and duplicates survive.
- **Precision / recall** — `TP/(TP+FP)` and `TP/(TP+FN)`. Recall dominates here:
  an undetected vehicle can never have its plate read.

### 2. Tracking

ByteTrack assigns a persistent ID per vehicle. Its trick: instead of discarding
low-confidence detections before association, it runs a *second* association pass
over them. Occluded and blurred objects usually appear as low-confidence
detections rather than as nothing, so that pass recovers exactly the tracks that
would otherwise be lost — at negligible cost.

A dependency-free `SimpleTracker` (greedy IoU + centroid fallback, with track
ageing through occlusions) is included as a readable reference implementation and
is unit-tested.

Terms: **object ID**, **centroid**, **association**, **occlusion** (object hidden;
the track coasts for `max_age` frames), **ID switch** (two objects cross and swap
identities — the classic failure of IoU-only tracking, and the reason aggregation
keeps a per-track history).

### 3. Plate detection

A separate single-class YOLO model, run **inside each vehicle crop**. A plate 60 px
wide in a 1080p frame becomes ~200 px wide once the car is cropped and letterboxed
to the model's 640 px input — far more pixels on target, and fewer false positives
from the background. A whole-frame sweep runs as a safety net.

Plates are matched to vehicles by **containment** (`area(plate ∩ car) / area(plate)`),
not IoU — a plate fully inside a car has containment 1.0 but IoU near 0.02.

### 4. Perspective correction

Two views of a plane are related by a 3×3 **homography** `H`:

```
    ⎡x'⎤       ⎡x⎤
  s ⎢y'⎥ = H · ⎢y⎥        H is defined up to scale → 8 degrees of freedom
    ⎣1 ⎦       ⎣1⎦
```

Each point correspondence contributes two linear equations, so **four**
non-collinear corners determine `H` exactly. The pipeline finds corners by
edge map → contours → polygon approximation, accepts them only if they form a
convex quadrilateral with a plate-like aspect ratio, then warps to a
fronto-parallel rectangle. Characters become upright and evenly spaced, which is
what the recognition network was trained on.

If corners are not reliable, it falls back to the raw crop — a bad homography
built from the edge of a bumper is worse than none.

### 5. Preprocessing

Upscale → grayscale → bilateral denoise → CLAHE → unsharp mask → adaptive
threshold → morphology, exposed as separate variants rather than one fixed chain.

- **CLAHE** equalises contrast within tiles and clips the histogram first, so a
  half-shadowed plate is lifted without amplifying noise (plain histogram
  equalisation blows it out).
- **Bilateral filter** averages neighbours only when they are also similar in
  intensity, so grain goes and character edges stay.
- **Unsharp mask** — `sharp = img + amount·(img − blur(img))`; the difference is
  the high-frequency detail, added back to strengthen strokes.
- **Adaptive threshold** computes a threshold per neighbourhood, handling a
  lighting gradient across the plate that one global threshold cannot.
- **Morphology** — opening kills speckles, closing bridges gaps punched into thin
  strokes by thresholding.

Polarity is normalised to dark-text-on-light, since Indian plates come in both.

### 6. OCR and validation

PaddleOCR reads every variant. The winner is chosen by

```
score = 0.55·ocr_confidence + 0.35·format_score + 0.10·length_bonus
```

so format plausibility acts as a prior — a confidently-read piece of nonsense
loses to a slightly-less-confident string that looks like a registration number.
Multi-line results (double-row plates on two-wheelers) are merged in reading order.

Validation covers `STANDARD` (`UP32AB1234`, plus 9-, 10- and 11-character
variants), `BH_SERIES` (`22BH1234AA`), and relaxed short/no-series layouts, and
identifies the state/UT prefix. Results are classified as:

| Status | Meaning |
|--------|---------|
| `VALID_FORMAT` | Matches a known layout with a recognised state code and adequate confidence |
| `SUSPICIOUS_FORMAT` | Readable but structurally odd — worth a human look |
| `LOW_CONFIDENCE` | The recogniser itself was unsure; treat as unverified |
| `NO_TEXT` | A plate was located but no characters could be read |

### 7. Temporal aggregation

For each track, observations accumulate:

```
frame 100   UP32AB1234   72%
frame 110   UP32AB1234   89%
frame 120   UP32A81234   63%      ← B misread as 8
frame 130   UP32AB1234   94%

→ Final plate:  UP32AB1234
  Confidence:   91%
  Observations: 4      (75% agreement, weighted_vote)
```

Each observation votes with weight `ocr_conf^1.5 × format_weight`. The reported
confidence is the weighted mean of the agreeing observations plus a small,
**capped** consensus bonus — agreement is evidence, but four mediocre reads must
not manufacture 99%.

**Why this works:** every frame samples the same constant string at a different
distance, angle, blur and exposure. The errors are largely independent across
frames while the correct characters are not, so they do not reinforce each other
but agreements do. It is the variance-reduction argument behind ensembling;
reading a plate at twenty metres and at five is close to using two sensors.

---

## Quick start

```bash
git clone <your-repo-url> anpr-vision-ai
cd anpr-vision-ai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Add a plate detector — see models/README.md
#   models/license_plate_detector.pt

streamlit run app.py
```

Check that every component is wired up before a long run:

```bash
python cli.py --check
```

```
[OK ] vehicle_detector: yolov8n.pt
[OK ] plate_detector: license_plate_detector.pt
[OK ] ocr: PaddleOCR (en)
[OK ] device: cpu
```

Headless use:

```bash
python cli.py --image data/input/car.jpg
python cli.py --video data/input/traffic.mp4 --max-frames 500 --frame-skip 2
```

---

## Modes

### Image
Upload one frame. Detects vehicles and plates, reads and validates the text, and
shows the per-variant OCR comparison so you can see *which* preprocessing won and
by how much.

### Video
Full pipeline with tracking, throttled OCR and aggregation. Produces one row per
tracked vehicle, a downloadable annotated video, and an expandable per-track view
of every individual observation behind each final answer.

### Live camera
Reads from a locally attached camera. If none is reachable — which is normal for a
remotely hosted Streamlit app — it says so plainly and points you to the Image and
Video tabs, which run the identical pipeline.

---

## Configuration

Everything lives in `config/config.py`, grouped by pipeline stage, and every value
can be overridden by an `ANPR_`-prefixed environment variable. The sidebar mutates
a runtime copy, so experiments never need a code edit.

| Setting | Default | Effect |
|---------|---------|--------|
| `ANPR_VEHICLE_CONF` | 0.35 | Vehicle detection threshold |
| `ANPR_PLATE_CONF` | 0.25 | Plate detection threshold |
| `ANPR_VEHICLE_IOU` | 0.50 | NMS IoU |
| `ANPR_FRAME_SKIP` | 2 | Process every (N+1)-th frame |
| `ANPR_OCR_INTERVAL` | 8 | Frames between OCR calls per track |
| `ANPR_RESIZE_WIDTH` | 1280 | Processing width (0 = original) |
| `ANPR_DB_COOLDOWN` | 60 | Duplicate-suppression window, seconds |
| `ANPR_DB_MIN_CONF` | 0.45 | Minimum confidence to store |
| `ANPR_TRACKER` | bytetrack | `bytetrack` · `botsort` · `simple` |
| `ANPR_DEVICE` | auto | `auto` · `cpu` · `cuda:0` |
| `ANPR_PLATE_MODEL` | `models/license_plate_detector.pt` | Swap the plate model |

---

## Project structure

```
anpr-vision-ai/
├── app.py                      Streamlit UI (no CV logic)
├── cli.py                      headless runner
├── requirements.txt
├── config/config.py            all thresholds and paths
├── models/README.md            weights setup + training recipe
├── src/
│   ├── pipeline.py             orchestrator
│   ├── detection/
│   │   ├── vehicle_detector.py YOLO + tracking entry points
│   │   └── plate_detector.py   swappable plate model, classical fallback
│   ├── tracking/tracker.py     ByteTrack facade + SimpleTracker
│   ├── ocr/
│   │   ├── ocr_engine.py       PaddleOCR, variant scoring
│   │   ├── preprocessing.py    variants + perspective correction
│   │   └── text_validator.py   normalisation, Indian formats, correction
│   ├── analytics/
│   │   ├── confidence.py       three-signal fusion
│   │   ├── aggregation.py      temporal voting
│   │   └── statistics.py       FPS, per-stage timings, session counters
│   ├── database/database.py    SQLite schema, queries, dedup
│   └── utils/
│       ├── geometry.py         IoU, containment, homography
│       └── video.py            readers, writers, annotation
├── data/{input,output,crops}/
├── docs/                       architecture notes, interview prep
└── tests/                      95 tests, no model weights required
```

---

## Database

```sql
CREATE TABLE detections (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number         TEXT    NOT NULL,
    vehicle_type         TEXT,
    tracking_id          INTEGER,
    ocr_confidence       REAL,
    detection_confidence REAL,
    validation_status    TEXT,
    observations         INTEGER,
    timestamp            TEXT    NOT NULL,
    source               TEXT,
    image_path           TEXT,
    session_id           TEXT,
    notes                TEXT
);
```

Indexed on `plate_number`, `timestamp`, and `(session_id, tracking_id)`.

**Duplicate control.** A vehicle in frame for ten seconds would otherwise write a
row per OCR call. A write is suppressed when the same plate — or the same track
within the same session — was stored inside the cooldown window, *unless* the new
reading is better by at least the improvement margin, in which case the improved
reading is recorded. One sighting becomes one row, and the stored confidence still
improves as the vehicle gets closer.

---

## Performance

Indicative, 720p dashcam footage, 4-core laptop CPU, no GPU:

| Setting | Effective FPS | Note |
|---------|---------------|------|
| Every frame, OCR every frame | ~1–2 | The naive baseline |
| `frame_skip=2`, `ocr_interval=8` | ~6–9 | Default; no measurable accuracy loss |
| `frame_skip=2`, `ocr_interval=8`, CUDA | ~25–40 | GPU detection, CPU OCR |

The HUD burned into the output video and the dashboard both read from
`PerformanceMonitor`, so they cannot disagree: FPS, frames processed, average OCR
time, average detection time.

Why the throttling is nearly free: at 25 fps a vehicle barely moves between
adjacent frames, and ByteTrack maintains identity fine at ~10 fps effective. The
information lost by skipping is redundant; the cost saved is not.

---

## Testing

```bash
pytest -q          # 95 tests, ~0.3s, no model weights needed
```

Coverage focuses on the logic that is easy to get subtly wrong:

- `test_validator.py` — formats, scoring, and that correction is *position-aware*
  (an `O` in a letter slot must survive; an `I` in a digit slot must not)
- `test_aggregation.py` — weighted voting, the capped consensus bonus, character-
  level reconstruction, confidence fusion, IoU vs containment, tracker ID
  persistence through occlusion
- `test_database.py` — schema, queries, and every duplicate-control rule
- `test_ocr.py` — normalisation, and tolerance of both PaddleOCR 2.x and 3.x
  output shapes

---

## Limitations and honest caveats

- **This is visual format validation only.** It reports that a string *looks like*
  an Indian registration number. It cannot tell you whether a vehicle is
  registered, insured, taxed or road-legal — that needs an authoritative
  government database, not a camera.
- **Character correction is a heuristic** and can introduce its own errors. Every
  correction is recorded and surfaced in the UI rather than applied silently.
- **Accuracy degrades** with motion blur, night glare, heavy rain, steep angles,
  dirty or damaged plates, and any plate below roughly 60 px wide in the frame.
- **ID switches happen** when vehicles cross or occlude each other; an ID switch
  can contaminate a track's aggregated reading.
- **Non-standard plates** — vanity, military, diplomatic and older state formats —
  will often be flagged `SUSPICIOUS_FORMAT`. That is the honest answer, not a bug.
- **Deployment of ANPR raises real privacy questions.** Plate data is personal
  data in many jurisdictions. Retention limits, access control and a lawful basis
  for processing are part of building this responsibly, not an afterthought.

---

## Licence

Add your preferred licence before publishing. Check the licence of any pretrained
weights you ship or link to — they carry their own terms.
