# Interview notes

Prep material for defending this project. Answer in your own words — if a phrase
here does not feel like something you would say, rewrite it before you use it.

---

## The 60-second walkthrough

> "It's an ANPR system. A frame goes through vehicle detection with YOLO, then
> ByteTrack gives each vehicle a persistent ID. I run a separate plate detector
> *inside* each vehicle crop so the plate gets far more pixels at the model's
> input size. The crop gets perspective-corrected with a homography when I can
> find four reliable corners, then I build several preprocessing variants —
> CLAHE, sharpened, thresholded — and run PaddleOCR on each, picking the winner
> by a score that blends OCR confidence with how plausible the string is as an
> Indian registration number.
>
> The part I'd point to first is temporal aggregation. I don't trust any single
> frame. Each track accumulates readings and I fuse them by weighted voting. If
> no single string wins, I vote character by character, which can reconstruct a
> plate no individual frame read correctly. Then duplicate-controlled writes into
> SQLite, and a Streamlit dashboard on top."

Then stop and let them pick a thread.

---

## Questions you should expect

### "Why not just run OCR on every frame?"

Cost and accuracy, in that order. OCR is the most expensive stage by roughly an
order of magnitude — one batched detector pass covers a whole frame, but OCR runs
per plate crop. With tracking I only need a handful of readings per vehicle across
its time on screen. And those few, aggregated, are *more* reliable than ninety
independent ones, because reliability comes from fusing observations, not from
repeating them. `OCR_INTERVAL` controls it directly.

### "Why search for plates inside the vehicle box instead of the whole frame?"

Resolution. A plate might be 60 px wide in a 1080p frame. YOLO letterboxes its
input to 640 px, so on the full frame that plate is down to ~20 px — below what
the detector can reliably fire on. Crop the car first and the same plate arrives
at ~200 px. It also cuts false positives, because road signs and shop hoardings
are outside the search region. I keep a whole-frame sweep as a fallback for
vehicles clipped by the frame edge.

### "How do you associate a plate with a vehicle?"

Containment, not IoU: `area(plate ∩ car) / area(plate)`. A plate fully inside a
car box has containment 1.0 but IoU around 0.02, because IoU punishes the size
difference. When two vehicle boxes overlap, I break the tie toward the smaller
box — the nearer vehicle is the more plausible owner of the plate.

### "Explain IoU and NMS."

IoU is intersection over union — scale-invariant overlap in `[0,1]`. NMS uses it
to remove duplicates: sort boxes by confidence, keep the top one, discard any
same-class box whose IoU with it exceeds the threshold, repeat. Set it too low
and you delete a genuinely occluded second vehicle; too high and duplicates
survive into tracking, which causes ID churn.

### "Why ByteTrack?"

Most trackers throw away low-confidence detections before association. ByteTrack
keeps them for a second association pass over whatever is left unmatched. That
matters because an occluded or blurred vehicle usually shows up as a
*low-confidence detection* rather than as nothing at all — so the second pass
recovers exactly the tracks that would otherwise break. It costs almost nothing
because it reuses detections you already paid for.

I also wrote a simpler tracker — greedy IoU matching with a centroid fallback and
track ageing — partly as a fallback and partly because implementing it is how I
understood what ByteTrack's Kalman prediction actually buys you.

### "What's an ID switch and why do you care?"

Two vehicles cross, and the tracker swaps their identities. It's the classic
failure of IoU-only association. I care because aggregation is keyed on track ID —
a switch mid-track means readings from two different plates get pooled. Symptom:
a track with high observation count but low agreement. That's visible in the
aggregation detail table in the UI.

### "Walk me through the homography."

Two views of a plane are related by a 3×3 matrix defined up to scale, so 8 degrees
of freedom. Each point correspondence gives two equations, so four non-collinear
corners determine it exactly — `cv2.getPerspectiveTransform` then
`cv2.warpPerspective`. I find corners with Canny → contours → `approxPolyDP`, and
I only accept a convex quadrilateral with a plate-like aspect ratio covering
enough of the crop.

The important part is the fallback. If the corners aren't trustworthy I use the
unrectified crop, because a homography built from the edge of a bumper warps the
plate into garbage — worse than not correcting at all.

### "Why multiple preprocessing variants? Isn't that wasteful?"

It's a deliberate trade. There's no single recipe that wins on every plate: CLAHE
rescues a shadowed plate and washes out a clean one; thresholding helps a cluttered
background and destroys a low-contrast one. Rather than guess, I run four or five
variants and let the scoring pick. It's a small, bounded search instead of a
fragile assumption. I also short-circuit: if a variant returns a strictly valid
format above 90% confidence, I skip the rest.

### "Explain CLAHE."

Contrast Limited Adaptive Histogram Equalisation. Global histogram equalisation
stretches contrast across the whole image, which blows out a plate that's half in
shadow. CLAHE works on small tiles so contrast is equalised locally, and it clips
the histogram before redistributing it so it doesn't amplify noise in flat regions.

### "Your character correction — isn't O→0 going to break valid plates?"

Yes, if you do it globally, which is exactly why I don't. Correction is
position-aware: I know the expected layout, say `AA99AA9999`, so digit→letter
mappings only apply in letter slots and letter→digit only in digit slots. I only
attempt it when the raw string doesn't already match a strict pattern, and I only
accept the correction if it *upgrades* the match. Every change is recorded and
displayed. It's still a heuristic and it can be wrong — I say so in the UI rather
than hiding it.

### "Why does aggregation actually improve accuracy? Convince me."

Each frame is a noisy measurement of a constant underlying string, taken at a
different distance, angle, blur level and exposure. The errors are largely
independent across frames; the correct characters are not. So errors don't
reinforce each other, agreements do. It's the same variance-reduction argument as
ensembling. Reading the plate at twenty metres and at five metres is close to
having two sensors.

Concretely: three frames say `UP32AB1234` at 72/89/94%, one says `UP32A81234` at
63%. Weighted voting gives the right answer and the wrong read barely registers.
The character-level fallback goes further — with five frames each wrong in a
*different* position, per-position plurality recovers the correct plate even
though no frame was fully right.

### "Why cap the consensus bonus?"

Because agreement is evidence, not proof. Four readings at 70% that agree deserve
slightly more than 70% — but if the bonus were uncapped, twenty mediocre readings
would report 99%, which would be a lie. It's capped at 12 percentage points.

### "How do you prevent one car generating hundreds of database rows?"

Two layers. In video mode, aggregation already collapses a track into one answer,
written once at the end. At the database layer, a write is suppressed if the same
plate — or the same track within the session — was stored inside the cooldown
window, unless the new reading beats the stored one by the improvement margin. So
one sighting is one row, but the stored confidence can still improve as the
vehicle approaches. Both the cooldown and the track query are time-bounded.

### "Why SQLite?"

It fits the workload: a single writer, modest volume, and access patterns — insert
on detection, range scan by timestamp, exact lookup by plate — served by two
indexes. Zero setup, and the database is one file I can ship as a demo fixture. If
this went multi-camera with concurrent writers I'd move to PostgreSQL; the DAO is
a thin layer specifically so that swap is contained.

### "What are the failure modes?"

Motion blur at speed, night-time glare and headlight bloom, heavy rain, steep
angles beyond what the homography recovers, dirty or damaged plates, and anything
below roughly 60 px wide. Non-standard formats — vanity, military, diplomatic,
older state layouts — get flagged `SUSPICIOUS_FORMAT`, which is the honest answer
rather than a false positive. And ID switches contaminate aggregation.

### "How would you productionise this?"

- Replace the per-frame Python loop with batched GPU inference and decouple
  capture, inference and OCR into queues, so a slow OCR doesn't stall decoding.
- Export detectors to TensorRT or ONNX Runtime; that's usually a 2–4× win.
- Replace SQLite with PostgreSQL and move crops to object storage.
- Add a calibration step per camera — a fixed camera has a known ground plane, so
  the homography can be pre-computed instead of recovered per crop.
- Log every low-confidence reading for human review and use that as a labelling
  pipeline to fine-tune both the detector and the recogniser on *your* camera's
  distribution. That's usually worth more than a bigger model.
- Add retention limits and access control. Plate data is personal data.

### "What would you do differently with more time?"

Train a dedicated plate-recognition head rather than using general-purpose OCR —
plates are a constrained alphabet with a known layout, so a small CRNN trained on
plate crops would beat PaddleOCR and be much faster. I'd also add a proper
evaluation harness: character error rate and plate-level accuracy on a held-out
set, sliced by distance and lighting, so improvements are measured rather than
guessed at.

---

## Things to be honest about

Do not oversell. Interviewers respect calibration, and claiming more than the
system does is the fastest way to lose a room.

- Format validation is **visual only**. It never verifies registration.
- Correction is heuristic and can introduce errors.
- If you have not evaluated on a labelled dataset, say so: "I haven't measured CER
  on a held-out set yet — that's the next thing I'd build."
- If you used pretrained plate weights rather than training your own, say that
  plainly and explain the training recipe you *would* use (`models/README.md`).

---

## Numbers worth remembering

| Thing | Value |
|-------|-------|
| COCO classes used | car 2, motorcycle 3, bus 5, truck 7 |
| Homography DoF | 8 (3×3 up to scale), 4 point correspondences |
| Default frame skip / OCR interval | 2 / 8 |
| Variant score | 0.55 OCR + 0.35 format + 0.10 length |
| Combined confidence | 0.60 OCR + 0.15 detection + 0.25 format |
| Aggregation weight | `ocr_conf^1.5 × format_weight` |
| Consensus bonus cap | 0.12 |
| Standard Indian format | `XX00XX0000` — state(2) + RTO(1–2) + series(1–3) + number(4) |

---

## Before the interview

1. `python cli.py --check` — confirm every component loads.
2. Run one image and one 20-second video end to end so the dashboard has data.
3. `pytest -q` — be able to say "95 tests, and here's the one that proves
   correction is position-aware."
4. Open `src/analytics/aggregation.py` and reread `aggregate()`. It's the function
   most worth being able to explain line by line.
5. Have one failure case ready to show. Being able to say "here's where it breaks
   and here's why" is stronger than a demo that only ever succeeds.
