# Model weights

This directory holds the two detector checkpoints. Neither is committed to git
(see `.gitignore`) because model binaries do not belong in source control.

```
models/
├── yolov8n.pt                     # vehicle detector (auto-downloaded)
└── license_plate_detector.pt      # plate detector (you supply this)
```

---

## 1. Vehicle detector — automatic

Nothing to do. Ultralytics downloads `yolov8n.pt` on first use. It is
COCO-pretrained, and COCO already contains the classes this project needs:

| COCO id | class      |
|---------|------------|
| 2       | car        |
| 3       | motorcycle |
| 5       | bus        |
| 7       | truck      |

Swap in a larger checkpoint (`yolov8s.pt`, `yolov8m.pt`, `yolo11n.pt`) for
better recall at greater cost — set it in the sidebar or via
`ANPR_VEHICLE_MODEL=yolov8s.pt`.

---

## 2. License-plate detector — you supply this

COCO has no "license plate" class, so this stage needs a dedicated model. Put
any single-class YOLO detector here and the pipeline will use it:

```
models/license_plate_detector.pt
```

Or point somewhere else without touching code:

```bash
export ANPR_PLATE_MODEL=/path/to/your/plate_model.pt
```

### Where to get one

| Option | Notes |
|--------|-------|
| Pretrained community weights | Several YOLOv8 license-plate detectors are published on GitHub and Hugging Face. Check the licence before using one in anything but a demo. |
| Roboflow Universe | Search "license plate detection"; many datasets include exported YOLOv8 weights. Indian-plate datasets exist and transfer better to Indian traffic. |
| Train your own | Best results, and the most defensible choice in an interview. Recipe below. |

### Training your own (≈30–60 min on a free Colab T4)

Dataset layout in YOLO format:

```
plates/
├── data.yaml
├── train/images/  train/labels/
└── valid/images/  valid/labels/
```

`data.yaml`:

```yaml
train: train/images
val: valid/images
nc: 1
names: ["license_plate"]
```

Each label file holds one line per plate, normalised to the image size:

```
0 x_center y_center width height
```

Train:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")          # start from COCO weights, not scratch
model.train(
    data="plates/data.yaml",
    epochs=60,
    imgsz=640,
    batch=16,
    patience=15,                    # early stopping
    name="plate_detector",
)
```

Then copy `runs/detect/plate_detector/weights/best.pt` to
`models/license_plate_detector.pt`.

**Why fine-tune rather than train from scratch.** The early convolutional layers
of a COCO-pretrained network already encode edges, corners and texture — features
that are just as useful for plates as for the 80 COCO classes. Fine-tuning reuses
them and only has to learn the plate-specific decision boundary, which is why a
few thousand images and an hour of GPU time are enough. Training from random
initialisation on the same data would badly overfit.

**What to report after training.** mAP@0.5 and mAP@0.5:0.95 on the validation
split, plus precision/recall at your chosen confidence. For ANPR, recall matters
more than precision at this stage: a plate you never detect is a plate you can
never read, whereas a false positive usually dies at the OCR step because no
plausible text comes out of it.

---

## 3. Running without plate weights

The app does **not** invent detections. With no weights it tells you so and
points here.

For a demo without weights you can enable the classical contour proposer:

```bash
export ANPR_PLATE_CLASSICAL_FALLBACK=true
```

Be honest about what this is: an edge-density and aspect-ratio heuristic from
the pre-deep-learning era. It finds some plates on clean, front-on shots and
fails on angles, shadows and clutter. Everything it produces is tagged
`source="classical"`. It exists to keep the pipeline demonstrable and to make
the contrast with a learned detector concrete — not as a substitute for one.
