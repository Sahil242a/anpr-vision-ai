"""
Geometry helpers shared by detection, tracking and perspective correction.

Vocabulary used across the codebase (worth knowing for the interview):

* **Box** — axis-aligned rectangle in pixel coordinates, ``(x1, y1, x2, y2)``,
  top-left and bottom-right, in the coordinate frame of the *full frame*.
* **IoU (Intersection over Union)** — ``area(A ∩ B) / area(A ∪ B)``. Scale
  invariant, in ``[0, 1]``. Used for NMS, for detection/track association and
  for matching plates to vehicles.
* **Containment** — ``area(A ∩ B) / area(A)``. Unlike IoU this does not punish
  a large size difference, which is exactly what we want when asking "is this
  small plate inside this big car?". A plate that sits fully inside a car has
  containment 1.0 but an IoU of maybe 0.02.
* **Centroid** — box centre. Cheap association signal and the basis of the
  fallback tracker.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# Basic box maths
# --------------------------------------------------------------------------- #


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    """Intersection over Union of two boxes."""
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def containment(inner: Box, outer: Box) -> float:
    """Fraction of ``inner`` that lies inside ``outer`` (in ``[0, 1]``)."""
    a = box_area(inner)
    if a <= 0:
        return 0.0
    return float(intersection_area(inner, outer) / a)


def centroid(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(a: Box, b: Box) -> float:
    ax, ay = centroid(a)
    bx, by = centroid(b)
    return math.hypot(ax - bx, ay - by)


def clamp_box(box: Box, width: int, height: int) -> Box:
    """Clamp a box to image bounds so crops never index outside the array."""
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), width - 1.0))
    y1 = max(0.0, min(float(y1), height - 1.0))
    x2 = max(0.0, min(float(x2), float(width)))
    y2 = max(0.0, min(float(y2), float(height)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def expand_box(box: Box, padding: float, width: int, height: int) -> Box:
    """Grow a box by ``padding`` (fraction of its own size), then clamp.

    Plate detectors often crop tight against the characters; a few percent of
    padding keeps ascenders/descenders and the plate border, which measurably
    helps OCR.
    """
    x1, y1, x2, y2 = box
    pw = (x2 - x1) * padding
    ph = (y2 - y1) * padding
    return clamp_box((x1 - pw, y1 - ph, x2 + pw, y2 + ph), width, height)


def to_int_box(box: Box) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def crop(image: np.ndarray, box: Box) -> Optional[np.ndarray]:
    """Safely crop a box from an image. Returns ``None`` for degenerate boxes."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = to_int_box(clamp_box(box, w, h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    patch = image[y1:y2, x1:x2]
    return patch if patch.size else None


# --------------------------------------------------------------------------- #
# Association
# --------------------------------------------------------------------------- #


def associate_plate_to_vehicle(
    plate_box: Box,
    vehicles: Sequence,
    min_containment: float = 0.55,
) -> Optional[int]:
    """Return the index of the vehicle a plate most likely belongs to.

    Strategy: prefer the vehicle that *contains* the largest fraction of the
    plate. Ties (overlapping vehicles, e.g. a car partly behind a truck) are
    broken by the smaller vehicle box, because the plate of the nearer/smaller
    box is the more plausible owner when one vehicle is occluding another.
    """
    best_idx, best_score, best_area = None, min_containment, float("inf")
    for i, v in enumerate(vehicles):
        vbox = v.box if hasattr(v, "box") else tuple(v)
        score = containment(plate_box, vbox)
        area = box_area(vbox)
        if score > best_score or (abs(score - best_score) < 1e-6 and area < best_area):
            if score >= min_containment:
                best_idx, best_score, best_area = i, score, area
    return best_idx


def greedy_iou_match(
    tracks: Sequence[Box],
    detections: Sequence[Box],
    iou_threshold: float = 0.3,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy IoU matching used by the fallback tracker.

    Returns ``(matches, unmatched_track_indices, unmatched_detection_indices)``.
    A greedy pass is used instead of the Hungarian algorithm because with the
    small number of objects in a traffic frame the optimal assignment rarely
    differs, and the greedy version is easy to reason about out loud.
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    pairs = []
    for ti, t in enumerate(tracks):
        for di, d in enumerate(detections):
            score = iou(t, d)
            if score >= iou_threshold:
                pairs.append((score, ti, di))
    pairs.sort(reverse=True)

    matches: List[Tuple[int, int]] = []
    used_t, used_d = set(), set()
    for _, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        matches.append((ti, di))
        used_t.add(ti)
        used_d.add(di)

    unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_d = [i for i in range(len(detections)) if i not in used_d]
    return matches, unmatched_t, unmatched_d


# --------------------------------------------------------------------------- #
# Perspective correction (homography)
# --------------------------------------------------------------------------- #
#
# A plate photographed off-axis is a planar surface seen under a projective
# transform. Two views of the same plane are related by a 3x3 homography H:
#
#       [x']       [x]
#   s * [y'] = H * [y]          with H defined up to scale (8 DoF)
#       [1 ]       [1]
#
# Each point correspondence gives 2 linear equations in the entries of H, so
# 4 non-collinear corners are exactly enough to solve for H. Applying H^-1 to
# the crop "un-slants" the plate into a fronto-parallel rectangle, which makes
# characters upright and uniformly spaced - far easier for the OCR recogniser,
# which is trained mostly on rectified text lines.


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    Uses the classic sum/difference trick: the top-left corner has the smallest
    ``x + y``, the bottom-right the largest; the top-right has the smallest
    ``y - x``, the bottom-left the largest.
    """
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    """Warp the quadrilateral ``pts`` in ``image`` to an upright rectangle."""
    import cv2

    try:
        rect = order_corners(pts)
        (tl, tr, br, bl) = rect
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if width < 12 or height < 8:
            return None
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (width, height))
    except Exception:  # noqa: BLE001 - degenerate corners must not kill a frame
        return None


def find_plate_corners(plate_crop: np.ndarray) -> Optional[np.ndarray]:
    """Try to recover the four corners of the plate inside a rectangular crop.

    Approach: edge map -> external contours -> largest contour -> polygon
    approximation. Accept the result only if it is a convex quadrilateral that
    covers a plausible share of the crop and has a plate-like aspect ratio.
    Anything less certain returns ``None`` and the caller falls back to the
    unrectified crop, which is the safe default.
    """
    import cv2

    if plate_crop is None or plate_crop.size == 0:
        return None
    h, w = plate_crop.shape[:2]
    if h < 16 or w < 32:
        return None

    try:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if plate_crop.ndim == 3 else plate_crop
        gray = cv2.bilateralFilter(gray, 7, 60, 60)
        edges = cv2.Canny(gray, 40, 140)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        crop_area = float(h * w)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(cnt)
            if area < 0.25 * crop_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            rect = order_corners(approx.reshape(4, 2))
            width = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))
            height = max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))
            if height <= 0:
                continue
            aspect = width / height
            if 1.5 <= aspect <= 7.0:  # single- and double-row plates
                return rect
    except Exception:  # noqa: BLE001
        return None
    return None
