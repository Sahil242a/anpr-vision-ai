"""
Video I/O and frame annotation.

Keeping OpenCV drawing code in one place means the image mode, the video mode
and the webcam mode all produce visually identical output, and the annotator
can be unit-tested independently of the models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from config.config import CLASS_COLORS, STATUS_COLORS

logger = logging.getLogger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX


class VideoError(RuntimeError):
    """Raised when a video source cannot be opened or decoded."""


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class VideoReader:
    """Context-managed frame reader that survives corrupted frames.

    A real traffic video will occasionally hand back a decode failure in the
    middle of the stream. Rather than aborting the run we count the consecutive
    failures and only stop once the stream looks genuinely dead.
    """

    def __init__(self, source, max_consecutive_failures: int = 15):
        self.source = source
        self.max_consecutive_failures = max_consecutive_failures
        self.cap: Optional[cv2.VideoCapture] = None
        self.meta: Optional[VideoMeta] = None
        self.corrupted_frames = 0

    def open(self) -> "VideoReader":
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap or not self.cap.isOpened():
            raise VideoError(
                f"Could not open video source '{self.source}'. "
                "Check the file path/codec, or try converting to H.264 MP4."
            )
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.meta = VideoMeta(
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if fps and fps > 0 else 25.0,
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
        return self

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        if self.cap is None:
            self.open()
        idx, failures = 0, 0
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None or frame.size == 0:
                failures += 1
                if failures >= self.max_consecutive_failures:
                    # A trailing run of failures is just the end of the stream,
                    # so it is not reported as corruption. Only failures that
                    # are followed by a successful read were real dropouts.
                    break
                idx += 1
                continue
            if failures:
                self.corrupted_frames += failures
                failures = 0
            yield idx, frame
            idx += 1

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "VideoReader":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


class VideoWriter:
    """Thin wrapper that lazily creates the writer once the first frame size is known."""

    def __init__(self, path: str | Path, fps: float, codec: str = "mp4v"):
        self.path = str(path)
        self.fps = fps if fps and fps > 0 else 25.0
        self.codec = codec
        self.writer: Optional[cv2.VideoWriter] = None

    def write(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
            if not self.writer.isOpened():
                raise VideoError(
                    f"Could not open '{self.path}' for writing with codec '{self.codec}'."
                )
        self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def resize_keep_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Downscale to ``target_width`` (never upscale) preserving aspect ratio."""
    if target_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    return cv2.resize(frame, (target_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def _label_block(
    frame: np.ndarray,
    lines,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw a filled label block with one line of text per entry."""
    x, y = origin
    pad = 4
    sizes = [cv2.getTextSize(t, FONT, scale, thickness)[0] for t in lines]
    bw = max(s[0] for s in sizes) + pad * 2
    lh = max(s[1] for s in sizes) + 6
    bh = lh * len(lines) + pad

    h, w = frame.shape[:2]
    y_top = max(0, y - bh)
    x = max(0, min(x, w - bw - 1))
    y_top = max(0, min(y_top, h - bh - 1))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y_top), (x + bw, y_top + bh), color, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    for i, text in enumerate(lines):
        ty = y_top + pad + lh * (i + 1) - 6
        cv2.putText(frame, text, (x + pad, ty), FONT, scale, (15, 15, 15), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x + pad, ty), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_vehicle(frame: np.ndarray, det, plate_text: str = "", plate_conf: float = 0.0,
                 status: str = "", draw_labels: bool = True) -> None:
    """Draw one tracked vehicle and, if known, its aggregated plate reading."""
    x1, y1, x2, y2 = [int(v) for v in det.box]
    color = CLASS_COLORS.get(det.class_name, (200, 200, 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if not draw_labels:
        return

    tid = f" #{det.track_id}" if getattr(det, "track_id", None) is not None else ""
    lines = [f"{det.class_name.upper()}{tid}  {det.confidence:.0%}"]
    if plate_text:
        lines.append(f"Plate: {plate_text}")
        lines.append(f"OCR: {plate_conf:.0%}" + ("  LOW" if plate_conf < 0.5 else ""))
    elif status == "NO_TEXT":
        lines.append("Plate: unreadable")
    _label_block(frame, lines, (x1, y1), color)


def draw_plate(frame: np.ndarray, box, text: str = "", conf: float = 0.0,
               status: str = "") -> None:
    """Draw a plate box, colour-coded by validation status."""
    x1, y1, x2, y2 = [int(v) for v in box]
    color = STATUS_COLORS.get(status, CLASS_COLORS["plate"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if text:
        _label_block(frame, [f"{text} {conf:.0%}"], (x1, y2 + 22), color, scale=0.45)


def draw_hud(frame: np.ndarray, lines) -> None:
    """Top-left heads-up display with live processing statistics."""
    _label_block(frame, list(lines), (10, 18 + 20 * len(lines)), (35, 35, 35), scale=0.5)


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """BGR (OpenCV) -> RGB (Streamlit/Matplotlib)."""
    if frame is None:
        return frame
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_image(path: str | Path) -> np.ndarray:
    """Read an image, raising a user-friendly error instead of returning None."""
    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        raise VideoError(
            f"'{path}' is not a readable image. Supported: JPG, PNG, BMP, WEBP."
        )
    return img


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode an uploaded file's bytes into a BGR array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise VideoError("The uploaded file could not be decoded as an image.")
    return img
