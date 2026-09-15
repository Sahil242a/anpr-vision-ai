"""
Headless command-line runner.

Useful for batch jobs, for profiling without Streamlit in the way, and as proof
that the pipeline has no UI dependencies.

    python cli.py --image data/input/car.jpg
    python cli.py --video data/input/traffic.mp4 --max-frames 500
    python cli.py --check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import AppConfig  # noqa: E402
from src.pipeline import ANPRPipeline  # noqa: E402
from src.utils.video import VideoError, read_image  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ANPR Vision AI - command line runner")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", help="path to an image")
    src.add_argument("--video", help="path to a video")
    src.add_argument("--check", action="store_true", help="report component readiness and exit")
    p.add_argument("--output", help="output path for the annotated video")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames")
    p.add_argument("--frame-skip", type=int, default=None, help="process every (N+1)-th frame")
    p.add_argument("--ocr-interval", type=int, default=None, help="frames between OCR per track")
    p.add_argument("--device", default=None, help="auto | cpu | cuda:0")
    p.add_argument("--no-video", action="store_true", help="do not write an annotated video")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = AppConfig()
    if args.device:
        config.device = args.device
    if args.frame_skip is not None:
        config.video.frame_skip = args.frame_skip
    if args.ocr_interval is not None:
        config.ocr.interval = args.ocr_interval

    pipeline = ANPRPipeline(config)

    if args.check or (not args.image and not args.video):
        report = pipeline.readiness()
        for name, info in report.items():
            print(f"[{'OK ' if info.get('ok') else 'MISS'}] {name}: {info['detail']}")
        return 0 if all(i.get("ok") for i in report.values()) else 1

    if args.image:
        try:
            image = read_image(args.image)
        except VideoError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        result = pipeline.process_image(image, source=Path(args.image).name)
        payload = {
            "vehicles": [v.as_dict() for v in result.vehicles],
            "plates": [
                {
                    "text": r.text,
                    "confidence": round(r.combined_confidence, 3),
                    "status": r.status,
                    "variant": r.ocr.variant,
                    "validation": r.ocr.validation.as_dict() if r.ocr.validation else None,
                }
                for r in result.plates
            ],
            "warnings": result.warnings,
        }
        print(json.dumps(payload, indent=2))
        return 0

    outcome = pipeline.process_video(
        args.video,
        output_path=args.output,
        write_video=not args.no_video,
        max_frames=args.max_frames or None,
    )
    print(json.dumps(
        {
            "output": outcome.output_path,
            "stats": outcome.stats.as_dict(),
            "performance": outcome.performance,
            "plates": [p.as_dict() for p in outcome.plates],
            "warnings": outcome.warnings,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
