from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sfod.video_inference import run_video_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run object detection on a video file")
    parser.add_argument("--source", type=Path, required=True, help="input video path")
    parser.add_argument("--weights", type=Path, required=True, help="path to trained weights")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/video_inference"), help="directory for output")
    parser.add_argument("--detector", choices=["yolo"], default="yolo")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--show", action="store_true", default=False)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    info = run_video_inference(
        source=args.source,
        weights=args.weights,
        detector=args.detector,
        output_dir=args.output_dir,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        imgsz=args.imgsz,
        save=True,
        show=args.show,
        stream=False,
    )

    print("\nvideo inference complete")
    print("source:", info["source"])
    print("weights:", info["weights"])
    print("output_dir:", info["output_dir"])
    print("saved results:", info["results"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
