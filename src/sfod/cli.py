from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from detection.config import DataConfig, dfine_experiment, yolo_experiment
from detection.data import (
    DatasetLayout,
    bin_statistics,
    download_from_drive,
    export_yolo_labels,
    mount_drive_dataset,
)
from detection.detectors import build_detector
from detection.evaluation import evaluate_detector, print_report
from detection.metrics import METRIC_IMGSZ
from .video_inference import run_video_inference


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", type=Path, default=Path("/content/sfo-2class"))
    parser.add_argument("--drive-file-id", default="")
    parser.add_argument("--drive-path", type=Path, default=None)
    parser.add_argument("--archive-path", type=Path, default=Path("/content/sfo-2class.tar"))
    parser.add_argument("--class-names", nargs="+", default=["drone", "bird"])
    parser.add_argument("--metric-imgsz", type=int, default=METRIC_IMGSZ)
    parser.add_argument("--export-yolo-labels", dest="export_yolo_labels", action="store_true", default=True)
    parser.add_argument("--no-export-yolo-labels", dest="export_yolo_labels", action="store_false")
    parser.add_argument("--force-label-rebuild", action="store_true", default=False)


def prepare_dataset(args: argparse.Namespace) -> DatasetLayout:
    layout = DatasetLayout.at(args.dataset_dir, tuple(args.class_names))

    if not layout.exists():
        if args.drive_file_id:
            download_from_drive(args.drive_file_id, args.archive_path, extract_to=args.dataset_dir.parent)
            layout = DatasetLayout.at(args.dataset_dir, tuple(args.class_names))
        elif args.drive_path and args.drive_path.exists():
            layout = mount_drive_dataset(args.drive_path, args.dataset_dir)
        else:
            raise FileNotFoundError(
                f"no dataset at {args.dataset_dir}; set --drive-file-id or --drive-path"
            )

    layout.check()

    if args.export_yolo_labels:
        export_yolo_labels(layout, force=args.force_label_rebuild)

    print("\n=== BOXES PER RELATIVE-SCALE BIN ===")
    print("(r = longest box side / longest image side)")
    print(bin_statistics(layout, imgsz=args.metric_imgsz).to_string(index=False))

    print("dataset ready:", layout.root)
    return layout


def train(args: argparse.Namespace) -> Any:
    presets = {"yolo": yolo_experiment, "dfine": dfine_experiment}
    config = presets[args.detector](output_dir=args.output_dir)
    config.data = DataConfig(dataset_dir=args.dataset_dir)
    config.model.epochs = args.epochs
    config.model.imgsz = args.image_size
    config.model.batch = args.batch_size
    config.model.device = "cuda" if args.detector == "dfine" else args.device
    config.model.workers = 2 if args.detector == "dfine" else args.workers
    config.model.seed = args.seed
    config.evaluation.split = args.eval_split
    config.eval_every_epochs = args.eval_every_epochs
    config.eval_during_training = args.eval_during_training
    config.sync()

    paths = config.paths().create()
    dataset = DatasetLayout.at(config.data.dataset_dir, config.class_names)
    dataset.check()
    if config.data.export_yolo_labels:
        export_yolo_labels(dataset, force=config.data.force_label_rebuild)

    detector = build_detector(config, paths)

    from detection.tracking import RunTracker, periodic_evaluator

    tracker = RunTracker(
        config,
        paths,
        evaluate_fn=periodic_evaluator(detector, dataset, config, paths),
    )

    best_weights = detector.train(dataset, tracker)
    print("\nbest checkpoint:", best_weights)

    report = evaluate_detector(
        detector, dataset, config, paths.eval_dir / "final", weights=best_weights
    )
    tracker.record_evaluation(None, report, tag="final")
    tracker.finalize()

    print_report(report)
    print("\nrun directory:", paths.root)
    return report


def evaluate(args: argparse.Namespace) -> Any:
    presets = {"yolo": yolo_experiment, "dfine": dfine_experiment}
    config = presets[args.detector]()
    config.data = DataConfig(dataset_dir=args.dataset_dir)
    config.model.imgsz = args.image_size
    config.model.device = "cuda" if args.detector == "dfine" else args.device
    config.evaluation.split = args.split
    config.evaluation.conf = args.confidence
    config.evaluation.nms_iou = args.nms_iou
    config.evaluation.max_det = args.max_det
    config.sync()
    config.evaluation.metrics.iou_threshold = args.iou_threshold
    config.evaluation.metrics.operating_point = args.operating_point

    dataset = DatasetLayout.at(config.data.dataset_dir, config.class_names)
    dataset.check()

    detector = build_detector(config)
    report = evaluate_detector(detector, dataset, config, args.output_dir, weights=args.weights)
    print_report(report)
    print("\nsaved to", args.output_dir)
    return report


def video(args: argparse.Namespace) -> dict[str, Any]:
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
    print("generated:", info["results"])
    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small flying object detection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare the dataset")
    _add_common_dataset_args(prepare)
    prepare.set_defaults(func=lambda args: prepare_dataset(args))

    train_cmd = subparsers.add_parser("train", help="Train a detector")
    train_cmd.add_argument("--detector", choices=["yolo", "dfine"], default="yolo")
    train_cmd.add_argument("--dataset-dir", type=Path, default=Path("/content/sfo-2class"))
    train_cmd.add_argument("--output-dir", type=Path, default=Path("/content/drive/MyDrive/sfo/runs"))
    train_cmd.add_argument("--epochs", type=int, default=100)
    train_cmd.add_argument("--image-size", type=int, default=640)
    train_cmd.add_argument("--batch-size", type=int, default=16)
    train_cmd.add_argument("--device", default="0")
    train_cmd.add_argument("--workers", type=int, default=8)
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--eval-every-epochs", type=int, default=10)
    train_cmd.add_argument("--eval-during-training", dest="eval_during_training", action="store_true", default=True)
    train_cmd.add_argument("--no-eval-during-training", dest="eval_during_training", action="store_false")
    train_cmd.add_argument("--eval-split", default="val")
    train_cmd.set_defaults(func=lambda args: train(args))

    evaluate_cmd = subparsers.add_parser("evaluate", help="Evaluate an existing checkpoint")
    evaluate_cmd.add_argument("--detector", choices=["yolo", "dfine"], default="yolo")
    evaluate_cmd.add_argument("--dataset-dir", type=Path, default=Path("/content/sfo-2class"))
    evaluate_cmd.add_argument("--weights", type=Path, required=True)
    evaluate_cmd.add_argument("--output-dir", type=Path, required=True)
    evaluate_cmd.add_argument("--split", default="test")
    evaluate_cmd.add_argument("--image-size", type=int, default=640)
    evaluate_cmd.add_argument("--device", default="0")
    evaluate_cmd.add_argument("--confidence", type=float, default=0.001)
    evaluate_cmd.add_argument("--nms-iou", type=float, default=0.70)
    evaluate_cmd.add_argument("--max-det", type=int, default=300)
    evaluate_cmd.add_argument("--iou-threshold", default="dynamic")
    evaluate_cmd.add_argument("--operating-point", default="best_f1")
    evaluate_cmd.set_defaults(func=lambda args: evaluate(args))

    video_cmd = subparsers.add_parser("video", help="Run inference on a single video")
    video_cmd.add_argument("--source", type=Path, required=True, help="path to input video")
    video_cmd.add_argument("--weights", type=Path, required=True, help="path to trained weights")
    video_cmd.add_argument("--detector", choices=["yolo"], default="yolo")
    video_cmd.add_argument("--output-dir", type=Path, default=Path("runs/video_inference"), help="directory for annotated output")
    video_cmd.add_argument("--device", default="0", help="cuda or numeric GPU id")
    video_cmd.add_argument("--imgsz", type=int, default=640)
    video_cmd.add_argument("--conf", type=float, default=0.001)
    video_cmd.add_argument("--iou", type=float, default=0.70)
    video_cmd.add_argument("--max-det", type=int, default=300)
    video_cmd.add_argument("--show", action="store_true", default=False)
    video_cmd.set_defaults(func=lambda args: video(args))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
