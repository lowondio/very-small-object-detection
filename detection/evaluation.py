"""Full metric pass, identical for every detector.

Ground truth always comes from the split's COCO annotations, and predictions
always arrive as the detector-agnostic table produced by `Detector.predict`,
so a D-FINE run and a YOLO run are scored by exactly the same code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .data import DatasetLayout
from .detectors import Detector
from .metrics import MetricsReport, evaluate, load_coco_ground_truth


def evaluate_detector(
    detector: Detector,
    dataset: DatasetLayout,
    config: ExperimentConfig,
    output_dir: Path | str,
    weights: Path | str | None = None,
    save_plots: bool = True,
) -> MetricsReport:
    """Run inference on a split and write the whole metric report to disk."""
    block = config.evaluation
    output_dir = Path(output_dir)

    detector.prepare(dataset)
    detector.load(weights)

    # The ground truth image table is authoritative: it also covers images the
    # model produced no boxes for, which is where recall is actually lost.
    images, ground_truth = load_coco_ground_truth(
        dataset.annotation_file(block.split), config.class_names
    )
    _, predictions = detector.predict(
        dataset.images_dir(block.split),
        conf=block.conf,
        iou=block.nms_iou,
        max_det=block.max_det,
    )

    report = evaluate(images, ground_truth, predictions, block.metrics)

    report.save(output_dir)
    if save_plots:
        report.plot(output_dir / "figures")

    return report


def headline_metrics(report: MetricsReport) -> dict[str, Any]:
    """Flat per-class dict tracked across epochs: drone/ap, bird/ap, ..."""
    return report.headline()


def print_report(report: MetricsReport) -> None:
    report.print_summary()
