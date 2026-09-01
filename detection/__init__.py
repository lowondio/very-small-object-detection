"""Detection of small flying objects: one training and evaluation interface
for several architectures (Ultralytics YOLO, D-FINE), two classes.

Metrics are reported per class and per relative-scale bin, never averaged
across classes. See `detection/metrics.py`.
"""

from .config import (
    CLASS_NAMES,
    DataConfig,
    EvalConfigBlock,
    ExperimentConfig,
    ModelConfig,
    RunPaths,
    dfine_experiment,
    yolo_experiment,
)
from .data import (
    DatasetLayout,
    bin_statistics,
    download_from_drive,
    export_yolo_labels,
    mount_drive_dataset,
)
from .detectors import Detector, build_detector, register_detector
from .evaluation import evaluate_detector, headline_metrics, print_report
from .metrics import (
    DEFAULT_RATIO_BINS,
    METRIC_IMGSZ,
    MetricsConfig,
    MetricsReport,
    evaluate,
    load_coco_ground_truth,
    load_predictions,
    normalize_predictions,
)
from .tracking import RunTracker, periodic_evaluator, plot_learning_curves

__all__ = [
    "CLASS_NAMES",
    "DEFAULT_RATIO_BINS",
    "METRIC_IMGSZ",
    "DataConfig",
    "DatasetLayout",
    "Detector",
    "EvalConfigBlock",
    "ExperimentConfig",
    "MetricsConfig",
    "MetricsReport",
    "ModelConfig",
    "RunPaths",
    "RunTracker",
    "bin_statistics",
    "build_detector",
    "dfine_experiment",
    "download_from_drive",
    "evaluate",
    "evaluate_detector",
    "export_yolo_labels",
    "headline_metrics",
    "load_coco_ground_truth",
    "load_predictions",
    "normalize_predictions",
    "periodic_evaluator",
    "plot_learning_curves",
    "print_report",
    "register_detector",
    "yolo_experiment",
]
