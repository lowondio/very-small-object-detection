"""Experiment configuration.

Single control panel for the whole project. Nothing is read from the command
line: every runnable script imports a config object from here, or builds its
own inline, which keeps the code copy-pasteable into a Colab cell.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .metrics import MetricsConfig

# Order matters: index = model class id. COCO ids in the dataset are 1-based,
# so drone = COCO 1 = model 0, bird = COCO 2 = model 1.
CLASS_NAMES = ("drone", "bird")


@dataclass
class DataConfig:
    """Where the dataset comes from and where it lives locally.

    The dataset is not in the repo. Either it is already unpacked at
    `dataset_dir`, or it is fetched from Drive by id, or copied from a mounted
    Drive path.
    """

    dataset_dir: Path = Path("/content/sfo-2class")

    # Google Drive source; leave both unset if dataset_dir is already populated.
    drive_file_id: str | None = None
    drive_path: Path | None = None  # path on a mounted Drive (folder or archive)

    class_names: tuple[str, ...] = CLASS_NAMES

    # Ultralytics needs YOLO txt labels derived from the COCO annotations.
    export_yolo_labels: bool = True
    force_label_rebuild: bool = False


@dataclass
class ModelConfig:
    """Detector selection and training hyperparameters.

    `train_overrides` / `predict_overrides` are passed straight through to the
    underlying framework, so a new knob never needs a change here.
    """

    detector: str = "yolo"  # registry key: "yolo" | "dfine"
    weights: str = "yolo26s.pt"  # pretrained checkpoint or model name

    imgsz: int = 640
    batch: int = 16
    epochs: int = 100
    workers: int = 8
    device: str = "0"
    seed: int = 42
    amp: bool = True

    train_overrides: dict[str, Any] = field(default_factory=dict)
    predict_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfigBlock:
    """Inference settings plus the metric configuration they feed."""

    split: str = "test"

    # Keep confidence very low: a high floor truncates the PR curve and
    # understates AP. Filtering happens later, at the operating point.
    conf: float = 0.001
    nms_iou: float = 0.70
    max_det: int = 300

    metrics: MetricsConfig = field(default_factory=lambda: MetricsConfig(class_names=CLASS_NAMES))


@dataclass
class ExperimentConfig:
    name: str = "yolo26s_2class_640"
    output_dir: Path = Path("/content/drive/MyDrive/sfo/runs")

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvalConfigBlock = field(default_factory=EvalConfigBlock)

    # Full metric pass every N epochs, plus always at the end of training.
    eval_every_epochs: int = 10

    # Turn off if the GPU cannot hold a second model next to the trainer.
    eval_during_training: bool = True

    def __post_init__(self) -> None:
        self.sync()

    def sync(self) -> "ExperimentConfig":
        """One source of truth for the class list.

        The metric's reference resolution is deliberately not synced from
        `model.imgsz`: it is pinned at METRIC_IMGSZ so that a run trained at
        1280 and a run trained at 640 are scored by the same threshold and
        their AP stays comparable.
        """
        self.evaluation.metrics.class_names = tuple(self.data.class_names)
        return self

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self.data.class_names)

    @property
    def num_classes(self) -> int:
        return len(self.data.class_names)

    def paths(self) -> "RunPaths":
        return RunPaths.build(Path(self.output_dir) / self.name)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot, saved next to every run."""

        def encode(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if callable(value):
                return getattr(value, "__name__", repr(value))
            if isinstance(value, float) and math.isinf(value):
                return None
            if isinstance(value, (list, tuple)):
                return [encode(item) for item in value]
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            return value

        return encode(asdict(self))


@dataclass
class RunPaths:
    """Layout of a single run directory."""

    root: Path
    train_dir: Path  # framework-owned: checkpoints, native logs
    eval_dir: Path  # our metric reports, one subdirectory per checkpoint
    epochs_jsonl: Path
    epochs_csv: Path
    ap_history_csv: Path
    curves_png: Path
    config_json: Path

    @classmethod
    def build(cls, root: Path) -> "RunPaths":
        root = Path(root)
        return cls(
            root=root,
            train_dir=root / "train",
            eval_dir=root / "eval",
            epochs_jsonl=root / "epochs.jsonl",
            epochs_csv=root / "epochs.csv",
            ap_history_csv=root / "ap_history.csv",
            curves_png=root / "learning_curves.png",
            config_json=root / "config.json",
        )

    def create(self) -> "RunPaths":
        self.root.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------
# Ready-made presets
# --------------------------------------------------------------------------


def yolo_experiment(**overrides: Any) -> ExperimentConfig:
    config = ExperimentConfig(
        name="yolo26s_2class_640",
        model=ModelConfig(
            detector="yolo",
            weights="yolo26s.pt",
            train_overrides={
                "patience": 20,
                "cache": False,
                "optimizer": "auto",
                "cos_lr": True,
                "mosaic": 0.5,
                "close_mosaic": 10,
                "hsv_h": 0.015,
                "hsv_s": 0.5,
                "hsv_v": 0.4,
                "degrees": 0.0,
                # Objects are a few pixels wide, so scale augmentation is kept
                # mild: shrinking a 6 px box any further erases it.
                "translate": 0.1,
                "scale": 0.3,
                "shear": 0.0,
                "perspective": 0.0,
                "fliplr": 0.5,
                "flipud": 0.0,
                "deterministic": True,
                "save_period": 10,
                "plots": True,
            },
        ),
    )
    return _apply(config, overrides)


def dfine_experiment(**overrides: Any) -> ExperimentConfig:
    config = ExperimentConfig(
        name="dfine_s_2class_640",
        model=ModelConfig(
            detector="dfine",
            weights="s",  # model size; the COCO checkpoint is fetched by the adapter
            batch=16,
            workers=2,
            device="cuda",
            train_overrides={
                "repo_dir": "/content/D-FINE",
                "pretrained_path": "/content/dfine_s_coco.pth",
                "resume_if_available": True,
                # Stop the heavy augmentation policy this many epochs before the end.
                "policy_epoch_offset": 10,
            },
        ),
    )
    return _apply(config, overrides)


def _apply(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise AttributeError(f"unknown ExperimentConfig field: {key}")
        setattr(config, key, value)
    return config.sync()
