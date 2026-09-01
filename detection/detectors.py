"""Detector adapters behind one interface.

Every detector implements the same three things:

    train(dataset, monitor) -> best weights path
    load(weights)
    predict(images_dir, ...) -> (images_df, predictions_df)

`predict` returns the exact tables that sfod.metrics consumes (absolute xyxy
pixels in original image space), so the evaluation path is identical for all
architectures. To add a model, subclass `Detector` and call
`register_detector`; nothing else in the project needs to change.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .config import ExperimentConfig, RunPaths
from .data import DatasetLayout


# --------------------------------------------------------------------------
# Training callbacks
# --------------------------------------------------------------------------


class TrainingMonitor(Protocol):
    """What a detector reports back while training. Implemented by RunTracker."""

    def on_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Per-epoch metrics, already mapped to canonical keys."""

    def wants_checkpoint(self, epoch: int) -> bool:
        """True when a full metric pass should run on this epoch."""

    def on_checkpoint(self, epoch: int, weights: Path) -> None:
        """Weights are ready for a full metric pass."""


class NullMonitor:
    def on_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        pass

    def wants_checkpoint(self, epoch: int) -> bool:
        return False

    def on_checkpoint(self, epoch: int, weights: Path) -> None:
        pass


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------


class Detector(ABC):
    key: str = ""
    # Which label format the framework reads. Both are always exported.
    dataset_format: str = "yolo"

    def __init__(self, config: ExperimentConfig, paths: RunPaths) -> None:
        self.config = config
        self.paths = paths

    def prepare(self, dataset: DatasetLayout) -> None:
        """Framework setup that needs dataset paths. Called before train and
        before evaluating a checkpoint in a fresh session."""

    @abstractmethod
    def train(self, dataset: DatasetLayout, monitor: TrainingMonitor) -> Path:
        """Run training and return the path of the best checkpoint."""

    @abstractmethod
    def load(self, weights: Path | str | None = None) -> None:
        """Load weights for inference."""

    @abstractmethod
    def predict(
        self,
        images_dir: Path | str,
        conf: float = 0.001,
        iou: float = 0.70,
        max_det: int = 300,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run inference over a directory of images.

        Returns the two tables `detection.metrics` consumes:
            images       image_id, width, height
            predictions  image_id, class, x1, y1, x2, y2, score

        `image_id` is the file stem and coordinates are absolute pixels of the
        original image, so predictions line up with the COCO ground truth
        without an id table.
        """

    def class_name(self, class_id: int) -> str:
        names = self.config.class_names
        index = int(class_id)
        if not 0 <= index < len(names):
            raise ValueError(f"model produced class id {class_id}, expected one of {list(range(len(names)))}")
        return names[index]

    @abstractmethod
    def best_weights(self) -> Path:
        """Best checkpoint produced by the last training run."""

    def release(self) -> None:
        """Drop the inference model so a trainer can reuse the GPU."""


# --------------------------------------------------------------------------
# Ultralytics YOLO
# --------------------------------------------------------------------------


class YoloDetector(Detector):
    key = "yolo"
    dataset_format = "yolo"

    def __init__(self, config: ExperimentConfig, paths: RunPaths) -> None:
        super().__init__(config, paths)
        self._model = None

    def _device(self) -> Any:
        device = self.config.model.device
        return int(device) if str(device).isdigit() else device

    def train(self, dataset: DatasetLayout, monitor: TrainingMonitor) -> Path:
        from ultralytics import YOLO

        model_config = self.config.model
        model = YOLO(model_config.weights)

        def on_fit_epoch_end(trainer):
            # Fires after validation, so val metrics are already populated.
            epoch = int(trainer.epoch) + 1
            monitor.on_epoch(epoch, _yolo_epoch_metrics(trainer))
            if monitor.wants_checkpoint(epoch):
                monitor.on_checkpoint(epoch, Path(trainer.last))

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        arguments: dict[str, Any] = {
            "data": str(dataset.data_yaml),
            "epochs": model_config.epochs,
            "imgsz": model_config.imgsz,
            "batch": model_config.batch,
            "workers": model_config.workers,
            "device": self._device(),
            "amp": model_config.amp,
            "seed": model_config.seed,
            "project": str(self.paths.train_dir.parent),
            "name": self.paths.train_dir.name,
            "exist_ok": True,
        }
        arguments.update(model_config.train_overrides)

        model.train(**arguments)
        return self.best_weights()

    def best_weights(self) -> Path:
        return self.paths.train_dir / "weights" / "best.pt"

    def load(self, weights: Path | str | None = None) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(weights or self.best_weights()))

    def predict(
        self,
        images_dir: Path | str,
        conf: float = 0.001,
        iou: float = 0.70,
        max_det: int = 300,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._model is None:
            self.load()

        results = self._model.predict(
            source=str(images_dir),
            imgsz=self.config.model.imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=self._device(),
            stream=True,  # keeps memory flat over a large split
            verbose=False,
            **self.config.model.predict_overrides,
        )

        image_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []

        for index, result in enumerate(results):
            image_id = Path(result.path).stem if getattr(result, "path", None) else f"image_{index:06d}"
            height, width = result.orig_shape
            image_rows.append({"image_id": image_id, "width": int(width), "height": int(height)})

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy().astype(float)
            scores = boxes.conf.cpu().numpy().astype(float)
            classes = boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), score, class_id in zip(xyxy, scores, classes):
                prediction_rows.append(
                    {
                        "image_id": image_id,
                        "class": self.class_name(class_id),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": score,
                    }
                )

        return pd.DataFrame(image_rows), pd.DataFrame(prediction_rows)

    def release(self) -> None:
        self._model = None
        _empty_cuda_cache()


def _yolo_epoch_metrics(trainer) -> dict[str, Any]:
    """Map Ultralytics metric names onto the canonical keys used for plots."""
    raw: dict[str, Any] = {}
    for key, value in dict(getattr(trainer, "metrics", {}) or {}).items():
        try:
            raw[key] = float(value)
        except (TypeError, ValueError):
            continue

    try:
        train_losses = trainer.label_loss_items(trainer.tloss, prefix="train")
        raw.update({key: float(value) for key, value in train_losses.items()})
    except Exception:
        train_losses = {}

    learning_rate = None
    try:
        learning_rates = [float(value) for value in trainer.lr.values()]
        learning_rate = sum(learning_rates) / len(learning_rates)
    except Exception:
        pass

    def total(prefix: str) -> float | None:
        values = [value for key, value in raw.items() if key.startswith(prefix) and key.endswith("loss")]
        return float(sum(values)) if values else None

    return {
        "train/loss": total("train/"),
        "val/loss": total("val/"),
        "val/precision": raw.get("metrics/precision(B)"),
        "val/recall": raw.get("metrics/recall(B)"),
        "val/ap50": raw.get("metrics/mAP50(B)"),
        "val/ap50_95": raw.get("metrics/mAP50-95(B)"),
        "fitness": raw.get("fitness"),
        "lr": learning_rate,
        "raw": raw,
    }


# --------------------------------------------------------------------------
# D-FINE
# --------------------------------------------------------------------------


DFINE_REPO_URL = "https://github.com/Peterande/D-FINE.git"
DFINE_CHECKPOINT_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{size}_coco.pth"

# Names in the 12-value COCO eval vector that D-FINE writes to log.txt.
COCO_EVAL_NAMES = (
    "ap50_95",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_1",
    "ar_10",
    "ar_100",
    "ar_small",
    "ar_medium",
    "ar_large",
)


class DFineDetector(Detector):
    """D-FINE trains through its own train.py, so the epoch stream is recovered
    by tailing the JSON lines it appends to log.txt while the process runs."""

    key = "dfine"
    dataset_format = "coco"

    poll_seconds = 20.0

    def __init__(self, config: ExperimentConfig, paths: RunPaths) -> None:
        super().__init__(config, paths)
        self._model = None
        self._postprocessor = None
        self._transform = None
        self._device = None

        overrides = config.model.train_overrides
        self.repo_dir = Path(overrides.get("repo_dir", "/content/D-FINE"))
        self.model_size = str(config.model.weights or "s")
        self.pretrained_path = Path(
            overrides.get("pretrained_path", f"/content/dfine_{self.model_size}_coco.pth")
        )
        self.config_path = self.repo_dir / "configs" / "dfine" / "custom" / f"dfine_{config.name}.yml"

    # ---------------- setup ----------------

    def setup(self) -> None:
        """Clone the repo, install its requirements, fetch the COCO checkpoint."""
        if not self.repo_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", DFINE_REPO_URL, str(self.repo_dir)],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(self.repo_dir / "requirements.txt")],
                check=True,
            )

        if not self.pretrained_path.exists():
            import requests

            url = DFINE_CHECKPOINT_URL.format(size=self.model_size)
            self.pretrained_path.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(url, stream=True, timeout=600) as response:
                response.raise_for_status()
                with open(self.pretrained_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        file.write(chunk)
            print("pretrained checkpoint:", self.pretrained_path)

    def write_config(self, dataset: DatasetLayout) -> Path:
        """Inherit the official custom config and override only what we vary."""
        model_config = self.config.model
        overrides = model_config.train_overrides
        # Heavy augmentation is switched off this many epochs before the end.
        stop_epoch = max(1, model_config.epochs - int(overrides.get("policy_epoch_offset", 10)))

        text = f"""
__include__: ['./dfine_hgnetv2_{self.model_size}_custom.yml']

output_dir: {self.paths.train_dir}
num_classes: {self.config.num_classes}
remap_mscoco_category: False

use_amp: {model_config.amp}
sync_bn: False
use_wandb: False
print_freq: 50
checkpoint_freq: {self.config.eval_every_epochs}
epochs: {model_config.epochs}

train_dataloader:
  total_batch_size: {model_config.batch}
  num_workers: {model_config.workers}
  dataset:
    img_folder: {dataset.images_dir('train')}
    ann_file: {dataset.annotation_file('train')}
    transforms:
      policy:
        epoch: {stop_epoch}
  collate_fn:
    base_size: {model_config.imgsz}
    base_size_repeat: ~
    stop_epoch: {stop_epoch}
    ema_restart_decay: 0.9999

val_dataloader:
  total_batch_size: {model_config.batch}
  num_workers: {model_config.workers}
  dataset:
    img_folder: {dataset.images_dir('val')}
    ann_file: {dataset.annotation_file('val')}
"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(text)
        return self.config_path

    # ---------------- training ----------------

    def prepare(self, dataset: DatasetLayout) -> None:
        self.setup()
        self.write_config(dataset)

    def train(self, dataset: DatasetLayout, monitor: TrainingMonitor) -> Path:
        self.prepare(dataset)
        self.paths.train_dir.mkdir(parents=True, exist_ok=True)

        overrides = self.config.model.train_overrides
        last_checkpoint = self.paths.train_dir / "last.pth"
        log_path = self.paths.train_dir / "log.txt"

        command = [
            sys.executable,
            "train.py",
            "-c",
            str(self.config_path),
            "--seed",
            str(self.config.model.seed),
            "-d",
            str(self.config.model.device),
        ]
        if self.config.model.amp:
            command.append("--use-amp")

        if overrides.get("resume_if_available", True) and last_checkpoint.exists():
            command += ["-r", str(last_checkpoint)]
            print("resuming from", last_checkpoint)
        else:
            command += ["-t", str(self.pretrained_path)]
            print("fine-tuning from", self.pretrained_path)

        print("command:", " ".join(command))
        process = subprocess.Popen(command, cwd=str(self.repo_dir))

        consumed = 0
        while process.poll() is None:
            time.sleep(self.poll_seconds)
            consumed = self._drain_log(log_path, consumed, monitor, last_checkpoint)

        self._drain_log(log_path, consumed, monitor, last_checkpoint)

        if process.returncode != 0:
            raise RuntimeError(f"D-FINE training failed with exit code {process.returncode}")

        return self.best_weights()

    def _drain_log(
        self,
        log_path: Path,
        consumed: int,
        monitor: TrainingMonitor,
        last_checkpoint: Path,
    ) -> int:
        """Emit every log line that appeared since the previous poll."""
        if not log_path.exists():
            return consumed

        lines = [line for line in log_path.read_text().splitlines() if line.strip()]
        for line in lines[consumed:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            epoch = int(record.get("epoch", -1)) + 1
            monitor.on_epoch(epoch, _dfine_epoch_metrics(record))

            # last.pth is rewritten every epoch, so it holds the current weights.
            if monitor.wants_checkpoint(epoch) and last_checkpoint.exists():
                monitor.on_checkpoint(epoch, last_checkpoint)

        return len(lines)

    def best_weights(self) -> Path:
        for name in ("best_stg2.pth", "best_stg1.pth", "last.pth"):
            candidate = self.paths.train_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"no D-FINE checkpoint in {self.paths.train_dir}")

    # ---------------- inference ----------------

    def load(self, weights: Path | str | None = None) -> None:
        import torch
        import torchvision.transforms as transforms

        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        from src.core import YAMLConfig

        weights = Path(weights or self.best_weights())

        yaml_config = YAMLConfig(str(self.config_path))
        yaml_config.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        yaml_config.model.load_state_dict(state)

        self._device = torch.device(self.config.model.device if torch.cuda.is_available() else "cpu")
        self._model = yaml_config.model.deploy().to(self._device).eval()
        self._postprocessor = yaml_config.postprocessor.deploy().to(self._device)

        size = self.config.model.imgsz
        self._transform = transforms.Compose(
            [transforms.Resize((size, size)), transforms.ToTensor()]
        )

    def predict(
        self,
        images_dir: Path | str,
        conf: float = 0.001,
        iou: float = 0.70,  # unused: D-FINE is NMS-free
        max_det: int = 300,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        import torch
        from PIL import Image
        from tqdm.auto import tqdm

        if self._model is None:
            self.load()

        images_dir = Path(images_dir)
        image_paths = sorted(
            path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        batch_size = max(1, self.config.model.batch)

        image_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []

        with torch.inference_mode():
            for start in tqdm(range(0, len(image_paths), batch_size), desc="predict"):
                batch_paths = image_paths[start : start + batch_size]
                pil_images = [Image.open(path).convert("RGB") for path in batch_paths]
                sizes = [image.size for image in pil_images]  # (width, height)

                tensors = torch.stack([self._transform(image) for image in pil_images]).to(self._device)
                original_sizes = torch.tensor(sizes, device=self._device)

                if self._device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self._model(tensors)
                else:
                    outputs = self._model(tensors)

                labels, boxes, scores = self._postprocessor(outputs, original_sizes)

                for path, (width, height), image_labels, image_boxes, image_scores in zip(
                    batch_paths, sizes, labels, boxes, scores
                ):
                    image_id = path.stem
                    image_rows.append(
                        {"image_id": image_id, "width": int(width), "height": int(height)}
                    )

                    label_array = image_labels.detach().cpu().numpy().astype(int)
                    box_array = image_boxes.detach().float().cpu().numpy()
                    score_array = image_scores.detach().float().cpu().numpy()
                    order = score_array.argsort()[::-1][:max_det]

                    for index in order:
                        score = float(score_array[index])
                        if score < conf:
                            continue
                        x1, y1, x2, y2 = (float(value) for value in box_array[index])
                        prediction_rows.append(
                            {
                                "image_id": image_id,
                                "class": self.class_name(label_array[index]),
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "score": score,
                            }
                        )

        return pd.DataFrame(image_rows), pd.DataFrame(prediction_rows)

    def release(self) -> None:
        self._model = None
        self._postprocessor = None
        _empty_cuda_cache()


def _dfine_epoch_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Map a D-FINE log.txt record onto the canonical keys used for plots."""
    coco = record.get("test_coco_eval_bbox") or []
    coco_values = dict(zip(COCO_EVAL_NAMES, [float(value) for value in coco]))

    return {
        "train/loss": _as_float(record.get("train_loss")),
        "val/loss": _as_float(record.get("test_loss")),
        "val/precision": None,  # COCO eval reports AP/AR, not a fixed-threshold precision
        "val/recall": coco_values.get("ar_100"),
        "val/ap50": coco_values.get("ap50"),
        "val/ap50_95": coco_values.get("ap50_95"),
        "fitness": coco_values.get("ap50_95"),
        "lr": _as_float(record.get("train_lr")),
        "raw": {**{key: _as_float(value) for key, value in record.items() if _is_scalar(value)}, **coco_values},
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


DETECTORS: dict[str, type[Detector]] = {
    YoloDetector.key: YoloDetector,
    DFineDetector.key: DFineDetector,
}


def register_detector(detector_class: type[Detector]) -> type[Detector]:
    DETECTORS[detector_class.key] = detector_class
    return detector_class


def build_detector(config: ExperimentConfig, paths: RunPaths | None = None) -> Detector:
    key = config.model.detector
    if key not in DETECTORS:
        raise KeyError(f"unknown detector {key!r}, available: {sorted(DETECTORS)}")
    return DETECTORS[key](config, paths or config.paths())
