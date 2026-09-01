"""Detection metrics for small flying objects.

Three things make this different from a stock COCO evaluator:

1. Objects are binned by **relative scale** r = max(box side) / max(image side),
   with fine bins in the 0-2.5% range where this dataset lives. Absolute pixel
   bins are useless here: the same drone is 96 px at 4K and 32 px at HD.
2. Every metric is reported **per class**, never averaged across classes. A
   mean of drone AP and bird AP would hide exactly what we care about.
3. Cross-class confusion is counted explicitly: how many birds the model calls
   drones and the other way round, in total and per scale bin.

Input format (pandas DataFrame or list of dicts):

    images        image_id, width, height
    ground_truth  image_id, class, x1, y1, x2, y2
    predictions   image_id, class, x1, y1, x2, y2, score

Coordinates are absolute pixels of the original image, xyxy. `class` may be a
name ("drone") or a model class id (0) — ids are mapped through
`MetricsConfig.class_names`. This is exactly what `Detector.predict` returns.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12

# Relative-scale bin edges, matching dataset_preparation/bin_stats.py.
# r = max(box side) / max(image side); the dataset caps r at 2.5%.
DEFAULT_RATIO_BINS = (0.0, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.025)

# The reference resolution the metric is defined at. This is a property of the
# METRIC, not of the model: it is deliberately NOT tied to the training imgsz.
# Object size enters the metric as w = r * METRIC_IMGSZ, so letting this follow
# the training resolution would silently make a run at 1280 be scored by a
# stricter threshold than a run at 640, and the two AP numbers would no longer
# be comparable. Changing it re-defines the metric for every past run.
METRIC_IMGSZ = 640

# COCO-style IoU sweep for the AP50-95 column.
COCO_IOU_SWEEP = tuple(round(value, 2) for value in np.arange(0.50, 0.96, 0.05))

__all__ = [
    "MetricsConfig",
    "MetricsReport",
    "evaluate",
    "bin_labels",
    "relative_scale",
    "dynamic_iou_threshold",
    "load_coco_ground_truth",
    "load_predictions",
    "normalize_predictions",
]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class MetricsConfig:
    """Everything the evaluation depends on. No CLI, no hidden defaults."""

    # Order matters: index = model class id (YOLO 0 = drone, 1 = bird).
    class_names: tuple[str, ...] = ("drone", "bird")

    ratio_bins: tuple[float, ...] = DEFAULT_RATIO_BINS

    # Reference resolution of the metric, fixed at METRIC_IMGSZ. Object size
    # enters as w = r * imgsz. Do NOT set this from the training imgsz: runs at
    # different resolutions would be scored by different thresholds and their
    # AP would stop being comparable.
    imgsz: int = METRIC_IMGSZ

    # Primary matching threshold: a float, or "dynamic" for a size-dependent
    # threshold. At these scales a flat 0.5 is dominated by rounding error: it
    # demands a 6 px object be localized to within 1.1 px on both axes.
    iou_threshold: float | str = "dynamic"

    # Dynamic threshold shape, in pixels of the metric's reference resolution.
    # delta_max = 2 px is what makes the threshold genuinely size-dependent
    # across this dataset: it holds at 0.30 up to r = 0.98% (6.2 px), then
    # ramps and reaches 0.50 at r = 1.70% (10.9 px). A larger delta_max pushes
    # the ramp past the 2.5% cap, so every box would sit on the 0.30 plateau
    # and the "dynamic" threshold would degenerate into a flat lenient IoU.
    delta_max: float = 2.0
    delta_ratio: float = 0.32
    iou_min: float = 0.20
    iou_max: float = 0.50

    # Confidence at which precision/recall/confusion are reported.
    operating_point: str | float = "best_f1"  # "best_f1" | "best_f2" | float
    fbeta: float = 2.0

    # Predictions below this score are dropped before anything is computed.
    # Keep it low: a high floor truncates the PR curve and understates AP.
    min_score: float = 0.001

    def bin_edges(self) -> np.ndarray:
        return np.asarray(self.ratio_bins, dtype=float)

    def bin_labels(self) -> list[str]:
        return bin_labels(self.ratio_bins)

    def thresholds_for(self, ratios: np.ndarray) -> np.ndarray:
        """Per-object IoU threshold, given each object's relative scale."""
        if self.iou_threshold == "dynamic":
            return dynamic_iou_threshold(
                np.asarray(ratios, float) * self.imgsz,
                delta_max=self.delta_max,
                delta_ratio=self.delta_ratio,
                iou_min=self.iou_min,
                iou_max=self.iou_max,
            )
        return np.full(len(ratios), float(self.iou_threshold), dtype=float)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ratio_bins"] = list(self.ratio_bins)
        data["class_names"] = list(self.class_names)
        return data


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def relative_scale(
    box_width: np.ndarray, box_height: np.ndarray, image_width: np.ndarray, image_height: np.ndarray
) -> np.ndarray:
    """r = max(box side) / max(image side). The dataset's own size measure."""
    longer_box = np.maximum(np.asarray(box_width, float), np.asarray(box_height, float))
    longer_image = np.maximum(np.asarray(image_width, float), np.asarray(image_height, float))
    return longer_box / np.maximum(longer_image, EPS)


def dynamic_iou_threshold(
    size_px: np.ndarray,
    delta_max: float = 2.0,
    delta_ratio: float = 0.32,
    iou_min: float = 0.20,
    iou_max: float = 0.50,
) -> np.ndarray:
    """IoU a correctly sized box keeps after a delta-pixel shift on both axes.

        delta(w) = min(delta_max, delta_ratio * w)
        tau(w)   = clip( (w - delta)^2 / (2w^2 - (w - delta)^2), iou_min, iou_max )

    The threshold expresses a tolerated localization error in pixels rather
    than in fractions of the object, which is the only thing that makes sense
    when the object is a handful of pixels wide.
    """
    size = np.asarray(size_px, dtype=float)
    delta = np.minimum(delta_max, delta_ratio * size)

    numerator = np.square(np.maximum(size - delta, 0.0))
    denominator = 2.0 * np.square(size) - numerator

    threshold = np.full(size.shape, iou_max, dtype=float)
    usable = denominator > EPS
    threshold[usable] = numerator[usable] / denominator[usable]
    return np.clip(threshold, iou_min, iou_max)


def bin_labels(edges: Sequence[float]) -> list[str]:
    """['0-0.3%', '0.3-0.5%', ..., '2-2.5%', '>2.5%']"""

    def number(value: float) -> str:
        return f"{value * 100:.2f}".rstrip("0").rstrip(".")

    def percent(value: float) -> str:
        return f"{number(value)}%"

    labels = [f"{number(edges[i])}-{percent(edges[i + 1])}" for i in range(len(edges) - 1)]
    # Predictions are not capped by the dataset's size rule, so keep an
    # overflow bin. For ground truth it stays empty.
    labels.append(f">{percent(edges[-1])}")
    return labels


def assign_bins(ratios: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Bin index per object; the last index is the overflow bin.

    The top edge is inclusive. The dataset caps r at exactly 2.5% and a large
    share of boxes sit precisely on the cap, so a half-open [lo, hi) top bin
    would exile them all to the overflow bin.
    """
    edge_array = np.asarray(edges, dtype=float)
    ratios = np.asarray(ratios, float)

    index = np.searchsorted(edge_array, ratios, side="right") - 1
    index = np.clip(index, 0, len(edge_array) - 1)
    index[np.isclose(ratios, edge_array[-1])] = len(edge_array) - 2
    return index


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes, shape (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)

    ax1, ay1, ax2, ay2 = (boxes_a[:, i][:, None] for i in range(4))
    bx1, by1, bx2, by2 = (boxes_b[:, i][None, :] for i in range(4))

    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
    intersection = inter_w * inter_h

    area_a = np.clip(ax2 - ax1, 0.0, None) * np.clip(ay2 - ay1, 0.0, None)
    area_b = np.clip(bx2 - bx1, 0.0, None) * np.clip(by2 - by1, 0.0, None)
    return intersection / np.maximum(area_a + area_b - intersection, EPS)


# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------


def average_precision(scores: np.ndarray, is_true_positive: np.ndarray, n_ground_truth: int) -> float:
    """All-point interpolated AP."""
    if n_ground_truth == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0

    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    hits = np.asarray(is_true_positive, bool)[order]

    true_positives = np.cumsum(hits)
    false_positives = np.cumsum(~hits)

    recall = true_positives / n_ground_truth
    precision = true_positives / np.maximum(true_positives + false_positives, EPS)

    recall = np.concatenate([[0.0], recall, [recall[-1]]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])

    steps = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


def pr_curve(scores: np.ndarray, is_true_positive: np.ndarray, n_ground_truth: int) -> pd.DataFrame:
    if len(scores) == 0:
        return pd.DataFrame(
            columns=["confidence", "true_positives", "false_positives", "precision", "recall", "f1", "fbeta"]
        )

    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    hits = np.asarray(is_true_positive, bool)[order]

    true_positives = np.cumsum(hits)
    false_positives = np.cumsum(~hits)
    recall = true_positives / max(n_ground_truth, 1)
    precision = true_positives / np.maximum(true_positives + false_positives, EPS)

    return pd.DataFrame(
        {
            "confidence": np.asarray(scores, float)[order],
            "true_positives": true_positives,
            "false_positives": false_positives,
            "precision": precision,
            "recall": recall,
        }
    )


def fbeta_score(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    beta_squared = beta * beta
    numerator = (1 + beta_squared) * precision * recall
    denominator = beta_squared * precision + recall
    return np.where(denominator > EPS, numerator / np.maximum(denominator, EPS), 0.0)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def match_boxes(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greedy one-to-one matching, highest score first.

    Returns, aligned to the row order of the inputs:
        pred_is_tp    bool  per prediction
        pred_gt_index int   index of the matched GT row, -1 if none
        gt_matched    bool  per ground truth box
        gt_iou        float IoU of the box that matched it, 0 if unmatched
    """
    n_gt = len(ground_truth)
    n_pred = len(predictions)

    pred_is_tp = np.zeros(n_pred, dtype=bool)
    pred_gt_index = np.full(n_pred, -1, dtype=int)
    gt_matched = np.zeros(n_gt, dtype=bool)
    gt_iou = np.zeros(n_gt, dtype=float)

    if n_gt == 0 or n_pred == 0:
        return pred_is_tp, pred_gt_index, gt_matched, gt_iou

    gt_boxes = ground_truth[["x1", "y1", "x2", "y2"]].to_numpy(float)
    pred_boxes = predictions[["x1", "y1", "x2", "y2"]].to_numpy(float)

    gt_by_image: dict[Any, list[int]] = {}
    for position, image_id in enumerate(ground_truth["image_id"].to_numpy()):
        gt_by_image.setdefault(image_id, []).append(position)

    pred_images = predictions["image_id"].to_numpy()
    order = np.argsort(-predictions["score"].to_numpy(float), kind="mergesort")

    for prediction_index in order:
        candidates = gt_by_image.get(pred_images[prediction_index])
        if not candidates:
            continue

        candidate_array = np.asarray(candidates)
        free = candidate_array[~gt_matched[candidate_array]]
        if len(free) == 0:
            continue

        ious = iou_matrix(pred_boxes[prediction_index : prediction_index + 1], gt_boxes[free])[0]
        passing = ious >= thresholds[free]
        if not passing.any():
            continue

        # Best IoU among the ground truth boxes that clear their own threshold.
        best = free[np.argmax(np.where(passing, ious, -1.0))]
        pred_is_tp[prediction_index] = True
        pred_gt_index[prediction_index] = int(best)
        gt_matched[best] = True
        gt_iou[best] = float(ious[np.where(free == best)[0][0]])

    return pred_is_tp, pred_gt_index, gt_matched, gt_iou


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class MetricsReport:
    """Per class, never averaged across classes."""

    per_class: pd.DataFrame
    per_bin: pd.DataFrame
    confusion: pd.DataFrame
    confusion_per_bin: pd.DataFrame
    pr_curves: pd.DataFrame
    ground_truth: pd.DataFrame
    predictions: pd.DataFrame
    config: MetricsConfig

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        tables = {
            "per_class": self.per_class,
            "per_bin": self.per_bin,
            "confusion": self.confusion,
            "confusion_per_bin": self.confusion_per_bin,
            "pr_curves": self.pr_curves,
            "matched_ground_truth": self.ground_truth,
            "matched_predictions": self.predictions,
        }
        for name, table in tables.items():
            if table is not None and len(table):
                table.to_csv(directory / f"{name}.csv", index=False)

        summary = {
            "config": self.config.to_dict(),
            "per_class": self.per_class.to_dict(orient="records"),
            "confusion": self.confusion.to_dict(orient="records"),
        }
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)
        )
        return directory

    def ap(self, class_name: str) -> float:
        row = self.per_class[self.per_class["class"] == class_name]
        return float(row["ap"].iloc[0]) if len(row) else float("nan")

    def headline(self) -> dict[str, float]:
        """Flat dict for logging AP per class across epochs."""
        result: dict[str, float] = {}
        for _, row in self.per_class.iterrows():
            name = row["class"]
            for metric in ("ap", "ap50", "ap50_95", "recall", "precision", "f1"):
                result[f"{name}/{metric}"] = _as_float(row.get(metric))
        for _, row in self.confusion.iterrows():
            for column in row.index:
                # Skip a class's own column: "birds confused as birds" is not a thing.
                if column.startswith("confused_as_") and not pd.isna(row[column]):
                    result[f"{row['gt_class']}/{column}"] = _as_float(row[column])
        return result

    def print_summary(self) -> None:
        pd.set_option("display.width", 200)

        print("\n=== PER CLASS ===")
        print(self.per_class.round(4).to_string(index=False))

        print("\n=== PER RELATIVE-SCALE BIN ===")
        print("(r = longest box side / longest image side; px is r * imgsz)")
        for class_name in self.config.class_names:
            rows = self.per_bin[self.per_bin["class"] == class_name]
            if not len(rows):
                continue
            print(f"\n-- {class_name}")
            columns = [
                "bin",
                "px_at_imgsz",
                "n_ground_truth",
                "ap",
                "recall",
                "precision",
                "f1",
                "mean_iou_threshold",
            ]
            print(rows[[c for c in columns if c in rows.columns]].round(4).to_string(index=False))

        print("\n=== CLASS CONFUSION (ground truth view, at the operating point) ===")
        print(self.confusion.round(4).to_string(index=False))

        print("\n=== CLASS CONFUSION PER BIN ===")
        print(self.confusion_per_bin.round(4).to_string(index=False))

    def plot(self, directory: str | Path | None = None, show: bool = False):
        return plot_report(self, directory=directory, show=show)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------
# Input normalization
# --------------------------------------------------------------------------


def _to_frame(data: Any, name: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (list, tuple)):
        return pd.DataFrame(list(data))
    if isinstance(data, dict):
        return pd.DataFrame(data)
    raise TypeError(f"{name}: expected DataFrame or list of dicts, got {type(data).__name__}")


def _resolve_classes(frame: pd.DataFrame, class_names: Sequence[str], name: str) -> pd.DataFrame:
    """Accept 'class' names, 'class_id' model ids, or COCO 'category_id'."""
    frame = frame.copy()

    if "class" not in frame.columns:
        if "class_id" in frame.columns:
            frame["class"] = frame["class_id"].map(lambda index: _name_of(index, class_names))
        elif "category_id" in frame.columns:
            # COCO ids are 1-based in this dataset; model ids are 0-based.
            frame["class"] = frame["category_id"].map(lambda index: _name_of(int(index) - 1, class_names))
        else:
            raise ValueError(f"{name}: need a 'class', 'class_id' or 'category_id' column")

    unknown = set(frame["class"].unique()) - set(class_names)
    if unknown:
        raise ValueError(f"{name}: classes {sorted(unknown)} are not in class_names={list(class_names)}")
    return frame


def _name_of(index: Any, class_names: Sequence[str]) -> str:
    position = int(index)
    if not 0 <= position < len(class_names):
        raise ValueError(f"class id {index} outside class_names={list(class_names)}")
    return class_names[position]


def _normalize_boxes(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Accept xyxy directly, or COCO 'bbox' as [x, y, w, h]."""
    frame = frame.copy()

    if {"x1", "y1", "x2", "y2"} <= set(frame.columns):
        return frame

    if "bbox" in frame.columns:
        boxes = np.asarray([list(box) for box in frame["bbox"]], dtype=float)
        frame["x1"], frame["y1"] = boxes[:, 0], boxes[:, 1]
        frame["x2"], frame["y2"] = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
        return frame

    raise ValueError(f"{name}: need x1/y1/x2/y2 columns or a COCO 'bbox' column")


def normalize_predictions(
    predictions: Any, class_names: Sequence[str] = ("drone", "bird"), min_score: float = 0.0
) -> pd.DataFrame:
    """Bring any accepted prediction shape into the canonical table."""
    frame = _resolve_classes(_normalize_boxes(_to_frame(predictions, "predictions"), "predictions"), class_names, "predictions")
    if "score" not in frame.columns:
        raise ValueError("predictions: need a 'score' column")
    frame = frame[frame["score"] >= min_score]
    return frame[["image_id", "class", "x1", "y1", "x2", "y2", "score"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def load_coco_ground_truth(
    annotation_file: str | Path, class_names: Sequence[str] = ("drone", "bird")
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a COCO instances json into (images, ground_truth).

    `image_id` becomes the file stem, which is what detectors report, so
    predictions and ground truth line up without an id table.
    """
    payload = json.loads(Path(annotation_file).read_text())

    category_names = {category["id"]: category["name"] for category in payload["categories"]}
    unknown = set(category_names.values()) - set(class_names)
    if unknown:
        raise ValueError(f"annotation file has classes {sorted(unknown)} not in {list(class_names)}")

    image_rows = []
    stem_of: dict[int, str] = {}
    for image in payload["images"]:
        stem = Path(image["file_name"]).stem
        stem_of[image["id"]] = stem
        image_rows.append({"image_id": stem, "width": int(image["width"]), "height": int(image["height"])})

    gt_rows = []
    for annotation in payload["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        x, y, width, height = (float(value) for value in annotation["bbox"])
        gt_rows.append(
            {
                "image_id": stem_of[annotation["image_id"]],
                "class": category_names[annotation["category_id"]],
                "x1": x,
                "y1": y,
                "x2": x + width,
                "y2": y + height,
            }
        )

    return pd.DataFrame(image_rows), pd.DataFrame(gt_rows)


def load_predictions(
    path: str | Path, class_names: Sequence[str] = ("drone", "bird"), min_score: float = 0.0
) -> pd.DataFrame:
    """Read predictions from .csv or .json (a table, or COCO detection results)."""
    path = Path(path)

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        payload = json.loads(path.read_text())
        frame = pd.DataFrame(payload["annotations"] if isinstance(payload, dict) else payload)

    return normalize_predictions(frame, class_names, min_score)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate(
    images: Any,
    ground_truth: Any,
    predictions: Any,
    config: MetricsConfig | None = None,
) -> MetricsReport:
    """Score one split. Every number is per class; nothing is averaged over classes."""
    config = config or MetricsConfig()

    images_frame = _to_frame(images, "images")
    for column in ("image_id", "width", "height"):
        if column not in images_frame.columns:
            raise ValueError(f"images: missing column {column!r}")
    images_frame = images_frame.drop_duplicates(subset="image_id").reset_index(drop=True)

    gt = _resolve_classes(_normalize_boxes(_to_frame(ground_truth, "ground_truth"), "ground_truth"), config.class_names, "ground_truth")
    predictions_frame = normalize_predictions(predictions, config.class_names, config.min_score)

    gt = _attach_scale(gt, images_frame, config)
    predictions_frame = _attach_scale(predictions_frame, images_frame, config)

    labels = config.bin_labels()

    per_class_rows: list[dict[str, Any]] = []
    per_bin_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    operating_confidence: dict[str, float] = {}

    gt_parts: list[pd.DataFrame] = []
    pred_parts: list[pd.DataFrame] = []

    for class_name in config.class_names:
        class_gt = gt[gt["class"] == class_name].reset_index(drop=True)
        class_pred = predictions_frame[predictions_frame["class"] == class_name].reset_index(drop=True)

        thresholds = config.thresholds_for(class_gt["ratio"].to_numpy(float))
        class_gt["iou_threshold"] = thresholds

        is_tp, gt_index, matched, matched_iou = match_boxes(class_gt, class_pred, thresholds)
        class_pred["is_true_positive"] = is_tp
        class_pred["matched_gt_index"] = gt_index
        class_gt["is_detected"] = matched
        class_gt["matched_iou"] = matched_iou

        n_gt = len(class_gt)
        scores = class_pred["score"].to_numpy(float)

        curve = pr_curve(scores, is_tp, n_gt)
        if len(curve):
            curve["f1"] = fbeta_score(curve["precision"].to_numpy(), curve["recall"].to_numpy(), 1.0)
            curve["fbeta"] = fbeta_score(
                curve["precision"].to_numpy(), curve["recall"].to_numpy(), config.fbeta
            )
        operating = _operating_point(curve, config)
        operating_confidence[class_name] = operating["confidence"]

        curve.insert(0, "class", class_name)
        curve_frames.append(curve)

        # A TP is placed in the bin of the ground truth it matched; a FP in the
        # bin of its own predicted scale. That keeps every bin's PR curve
        # self-consistent instead of sharing false positives across bins.
        prediction_bin = class_pred["bin_index"].to_numpy(int).copy()
        matched_rows = gt_index >= 0
        if matched_rows.any():
            prediction_bin[matched_rows] = class_gt["bin_index"].to_numpy(int)[gt_index[matched_rows]]
        class_pred["effective_bin_index"] = prediction_bin

        per_class_rows.append(
            {
                "class": class_name,
                "n_images": len(images_frame),
                "n_ground_truth": n_gt,
                "n_predictions": len(class_pred),
                "ap": average_precision(scores, is_tp, n_gt),
                "ap50": _ap_at(class_gt, class_pred, 0.50),
                "ap50_95": float(
                    np.nanmean([_ap_at(class_gt, class_pred, value) for value in COCO_IOU_SWEEP])
                ),
                "operating_confidence": operating["confidence"],
                "precision": operating["precision"],
                "recall": operating["recall"],
                "f1": operating["f1"],
                f"f{config.fbeta:g}": operating["fbeta"],
                "true_positives": operating["true_positives"],
                "false_positives": operating["false_positives"],
                "false_negatives": n_gt - operating["true_positives"],
                "mean_iou_threshold": float(np.mean(thresholds)) if n_gt else float("nan"),
                "mean_iou_of_detected": float(np.mean(matched_iou[matched])) if matched.any() else float("nan"),
            }
        )

        per_bin_rows.extend(_per_bin_metrics(class_name, class_gt, class_pred, labels, config))

        gt_parts.append(class_gt)
        pred_parts.append(class_pred)

    gt_all = pd.concat(gt_parts, ignore_index=True) if gt_parts else pd.DataFrame()
    pred_all = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    confusion, confusion_per_bin = _confusion(gt_all, pred_all, operating_confidence, labels, config)

    return MetricsReport(
        per_class=pd.DataFrame(per_class_rows),
        per_bin=pd.DataFrame(per_bin_rows),
        confusion=confusion,
        confusion_per_bin=confusion_per_bin,
        pr_curves=pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame(),
        ground_truth=gt_all,
        predictions=pred_all,
        config=config,
    )


def _attach_scale(frame: pd.DataFrame, images: pd.DataFrame, config: MetricsConfig) -> pd.DataFrame:
    """Add relative scale and bin index using each box's own image size."""
    frame = frame.merge(images[["image_id", "width", "height"]], on="image_id", how="left")

    missing = frame["width"].isna()
    if missing.any():
        unknown = frame.loc[missing, "image_id"].unique()[:5]
        raise ValueError(f"{missing.sum()} boxes reference images not in the images table, e.g. {list(unknown)}")

    frame["ratio"] = relative_scale(
        frame["x2"] - frame["x1"],
        frame["y2"] - frame["y1"],
        frame["width"],
        frame["height"],
    )
    frame["bin_index"] = assign_bins(frame["ratio"].to_numpy(float), config.ratio_bins)
    return frame


def _ap_at(ground_truth: pd.DataFrame, predictions: pd.DataFrame, iou_threshold: float) -> float:
    """AP at a fixed IoU, for comparability with published numbers."""
    thresholds = np.full(len(ground_truth), float(iou_threshold))
    is_tp, *_ = match_boxes(ground_truth, predictions, thresholds)
    return average_precision(predictions["score"].to_numpy(float), is_tp, len(ground_truth))


def _operating_point(curve: pd.DataFrame, config: MetricsConfig) -> dict[str, float]:
    empty = {
        "confidence": float("nan"),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "fbeta": 0.0,
        "true_positives": 0,
        "false_positives": 0,
    }
    if not len(curve):
        return empty

    rule = config.operating_point
    if isinstance(rule, str):
        column = "f1" if rule == "best_f1" else "fbeta"
        position = int(np.argmax(curve[column].to_numpy(float)))
    else:
        below = np.where(curve["confidence"].to_numpy(float) >= float(rule))[0]
        if len(below) == 0:
            return empty
        position = int(below[-1])

    row = curve.iloc[position]
    return {
        "confidence": float(row["confidence"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "fbeta": float(row["fbeta"]),
        "true_positives": int(row["true_positives"]),
        "false_positives": int(row["false_positives"]),
    }


def _per_bin_metrics(
    class_name: str,
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    labels: Sequence[str],
    config: MetricsConfig,
) -> list[dict[str, Any]]:
    edges = config.bin_edges()
    rows = []

    gt_bins = ground_truth["bin_index"].to_numpy(int)
    pred_bins = predictions["effective_bin_index"].to_numpy(int)
    scores = predictions["score"].to_numpy(float)
    is_tp = predictions["is_true_positive"].to_numpy(bool)

    for index, label in enumerate(labels):
        in_gt = gt_bins == index
        in_pred = pred_bins == index
        n_gt = int(in_gt.sum())

        bin_scores = scores[in_pred]
        bin_is_tp = is_tp[in_pred]

        curve = pr_curve(bin_scores, bin_is_tp, n_gt)
        if len(curve):
            curve["f1"] = fbeta_score(curve["precision"].to_numpy(), curve["recall"].to_numpy(), 1.0)
            curve["fbeta"] = fbeta_score(curve["precision"].to_numpy(), curve["recall"].to_numpy(), config.fbeta)
        operating = _operating_point(curve, config)

        # Bin midpoint in pixels at training resolution; the overflow bin has
        # no upper edge, so report its lower edge.
        if index < len(edges) - 1:
            midpoint = (edges[index] + edges[index + 1]) / 2
        else:
            midpoint = edges[-1]

        rows.append(
            {
                "class": class_name,
                "bin": label,
                "bin_index": index,
                "px_at_imgsz": round(midpoint * config.imgsz, 1),
                "n_ground_truth": n_gt,
                "n_predictions": int(in_pred.sum()),
                "ap": average_precision(bin_scores, bin_is_tp, n_gt),
                "precision": operating["precision"],
                "recall": operating["recall"],
                "f1": operating["f1"],
                "n_detected": int(ground_truth.loc[in_gt, "is_detected"].sum()) if n_gt else 0,
                "mean_iou_threshold": float(ground_truth.loc[in_gt, "iou_threshold"].mean()) if n_gt else float("nan"),
            }
        )

    return rows


# --------------------------------------------------------------------------
# Cross-class confusion
# --------------------------------------------------------------------------


def _confusion(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    operating_confidence: dict[str, float],
    labels: Sequence[str],
    config: MetricsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per ground truth box, decide: found by its own class, taken for another
    class, or missed entirely.

    "Taken for another class" means no prediction of the correct class matched
    the box, but a prediction of a different class -- above that class's own
    operating confidence -- overlaps it past the same IoU threshold. This is
    the number that answers "how many birds does the model call drones".
    """
    if not len(ground_truth):
        return pd.DataFrame(), pd.DataFrame()

    class_names = list(config.class_names)
    outcome = np.where(ground_truth["is_detected"].to_numpy(bool), "correct", "missed").astype(object)

    # Only boxes their own class failed to find can be confused with another.
    undetected = ~ground_truth["is_detected"].to_numpy(bool)

    for other in class_names:
        other_predictions = predictions[
            (predictions["class"] == other)
            & (predictions["score"] >= operating_confidence.get(other, float("inf")))
        ]
        if not len(other_predictions):
            continue

        boxes_by_image: dict[Any, np.ndarray] = {
            image_id: group[["x1", "y1", "x2", "y2"]].to_numpy(float)
            for image_id, group in other_predictions.groupby("image_id")
        }

        gt_boxes = ground_truth[["x1", "y1", "x2", "y2"]].to_numpy(float)
        gt_images = ground_truth["image_id"].to_numpy()
        gt_classes = ground_truth["class"].to_numpy()
        gt_thresholds = ground_truth["iou_threshold"].to_numpy(float)

        for position in np.where(undetected)[0]:
            if gt_classes[position] == other or outcome[position] != "missed":
                continue
            candidates = boxes_by_image.get(gt_images[position])
            if candidates is None:
                continue
            ious = iou_matrix(gt_boxes[position : position + 1], candidates)[0]
            if ious.max(initial=0.0) >= gt_thresholds[position]:
                outcome[position] = f"confused_as_{other}"

    table = ground_truth[["class", "bin_index"]].copy()
    table["outcome"] = outcome

    def summarize(group: pd.DataFrame, class_name: str) -> dict[str, Any]:
        n_gt = len(group)
        counts = group["outcome"].value_counts()
        row: dict[str, Any] = {"gt_class": class_name, "n_ground_truth": n_gt}
        row["correct"] = int(counts.get("correct", 0))
        for other in class_names:
            if other == class_name:
                continue
            confused = int(counts.get(f"confused_as_{other}", 0))
            row[f"confused_as_{other}"] = confused
            row[f"confused_as_{other}_rate"] = confused / n_gt if n_gt else float("nan")
        row["missed"] = int(counts.get("missed", 0))
        return row

    confusion_rows = [
        summarize(table[table["class"] == class_name], class_name)
        for class_name in class_names
        if (table["class"] == class_name).any()
    ]

    per_bin_rows = []
    for class_name in class_names:
        class_rows = table[table["class"] == class_name]
        if not len(class_rows):
            continue
        for index, label in enumerate(labels):
            group = class_rows[class_rows["bin_index"] == index]
            row = summarize(group, class_name)
            row["bin"] = label
            row["bin_index"] = index
            per_bin_rows.append(row)

    def tidy(frame: pd.DataFrame, leading: list[str]) -> pd.DataFrame:
        if not len(frame):
            return frame
        confused = [c for c in frame.columns if c.startswith("confused_as_") and not c.endswith("_rate")]
        rates = [f"{c}_rate" for c in confused if f"{c}_rate" in frame.columns]
        columns = leading + ["n_ground_truth", "correct"] + confused + ["missed"] + rates
        frame = frame[[c for c in dict.fromkeys(columns) if c in frame.columns]]
        # Counts stay integers even where a class has no column of its own.
        for column in ["n_ground_truth", "correct", "missed", *confused]:
            if column in frame.columns:
                frame[column] = frame[column].astype("Int64")
        return frame

    return (
        tidy(pd.DataFrame(confusion_rows), ["gt_class"]),
        tidy(pd.DataFrame(per_bin_rows), ["gt_class", "bin", "bin_index"]),
    )


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_report(report: MetricsReport, directory: str | Path | None = None, show: bool = False):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = report.config
    class_names = [name for name in config.class_names if (report.per_class["class"] == name).any()]
    figures: dict[str, Any] = {}

    # 1. PR curve per class.
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    for class_name in class_names:
        curve = report.pr_curves[report.pr_curves["class"] == class_name]
        if not len(curve):
            continue
        axis.plot(curve["recall"], curve["precision"], label=f"{class_name} (AP {report.ap(class_name):.3f})")
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_title("PR curve per class")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figures["pr_curve"] = figure

    # 2. AP and recall per relative-scale bin.
    for metric in ("ap", "recall"):
        figure, axis = plt.subplots(figsize=(8, 4))
        labels = config.bin_labels()
        positions = np.arange(len(labels))
        width = 0.8 / max(len(class_names), 1)

        for offset, class_name in enumerate(class_names):
            rows = report.per_bin[report.per_bin["class"] == class_name].set_index("bin")
            values = [rows[metric].get(label, np.nan) for label in labels]
            axis.bar(positions + offset * width, values, width, label=class_name)

        axis.set_xticks(positions + width * (len(class_names) - 1) / 2)
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axis.set_ylabel(metric)
        axis.set_title(f"{metric} per relative-scale bin")
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figures[f"{metric}_per_bin"] = figure

    # 3. Confusion per bin, one stacked panel per class.
    if len(report.confusion_per_bin):
        figure, axes = plt.subplots(1, max(len(class_names), 1), figsize=(5.5 * max(len(class_names), 1), 4))
        axes = np.atleast_1d(axes)
        labels = config.bin_labels()

        for axis, class_name in zip(axes, class_names):
            rows = report.confusion_per_bin[report.confusion_per_bin["gt_class"] == class_name].set_index("bin")
            confused_columns = [c for c in rows.columns if c.startswith("confused_as_") and not c.endswith("_rate")]
            bottom = np.zeros(len(labels))

            for column in ["correct", *confused_columns, "missed"]:
                if column not in rows.columns:
                    continue
                # A class has no "confused as itself" column, so entries can be NA.
                series = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
                values = np.array([series.get(label, 0.0) for label in labels], dtype=float)
                if not values.any():
                    continue
                axis.bar(labels, values, bottom=bottom, label=column)
                bottom += values

            axis.set_title(f"ground truth: {class_name}")
            axis.set_ylabel("boxes")
            axis.tick_params(axis="x", rotation=30, labelsize=8)
            axis.legend(fontsize=8)
        figure.suptitle("outcome per relative-scale bin")
        figure.tight_layout()
        figures["confusion_per_bin"] = figure

    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(directory / f"{name}.png", dpi=140, bbox_inches="tight")

    if show:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)

    return figures
