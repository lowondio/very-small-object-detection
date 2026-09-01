"""Run tracking: per-epoch JSON metrics, periodic full evaluation, curves.

RunTracker is the `TrainingMonitor` every detector reports into. It owns three
artifacts per run:

    epochs.jsonl / epochs.csv   one record per epoch, canonical keys
    ap_history.csv              full metric pass every eval_every_epochs
    learning_curves.png         losses, native AP, and dynamic AP over epochs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import ExperimentConfig, RunPaths
from .data import DatasetLayout
from .detectors import Detector
from .evaluation import evaluate_detector, headline_metrics
from .metrics import MetricsReport

# Columns kept in epochs.csv; anything else stays in the jsonl `raw` block.
EPOCH_COLUMNS = (
    "epoch",
    "train/loss",
    "val/loss",
    "val/precision",
    "val/recall",
    "val/ap50",
    "val/ap50_95",
    "fitness",
    "lr",
)

EvaluateFn = Callable[[int, Path], MetricsReport]


class RunTracker:
    """Collects everything a run produces. Implements TrainingMonitor."""

    def __init__(
        self,
        config: ExperimentConfig,
        paths: RunPaths | None = None,
        evaluate_fn: EvaluateFn | None = None,
    ) -> None:
        self.config = config
        self.paths = (paths or config.paths()).create()
        self.evaluate_fn = evaluate_fn

        self.epochs: list[dict[str, Any]] = []
        self.evaluations: list[dict[str, Any]] = []

        self.paths.config_json.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
        self.paths.epochs_jsonl.write_text("")

    # ---------------- TrainingMonitor ----------------

    def on_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        record = {"epoch": epoch, **metrics}
        self.epochs.append(record)

        with open(self.paths.epochs_jsonl, "a") as file:
            file.write(json.dumps(record, default=_json_default) + "\n")

        self.epoch_table().to_csv(self.paths.epochs_csv, index=False)

        summary = "  ".join(
            f"{key.split('/')[-1]}={record[key]:.4f}"
            for key in ("train/loss", "val/ap50", "val/ap50_95")
            if isinstance(record.get(key), float)
        )
        print(f"[epoch {epoch:>4d}] {summary}")

    def wants_checkpoint(self, epoch: int) -> bool:
        if self.evaluate_fn is None or not self.config.eval_during_training:
            return False
        every = self.config.eval_every_epochs
        return every > 0 and epoch % every == 0

    def on_checkpoint(self, epoch: int, weights: Path) -> None:
        print(f"[epoch {epoch:>4d}] full metric pass on {weights}")
        try:
            report = self.evaluate_fn(epoch, Path(weights))
        except Exception as error:  # never let evaluation kill a training run
            print(f"[epoch {epoch:>4d}] evaluation failed: {error!r}")
            return
        self.record_evaluation(epoch, report)

    # ---------------- results ----------------

    def record_evaluation(self, epoch: int | None, report: MetricsReport, tag: str = "") -> None:
        record = {"epoch": epoch, "tag": tag or (f"epoch_{epoch:04d}" if epoch else "final")}
        record.update(headline_metrics(report))
        self.evaluations.append(record)

        pd.DataFrame(self.evaluations).to_csv(self.paths.ap_history_csv, index=False)

        # Per class, never averaged: a mean would hide the class we are losing.
        summary = "  ".join(
            f"{name} AP={record[f'{name}/ap']:.4f}"
            for name in self.config.class_names
            if isinstance(record.get(f"{name}/ap"), float)
        )
        print(f"[eval {record['tag']}] {summary}")

    def epoch_table(self) -> pd.DataFrame:
        if not self.epochs:
            return pd.DataFrame(columns=list(EPOCH_COLUMNS))
        table = pd.DataFrame(self.epochs)
        columns = [column for column in EPOCH_COLUMNS if column in table.columns]
        return table[columns]

    def evaluation_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.evaluations)

    def finalize(self) -> Path | None:
        """Write the learning curves. Safe to call on a partial run."""
        return plot_learning_curves(
            self.epoch_table(), self.evaluation_table(), self.paths.curves_png, self.config.name
        )


def _json_default(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------
# Periodic evaluation
# --------------------------------------------------------------------------


def periodic_evaluator(
    detector: Detector,
    dataset: DatasetLayout,
    config: ExperimentConfig,
    paths: RunPaths | None = None,
) -> EvaluateFn:
    """Build the callback RunTracker uses for its every-N-epochs metric pass.

    The detector is released afterwards so the trainer gets the GPU memory back.
    """
    paths = paths or config.paths()

    def evaluate(epoch: int, weights: Path) -> MetricsReport:
        output_dir = paths.eval_dir / f"epoch_{epoch:04d}"
        try:
            return evaluate_detector(
                detector, dataset, config, output_dir, weights=weights, save_plots=False
            )
        finally:
            detector.release()

    return evaluate


# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------


def plot_learning_curves(
    epochs: pd.DataFrame,
    evaluations: pd.DataFrame,
    output_path: Path,
    title: str = "",
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not len(epochs):
        print("no epoch records, skipping curves")
        return None

    panels = [
        ("loss", [("train/loss", "train"), ("val/loss", "val")]),
        ("native val AP", [("val/ap50", "AP50"), ("val/ap50_95", "AP50-95")]),
        ("val P / R", [("val/precision", "precision"), ("val/recall", "recall")]),
        ("learning rate", [("lr", "lr")]),
    ]

    figure, axes = plt.subplots(1, len(panels) + 1, figsize=(4.2 * (len(panels) + 1), 3.6))

    for axis, (name, series) in zip(axes, panels):
        drawn = False
        for column, label in series:
            if column not in epochs.columns:
                continue
            values = pd.to_numeric(epochs[column], errors="coerce")
            if values.notna().sum() == 0:
                continue
            axis.plot(epochs["epoch"], values, label=label, linewidth=1.4)
            drawn = True
        axis.set_title(name)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        if drawn:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "no data", ha="center", va="center", transform=axis.transAxes)

    # Last panel: AP per class from the full metric pass. One line per class,
    # never a mean -- the whole point is to see the classes diverge.
    axis = axes[-1]
    ap_columns = [column for column in evaluations.columns if column.endswith("/ap")] if len(evaluations) else []
    if ap_columns:
        periodic = evaluations.dropna(subset=["epoch"])
        for column in ap_columns:
            values = pd.to_numeric(periodic[column], errors="coerce")
            if values.notna().sum() == 0:
                continue
            axis.plot(periodic["epoch"], values, marker="o", label=column.split("/")[0], linewidth=1.4)
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "no full evaluations", ha="center", va="center", transform=axis.transAxes)
    axis.set_title("AP per class")
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.3)

    if title:
        figure.suptitle(title)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)

    print("curves saved to", output_path)
    return output_path
