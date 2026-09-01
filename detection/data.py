"""Dataset access.

The training dataset is SFO-2class, built by `dataset_preparation/` and shipped
as a COCO dataset. It is downloaded from Google Drive, not kept in the repo:

    sfo-2class/
        images/{train,val,test}/*.jpg
        annotations/instances_{train,val,test}.json   COCO, absolute xywh
        build_manifest.csv

COCO is the ground truth of record. Ultralytics needs YOLO txt labels, so
`export_yolo_labels` derives them from the same annotations rather than
maintaining a second copy of the truth. Category ids are 1 = drone, 2 = bird in
COCO and 0 = drone, 1 = bird in YOLO.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import DEFAULT_RATIO_BINS, METRIC_IMGSZ, assign_bins, bin_labels, relative_scale

SPLITS = ("train", "val", "test")
CLASS_NAMES = ("drone", "bird")


@dataclass
class DatasetLayout:
    """Handle on the dataset on disk. The only thing detectors depend on."""

    root: Path
    class_names: tuple[str, ...] = CLASS_NAMES

    @classmethod
    def at(cls, root: Path | str, class_names: Sequence[str] = CLASS_NAMES) -> "DatasetLayout":
        return cls(root=Path(root), class_names=tuple(class_names))

    def images_dir(self, split: str) -> Path:
        return self.root / "images" / split

    def annotation_file(self, split: str) -> Path:
        return self.root / "annotations" / f"instances_{split}.json"

    def labels_dir(self, split: str) -> Path:
        return self.root / "labels" / split

    @property
    def data_yaml(self) -> Path:
        return self.root / "data.yaml"

    def splits(self) -> list[str]:
        return [split for split in SPLITS if self.annotation_file(split).exists()]

    def exists(self) -> bool:
        splits = self.splits()
        return bool(splits) and any(self.images_dir(splits[0]).glob("*.jpg"))

    def check(self) -> None:
        """Fail early and loudly rather than deep inside a training run."""
        if not self.root.exists():
            raise FileNotFoundError(f"dataset not found at {self.root}")

        for split in self.splits():
            annotations = self.annotation_file(split)
            payload = json.loads(annotations.read_text())

            names = {category["name"] for category in payload["categories"]}
            if names != set(self.class_names):
                raise ValueError(f"{annotations}: classes {sorted(names)} != {list(self.class_names)}")

            on_disk = {path.name for path in self.images_dir(split).glob("*.jpg")}
            listed = {image["file_name"] for image in payload["images"]}
            missing = listed - on_disk
            if missing:
                raise FileNotFoundError(
                    f"{split}: {len(missing)} images listed but absent, e.g. {sorted(missing)[:3]}"
                )

        print(f"dataset ok: {self.root} splits={self.splits()}")


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------


def download_from_drive(file_id: str, target: Path | str, extract_to: Path | str | None = None) -> Path:
    """Pull the dataset archive from Google Drive and unpack it.

    `file_id` is the id in a Drive share link:
    https://drive.google.com/file/d/<file_id>/view
    """
    import gdown

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        gdown.download(id=file_id, output=str(target), quiet=False)

    if extract_to is not None:
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
        print("unpacking", target, "->", extract_to)
        shutil.unpack_archive(str(target), str(extract_to))
        return extract_to

    return target


def mount_drive_dataset(drive_path: Path | str, local_root: Path | str) -> DatasetLayout:
    """Use a dataset that already sits on a mounted Drive.

    Copying to local disk first is worth it in Colab: reading ~15k JPEGs per
    epoch straight from the Drive mount is far slower than from /content.
    """
    drive_path, local_root = Path(drive_path), Path(local_root)

    if not local_root.exists():
        if drive_path.is_file():
            local_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(drive_path), str(local_root.parent))
        else:
            print("copying", drive_path, "->", local_root)
            shutil.copytree(drive_path, local_root)

    layout = DatasetLayout.at(local_root)
    layout.check()
    return layout


# --------------------------------------------------------------------------
# YOLO export
# --------------------------------------------------------------------------


def export_yolo_labels(layout: DatasetLayout, force: bool = False) -> Path:
    """Derive YOLO txt labels and data.yaml from the COCO annotations.

    Images without annotations get an empty txt file: Ultralytics reads those
    as background frames, which is what they are.
    """
    import yaml

    class_index = {name: index for index, name in enumerate(layout.class_names)}

    for split in layout.splits():
        labels_dir = layout.labels_dir(split)
        if labels_dir.exists() and not force:
            print(f"labels/{split} exists, skipping (force=True to rebuild)")
            continue
        labels_dir.mkdir(parents=True, exist_ok=True)

        payload = json.loads(layout.annotation_file(split).read_text())
        category_names = {category["id"]: category["name"] for category in payload["categories"]}
        images = {image["id"]: image for image in payload["images"]}

        lines: dict[int, list[str]] = {image_id: [] for image_id in images}
        for annotation in payload["annotations"]:
            if annotation.get("iscrowd", 0):
                continue
            image = images[annotation["image_id"]]
            width, height = float(image["width"]), float(image["height"])
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])

            lines[annotation["image_id"]].append(
                f"{class_index[category_names[annotation['category_id']]]} "
                f"{(x + box_width / 2) / width:.6f} {(y + box_height / 2) / height:.6f} "
                f"{box_width / width:.6f} {box_height / height:.6f}"
            )

        for image_id, image in images.items():
            path = labels_dir / f"{Path(image['file_name']).stem}.txt"
            text = "\n".join(lines[image_id])
            path.write_text(text + "\n" if text else "")

        print(f"labels/{split}: {len(images)} files")

    config: dict[str, Any] = {
        "path": str(layout.root),
        "names": {index: name for index, name in enumerate(layout.class_names)},
    }
    for split in layout.splits():
        config[split] = f"images/{split}"

    layout.data_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    print("wrote", layout.data_yaml)
    return layout.data_yaml


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def bin_statistics(
    layout: DatasetLayout,
    ratio_bins: Sequence[float] = DEFAULT_RATIO_BINS,
    imgsz: int = METRIC_IMGSZ,
) -> pd.DataFrame:
    """Boxes per class per relative-scale bin, per split.

    Same binning as the metrics, so a weak bin in the results can be read
    straight against how much training signal that bin actually had.
    """
    labels = bin_labels(ratio_bins)
    rows: list[dict[str, Any]] = []

    for split in layout.splits():
        payload = json.loads(layout.annotation_file(split).read_text())
        category_names = {category["id"]: category["name"] for category in payload["categories"]}
        images = {image["id"]: image for image in payload["images"]}

        box_widths, box_heights, image_widths, image_heights, classes = [], [], [], [], []
        for annotation in payload["annotations"]:
            image = images[annotation["image_id"]]
            _, _, box_width, box_height = annotation["bbox"]
            box_widths.append(box_width)
            box_heights.append(box_height)
            image_widths.append(image["width"])
            image_heights.append(image["height"])
            classes.append(category_names[annotation["category_id"]])

        if not classes:
            continue

        ratios = relative_scale(
            np.asarray(box_widths), np.asarray(box_heights),
            np.asarray(image_widths), np.asarray(image_heights),
        )
        indices = assign_bins(ratios, ratio_bins)
        rows.extend(
            {"split": split, "class": class_name, "bin": labels[index]}
            for class_name, index in zip(classes, indices)
        )

    frame = pd.DataFrame(rows)
    if not len(frame):
        return frame

    table = frame.groupby(["class", "bin", "split"]).size().unstack("split", fill_value=0).reset_index()
    split_columns = [column for column in table.columns if column in SPLITS]
    table["total"] = table[split_columns].sum(axis=1)
    table["px_at_imgsz"] = table["bin"].map(
        {label: round(_bin_midpoint(index, ratio_bins) * imgsz, 1) for index, label in enumerate(labels)}
    )

    order = {label: index for index, label in enumerate(labels)}
    table["_order"] = table["bin"].map(order)
    return table.sort_values(["class", "_order"]).drop(columns="_order").reset_index(drop=True)


def _bin_midpoint(index: int, edges: Sequence[float]) -> float:
    if index < len(edges) - 1:
        return (edges[index] + edges[index + 1]) / 2
    return edges[-1]
