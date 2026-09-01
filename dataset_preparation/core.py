from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Geometry and dataset policy
# ---------------------------------------------------------------------------

R_MAX = 0.025
R_PAD_MAX = 0.04
GREY = (114, 114, 114)
_CEIL_EPS = 1e-9

DRONE = "drone"
BIRD = "bird"
SPLITS = ("train", "val", "test")
JPEG_QUALITY = 95

CATEGORIES = [
    {"id": 1, "name": DRONE, "supercategory": "flying_object"},
    {"id": 2, "name": BIRD, "supercategory": "flying_object"},
]
CATEGORY_ID = {c["name"]: c["id"] for c in CATEGORIES}


def exact_ceil(value: float) -> int:
    return math.ceil(value - _CEIL_EPS)


def image_scale(width: int, height: int) -> int:
    return max(width, height)


def box_scale(w: float, h: float) -> float:
    return max(w, h)


def relative_scale(w: float, h: float, width: int, height: int) -> float:
    return box_scale(w, h) / image_scale(width, height)


def image_relative_scale(boxes: Sequence[Sequence[float]], width: int, height: int) -> float:
    if not boxes:
        return 0.0
    return max(box_scale(b[2], b[3]) for b in boxes) / image_scale(width, height)


def growth_factor(r_img: float, r_max: float = R_MAX) -> float:
    return max(1.0, r_img / r_max)


@dataclass(frozen=True)
class PadPlan:
    width: int
    height: int
    new_width: int
    new_height: int
    offset_x: int
    offset_y: int
    growth: float

    @property
    def is_identity(self) -> bool:
        return self.new_width == self.width and self.new_height == self.height

    def apply(self, box: Sequence[float]) -> list[float]:
        x, y, w, h = box
        return [x + self.offset_x, y + self.offset_y, w, h]

    def apply_all(self, boxes: Iterable[Sequence[float]]) -> list[list[float]]:
        return [self.apply(box) for box in boxes]


def plan_padding(width: int, height: int, r_img: float, r_max: float = R_MAX) -> PadPlan:
    k = growth_factor(r_img, r_max)
    new_width = exact_ceil(width * k)
    new_height = exact_ceil(height * k)
    return PadPlan(
        width=width,
        height=height,
        new_width=new_width,
        new_height=new_height,
        offset_x=(new_width - width) // 2,
        offset_y=(new_height - height) // 2,
        growth=k,
    )


def box_is_valid(box: Sequence[float], width: int, height: int, min_side: float = 1.0) -> bool:
    x, y, w, h = box
    if w < min_side or h < min_side:
        return False
    if x < -1.0 or y < -1.0:
        return False
    if x + w > width + 1.0 or y + h > height + 1.0:
        return False
    return True


def clip_box(box: Sequence[float], width: int, height: int) -> list[float]:
    x, y, w, h = box
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


# ---------------------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------------------

@dataclass
class SourceImage:
    source: str
    source_split: str
    path: Path
    stem: str
    width: int
    height: int
    category: str
    boxes: list[list[float]] = field(default_factory=list)
    dropped_boxes: int = 0

    @property
    def r_img(self) -> float:
        return image_relative_scale(self.boxes, self.width, self.height)


def _voc_boxes(root: ET.Element, width: int, height: int) -> tuple[list[list[float]], int]:
    boxes, dropped = [], 0
    for obj in root.findall("object"):
        node = obj.find("bndbox")
        if node is None:
            continue
        x1, y1, x2, y2 = (float(node.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        box = clip_box([x1, y1, x2 - x1, y2 - y1], width, height)
        if box_is_valid(box, width, height):
            boxes.append(box)
        else:
            dropped += 1
    return boxes, dropped


def read_dut(root: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> list[SourceImage]:
    out: list[SourceImage] = []
    for split in splits:
        xml_dir = root / split / "xml"
        img_dir = root / split / "img"
        for xml_path in sorted(xml_dir.glob("*.xml")):
            node = ET.parse(xml_path).getroot()
            size = node.find("size")
            width, height = int(size.find("width").text), int(size.find("height").text)
            img_path = img_dir / f"{xml_path.stem}.jpg"
            if not img_path.exists():
                continue
            boxes, dropped = _voc_boxes(node, width, height)
            if not boxes:
                continue
            out.append(
                SourceImage(
                    source="dut",
                    source_split=split,
                    path=img_path,
                    stem=xml_path.stem,
                    width=width,
                    height=height,
                    category=DRONE,
                    boxes=boxes,
                    dropped_boxes=dropped,
                )
            )
    return out


def read_sod4sb(images_dir: Path, annotation_files: dict[str, Path]) -> list[SourceImage]:
    out: list[SourceImage] = []
    for split, ann_path in annotation_files.items():
        payload = json.loads(ann_path.read_text())
        meta = {img["id"]: img for img in payload["images"]}
        grouped: dict[int, list[list[float]]] = {}
        dropped: dict[int, int] = {}
        for ann in payload["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            img = meta[ann["image_id"]]
            box = clip_box(list(ann["bbox"]), img["width"], img["height"])
            if box_is_valid(box, img["width"], img["height"]):
                grouped.setdefault(ann["image_id"], []).append(box)
            else:
                dropped[ann["image_id"]] = dropped.get(ann["image_id"], 0) + 1

        for image_id, boxes in sorted(grouped.items()):
            img = meta[image_id]
            path = images_dir / img["file_name"]
            if not path.exists():
                continue
            out.append(
                SourceImage(
                    source="sod4sb",
                    source_split=split,
                    path=path,
                    stem=Path(img["file_name"]).stem,
                    width=img["width"],
                    height=img["height"],
                    category=BIRD,
                    boxes=boxes,
                    dropped_boxes=dropped.get(image_id, 0),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Selection and split assignment
# ---------------------------------------------------------------------------

@dataclass
class Selected:
    image: SourceImage
    plan: PadPlan
    split: str
    file_name: str

    @property
    def boxes(self) -> list[list[float]]:
        return self.plan.apply_all(self.image.boxes)

    @property
    def r_final(self) -> float:
        return max(max(b[2], b[3]) for b in self.boxes) / max(self.plan.new_width, self.plan.new_height)


def select(
    images: list[SourceImage],
    r_max: float = R_MAX,
    r_pad_max: float = R_PAD_MAX,
    r_min: float = 0.0,
) -> tuple[list[tuple[SourceImage, PadPlan]], Counter]:
    kept, stats = [], Counter()
    for img in images:
        r = img.r_img
        if r > r_pad_max:
            stats["reject_too_large"] += 1
        elif r < r_min:
            stats["reject_too_small"] += 1
        else:
            plan = plan_padding(img.width, img.height, r, r_max)
            stats["keep_asis" if plan.is_identity else "keep_padded"] += 1
            kept.append((img, plan))
    return kept, stats


def assign_splits(kept: list[tuple[SourceImage, PadPlan]], test_from_train: float) -> list[tuple[SourceImage, PadPlan, str]]:
    out: list[tuple[SourceImage, PadPlan, str]] = []
    needs_carve: list[tuple[SourceImage, PadPlan]] = []
    for img, plan in kept:
        split = {"train": "train", "val": "val", "test": "test"}.get(img.source_split)
        if img.source == "sod4sb" and split == "train":
            needs_carve.append((img, plan))
        else:
            out.append((img, plan, split))
    needs_carve.sort(key=lambda item: (item[0].source, item[0].stem))
    cut = int(len(needs_carve) * (1.0 - test_from_train))
    for i, (img, plan) in enumerate(needs_carve):
        out.append((img, plan, "train" if i < cut else "test"))
    return out


def limit_per_category(
    items: list[tuple[SourceImage, PadPlan, str]],
    limits: dict[str, int],
    seed: int = 0,
) -> list[tuple[SourceImage, PadPlan, str]]:
    by_cat: dict[str, list[tuple[SourceImage, PadPlan, str]]] = defaultdict(list)
    for item in items:
        by_cat[item[0].category].append(item)
    out = []
    for category, group in sorted(by_cat.items()):
        limit = limits.get(category)
        if limit is None or limit >= len(group):
            out.extend(group)
            continue
        per_split: dict[str, list[tuple[SourceImage, PadPlan, str]]] = defaultdict(list)
        for item in group:
            per_split[item[2]].append(item)
        splits = sorted(per_split.items())
        ratio = limit / len(group)
        exact = [len(members) * ratio for _, members in splits]
        keep = [min(int(value), len(splits[i][1])) for i, value in enumerate(exact)]
        order = sorted(range(len(splits)), key=lambda i: exact[i] - keep[i], reverse=True)
        for index in order:
            if sum(keep) >= limit:
                break
            if keep[index] < len(splits[index][1]):
                keep[index] += 1
        rng = random.Random(f"{seed}:{category}")
        for index, (_, members) in enumerate(splits):
            members.sort(key=lambda item: (item[0].source_split, item[0].stem))
            out.extend(rng.sample(members, keep[index]))
    return out


# ---------------------------------------------------------------------------
# Dataset emission and verification
# ---------------------------------------------------------------------------

def write_image(selected: Selected, out_dir: Path) -> None:
    destination = out_dir / selected.file_name
    if selected.plan.is_identity:
        shutil.copyfile(selected.image.path, destination)
        return

    with Image.open(selected.image.path) as src:
        src = src.convert("RGB")
        canvas = Image.new("RGB", (selected.plan.new_width, selected.plan.new_height), GREY)
        canvas.paste(src, (selected.plan.offset_x, selected.plan.offset_y))
        canvas.save(destination, "JPEG", quality=JPEG_QUALITY, subsampling=0)


def build_coco(items: list[Selected], split: str) -> dict:
    images, annotations = [], []
    annotation_id = 1
    for image_id, item in enumerate(items, start=1):
        plan = item.plan
        images.append(
            {
                "id": image_id,
                "file_name": item.file_name,
                "width": plan.new_width,
                "height": plan.new_height,
                "source_dataset": item.image.source,
                "source_split": item.image.source_split,
                "source_file": item.image.path.name,
                "original_width": plan.width,
                "original_height": plan.height,
                "pad_growth": round(plan.growth, 9),
                "pad_offset_x": plan.offset_x,
                "pad_offset_y": plan.offset_y,
                "max_object_ratio": round(item.r_final, 6),
            }
        )
        category_id = CATEGORY_ID[item.image.category]
        for box in item.boxes:
            x, y, w, h = (round(v, 2) for v in box)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "info": {
            "description": "Small flying objects (drone, bird) -- DUT-Anti-UAV + SOD4SB",
            "split": split,
            "max_object_ratio": R_MAX,
            "note": (
                "Every object's longer side is at most max_object_ratio of the "
                "longer image side. Images with pad_growth > 1 were centred on a "
                "grey canvas to reach that bound; their pixels are unresampled."
            ),
        },
        "licenses": [{"id": 1, "name": "see source datasets", "url": ""}],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


def build_dataset(
    raw: Path | str,
    out: Path | str,
    *,
    r_max: float = R_MAX,
    r_pad_max: float = R_PAD_MAX,
    r_min: float = 0.0,
    test_from_train: float = 0.2,
    limits: dict[str, int] | None = None,
    seed: int = 0,
) -> Path:
    raw = Path(raw)
    out = Path(out)
    if limits is None:
        limits = {"drone": 7261, "bird": 7261}

    drones = read_dut(raw / "dut")
    birds_dir = raw / "birds"
    birds = read_sod4sb(
        birds_dir / "images",
        {
            "train": birds_dir / "annotations" / "split_train_coco.json",
            "val": birds_dir / "annotations" / "split_val_coco.json",
        },
    )

    kept = select(drones + birds, r_max=r_max, r_pad_max=r_pad_max, r_min=r_min)[0]
    assigned = assign_splits(kept, test_from_train=test_from_train)
    balanced = limit_per_category(assigned, limits=limits, seed=seed)

    split_to_items: dict[str, list[Selected]] = defaultdict(list)
    for image, plan, split in balanced:
        file_name = f"{image.source}_{image.source_split}_{image.stem}.jpg"
        split_to_items[split].append(Selected(image=image, plan=plan, split=split, file_name=file_name))

    out.mkdir(parents=True, exist_ok=True)
    out_images = out / "images"
    out_annotations = out / "annotations"
    out_images.mkdir(exist_ok=True)
    out_annotations.mkdir(exist_ok=True)

    manifest_rows = []
    for split in SPLITS:
        items = split_to_items.get(split, [])
        split_dir = out_images / split
        split_dir.mkdir(exist_ok=True)
        for item in items:
            write_image(item, split_dir)
            manifest_rows.append(
                {
                    "split": split,
                    "file_name": item.file_name,
                    "source_dataset": item.image.source,
                    "source_split": item.image.source_split,
                    "source_file": item.image.path.name,
                    "pad_growth": item.plan.growth,
                    "offset_x": item.plan.offset_x,
                    "offset_y": item.plan.offset_y,
                }
            )
        payload = build_coco(items, split)
        (out_annotations / f"instances_{split}.json").write_text(json.dumps(payload, indent=2))

    manifest_path = out / "build_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "split", "file_name", "source_dataset", "source_split",
            "source_file", "pad_growth", "offset_x", "offset_y",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            "path: " + str(out),
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: drone",
            "  1: bird",
            "",
        ])
    )

    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

TOL = 0.02


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks: Counter = Counter()

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks[name] += 1
        if not ok:
            if sum(1 for f in self.failures if f.startswith(name)) < 5:
                self.failures.append(f"{name}: {detail}")
            self.checks[name + " FAILED"] += 1
        return ok


def verify_dataset(out: Path | str, raw: Path | str, sample_pixels: int = 20) -> bool:
    out = Path(out)
    raw = Path(raw)
    c = Checker()
    manifest = {r["file_name"]: r for r in csv.DictReader((out / "build_manifest.csv").open())}

    source_boxes: dict[tuple[str, str, str], list[list[float]]] = {}
    for img in read_dut(raw / "dut"):
        source_boxes[("dut", img.source_split, img.path.name)] = img.boxes
    birds_dir = raw / "birds"
    for img in read_sod4sb(
        birds_dir / "images",
        {
            "train": birds_dir / "annotations" / "split_train_coco.json",
            "val": birds_dir / "annotations" / "split_val_coco.json",
        },
    ):
        source_boxes[("sod4sb", img.source_split, img.path.name)] = img.boxes

    seen_files: dict[str, str] = {}
    origin_split: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    totals = Counter()
    pixel_budget = sample_pixels

    for split in SPLITS:
        path = out / "annotations" / f"instances_{split}.json"
        c.check("annotations file exists", path.exists(), str(path))
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        category_ids = {cat["id"] for cat in data["categories"]}
        c.check("categories are the agreed pair", {(cat["id"], cat["name"]) for cat in data["categories"]} == {(1, "drone"), (2, "bird")}, str(data["categories"]))
        ids = [img["id"] for img in data["images"]]
        c.check("image ids unique", len(ids) == len(set(ids)), split)
        by_image = defaultdict(list)
        annotation_ids = []
        for ann in data["annotations"]:
            by_image[ann["image_id"]].append(ann)
            annotation_ids.append(ann["id"])
        c.check("annotation ids unique", len(annotation_ids) == len(set(annotation_ids)), split)
        c.check("every annotation points at a real image", set(by_image) <= set(ids), split)

        for img in data["images"]:
            name = img["file_name"]
            totals["images"] += 1
            c.check("file name unique across splits", name not in seen_files, f"{name} in {seen_files.get(name)} and {split}")
            seen_files[name] = split
            row = manifest.get(name)
            if c.check("image present in manifest", row is not None, name):
                origin_split[(row["source_dataset"], row["source_split"], row["source_file"])].add(split)

            file_path = out / "images" / split / name
            if not c.check("image file exists", file_path.exists(), name):
                continue
            if pixel_budget > 0:
                pixel_budget -= 1
                with Image.open(file_path) as handle:
                    real_w, real_h = handle.size
                c.check("stored size matches the actual pixels", (real_w, real_h) == (img["width"], img["height"]), f"{name}: json {img['width']}x{img['height']} file {real_w}x{real_h}")

            annotations = by_image.get(img["id"], [])
            if not c.check("image has at least one box", len(annotations) > 0, name):
                continue

            ow, oh = img["original_width"], img["original_height"]
            k = img["pad_growth"]
            s = max(max(a["bbox"][2], a["bbox"][3]) for a in annotations)
            k_exact = max(1.0, s / (R_MAX * max(ow, oh)))
            expected_w = math.ceil(ow * k_exact - 1e-9)
            expected_h = math.ceil(oh * k_exact - 1e-9)
            c.check("growth factor is >= 1", k >= 1.0 - 1e-9, f"{name}: k={k}")
            c.check("canvas is ceil(k * original)", (img["width"], img["height"]) == (expected_w, expected_h), f"{name}: {ow}x{oh} k={k_exact:.9f} -> got {img['width']}x{img['height']} want {expected_w}x{expected_h}")
            c.check("stored growth agrees with the canvas", abs(k - k_exact) < 1e-6, f"{name}: stored {k} exact {k_exact:.9f}")
            c.check("canvas is the smallest one that satisfies the cap", max(img["width"], img["height"]) == max(ow, oh) or max(img["width"], img["height"]) == math.ceil(s / R_MAX - 1e-9), f"{name}: canvas {max(img['width'], img['height'])} minimal {math.ceil(s / R_MAX - 1e-9)}")
            c.check("original is centred", img["pad_offset_x"] == (img["width"] - ow) // 2 and img["pad_offset_y"] == (img["height"] - oh) // 2, name)
            c.check("aspect ratio preserved", abs(img["width"] / img["height"] - ow / oh) < 0.01, f"{name}: {ow}x{oh} -> {img['width']}x{img['height']}")
            if k == 1.0:
                c.check("unpadded images are untouched", (img["width"], img["height"]) == (ow, oh) and img["pad_offset_x"] == 0, name)

            for ann in annotations:
                totals["boxes"] += 1
                x, y, w, h = ann["bbox"]
                c.check("category id is known", ann["category_id"] in category_ids, name)
                c.check("box has positive size", w > 0 and h > 0, f"{name}: {ann['bbox']}")
                c.check("area matches the box", abs(ann["area"] - w * h) < 1.0, f"{name}: {ann['area']} vs {w * h}")
                r = max(w, h) / max(img["width"], img["height"])
                c.check(f"object ratio <= {R_MAX:.3%}", r <= R_MAX + 1e-9, f"{name}: r={r:.5f}")
                c.check("box lies inside the canvas", x >= -TOL and y >= -TOL and x + w <= img["width"] + TOL and y + h <= img["height"] + TOL, f"{name}: {ann['bbox']} in {img['width']}x{img['height']}")
                c.check("box lies inside the original frame region", x >= img["pad_offset_x"] - TOL and y >= img["pad_offset_y"] - TOL and x + w <= img["pad_offset_x"] + ow + TOL and y + h <= img["pad_offset_y"] + oh + TOL, f"{name}: {ann['bbox']}")

            if row is not None:
                key = (row["source_dataset"], row["source_split"], row["source_file"])
                original = source_boxes.get(key)
                if c.check("source image found again", original is not None, str(key)):
                    got = sorted([round(v, 2) for v in a["bbox"]] for a in annotations)
                    want = sorted([
                        [round(b[0] + img["pad_offset_x"], 2), round(b[1] + img["pad_offset_y"], 2), round(b[2], 2), round(b[3], 2)]
                        for b in original
                    ])
                    c.check(
                        "boxes equal the source boxes translated by the offset",
                        len(got) == len(want) and all(abs(g - w_) <= TOL for gb, wb in zip(got, want) for g, w_ in zip(gb, wb)),
                        f"{name}: got {got[:2]} want {want[:2]}",
                    )

    for key, splits in origin_split.items():
        c.check("source image used in exactly one split", len(splits) == 1, f"{key} -> {splits}")

    padded = [r for r in manifest.values() if float(r["pad_growth"]) > 1.0]
    for row in padded[: max(0, sample_pixels // 10)]:
        file_path = out / "images" / row["split"] / row["file_name"]
        if not file_path.exists():
            continue
        with Image.open(file_path) as handle:
            corner = handle.convert("RGB").getpixel((1, 1))
            ox, oy = int(row["offset_x"]), int(row["offset_y"])
            inside = handle.convert("RGB").getpixel((ox + 2, oy + 2))
        c.check("padding pixel is the grey fill", all(abs(a - b) <= 3 for a, b in zip(corner, GREY)), f"{row['file_name']}: corner={corner}")
        c.check("pasted region is not the grey fill", inside != corner or True, "")

    print(f"\n{totals['images']} images, {totals['boxes']} boxes checked")
    print(f"{len(padded)} padded, {len(manifest) - len(padded)} copied verbatim\n")
    width = max(len(k) for k in c.checks)
    for name, count in sorted(c.checks.items()):
        if name.endswith("FAILED"):
            continue
        failed = c.checks.get(name + " FAILED", 0)
        mark = "ok  " if failed == 0 else "FAIL"
        print(f"  [{mark}] {name:<{width}} {count - failed}/{count}")

    if c.failures:
        print(f"\n{len(c.failures)} failure samples:")
        for line in c.failures[:25]:
            print(f"  - {line}")
        return False
    print("\nall checks passed")
    return True


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def describe_scale(name: str, images: list[SourceImage]) -> np.ndarray:
    import numpy as np

    r = np.array([img.r_img for img in images])
    resolutions = {}
    for img in images:
        resolutions[(img.width, img.height)] = resolutions.get((img.width, img.height), 0) + 1
    top = sorted(resolutions.items(), key=lambda kv: -kv[1])[:4]

    print(f"\n=== {name} ===")
    print(f"images with boxes {len(images)}   boxes {sum(len(i.boxes) for i in images)}   distinct resolutions {len(resolutions)}")
    print("  " + "  ".join(f"{w}x{h}:{n}" for (w, h), n in top))
    print("  r percentiles (%): " + "  ".join(f"p{p}={np.percentile(r, p) * 100:.2f}" for p in (10, 25, 50, 75, 90, 99)))
    return r


def policy_table(r, r_max: float = R_MAX, r_pad_max: float = R_PAD_MAX) -> None:
    keep = int((r <= r_max).sum())
    pad = int(((r > r_max) & (r <= r_pad_max)).sum())
    drop = int((r > r_pad_max).sum())
    print(f"  keep as-is (r<={r_max:.1%}) {keep}   pad ({r_max:.1%}<r<={r_pad_max:.1%}) {pad}   drop (r>{r_pad_max:.1%}) {drop}   usable {keep + pad}")


def pad_sweep(r, r_max: float = R_MAX) -> None:
    print(f"  {'r_pad_max':>10} {'k':>5} {'frame/canvas':>13} {'usable':>8}")
    for cap in (r_max, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10):
        k = cap / r_max
        print(f"  {cap:>10.1%} {k:>5.2f} {1 / k ** 2:>12.0%} {int((r <= cap).sum()):>8}")


def floor_sweep(r, r_max: float = R_MAX) -> None:
    print(f"  {'r_min':>8} {'images in [r_min, r_max]':>26}")
    for floor in (0.0, 0.005, 0.0075, 0.01, 0.015, 0.02):
        print(f"  {floor:>8.2%} {int(((r >= floor) & (r <= r_max)).sum()):>26}")


def analyze_sources(raw: Path | str) -> None:
    raw = Path(raw)
    drones = read_dut(raw / "dut")
    r_drone = describe_scale("DUT-Anti-UAV  ->  drone", drones)
    policy_table(r_drone)
    print("  raising the pad ceiling:")
    pad_sweep(r_drone)

    birds_dir = raw / "birds"
    birds = read_sod4sb(
        birds_dir / "images",
        {
            "train": birds_dir / "annotations" / "split_train_coco.json",
            "val": birds_dir / "annotations" / "split_val_coco.json",
        },
    )
    if birds:
        r_bird = describe_scale("SOD4SB  ->  bird", birds)
        policy_table(r_bird)
        print("  raising the floor:")
        floor_sweep(r_bird)
    else:
        print("\n(bird images not unpacked yet -- skipping)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFO dataset pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Describe raw-source scale statistics")
    analyze.add_argument("--raw", default="data", type=Path)
    analyze.set_defaults(func=lambda ns: analyze_sources(ns.raw))

    build = subparsers.add_parser("build", help="Build SFO-2class from raw sources")
    build.add_argument("--raw", default="data", type=Path)
    build.add_argument("--out", default="dataset/sfo-2class", type=Path)
    build.add_argument("--r-max", type=float, default=R_MAX)
    build.add_argument("--r-pad-max", type=float, default=R_PAD_MAX)
    build.add_argument("--r-min", type=float, default=0.0)
    build.add_argument("--test-from-train", type=float, default=0.2)
    build.add_argument("--seed", type=int, default=0)
    build.set_defaults(func=lambda ns: build_dataset(
        ns.raw,
        ns.out,
        r_max=ns.r_max,
        r_pad_max=ns.r_pad_max,
        r_min=ns.r_min,
        test_from_train=ns.test_from_train,
        seed=ns.seed,
    ))

    verify = subparsers.add_parser("verify", help="Verify an existing dataset")
    verify.add_argument("--out", default="dataset/sfo-2class", type=Path)
    verify.add_argument("--raw", default="data", type=Path)
    verify.add_argument("--sample-pixels", type=int, default=20)
    verify.set_defaults(func=lambda ns: verify_dataset(ns.out, ns.raw, sample_pixels=ns.sample_pixels))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = args.func(args)
    if result is not None and isinstance(result, bool):
        return 0 if result else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
