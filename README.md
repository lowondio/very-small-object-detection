# Very Small Object Detection

This repository implements a small-object detection pipeline for two classes: `drone` and `bird`. The project is designed around a hard size constraint: every object is limited to at most 2.5% of the longer image side. This makes the task fundamentally different from standard detection benchmarks, where objects are usually large enough to survive resizing.

<img width="1778" height="1000" alt="image" src="https://github.com/user-attachments/assets/d34d1709-d859-4376-a716-f2b40a65769e" />

## Why this project exists

At this scale, an object may only occupy a handful of pixels. The exact number depends on resolution:

| image size | max object side |
|---|---:|
| 4K | ~100 px |
| FullHD | ~48 px |
| HD | ~32 px |
| resized to 640×640 | ~16 px |

Two consequences define the entire project:

1. Whole-frame resizing destroys the object. A bird in a 4K frame can shrink to about 3 px wide after letterboxing to 640×640, which is below a typical stride-8 feature-cell size.
2. Standard detection metrics become misleading at this scale. A fixed IoU threshold such as 0.5 is too strict for a 6 px object; the metric used here is intentionally size-aware and class-specific.

The repository therefore treats the task as a research pipeline rather than a single model implementation.

## Project goals

- build a clean dataset of small flying objects under a strict relative-size cap
- support multiple detector families with a shared evaluation interface
- evaluate results with small-object-aware metrics
- keep the training and evaluation flow reproducible and structured

## Repository structure

```text
.
├── dataset_preparation/     dataset packaging and validation logic
├── detection/               core model, metrics, and evaluation code
├── notebooks/               optional exploratory research notebooks
├── src/
│   └── sfod/                structured CLI entry point
├── pyproject.toml           package metadata and console entry points
├── requirements.txt         project dependencies
├── run_metrics_colab.py     metrics-only evaluation helper for Colab
├── example_image_birds.png
├── example_video_birds.MP4
├── README.md
└── LICENSE                  (if present in your checkout)
```

## Main components

### dataset_preparation

This package builds and validates the dataset used for training and evaluation.

It is responsible for:
- reading raw sources
- normalizing all boxes into a common COCO-like format
- enforcing the size cap and padding policy
- splitting data into train/val/test
- validating the final output on disk

Core responsibilities live in one compact dataset package rather than many scattered scripts.

### detection

This is the core ML and evaluation layer.

It contains:
- model configuration objects
- dataset access utilities
- detector adapters
- custom metric logic for small objects
- evaluation entry points
- training-time tracking and reporting

The key idea is that all detector families share the same contract and the same evaluation path.

### src/sfod

The project now exposes a small, explicit CLI for regular workflows:

```bash
pip install -e .
sfod prepare --dataset-dir /content/sfo-2class
sfod train --detector yolo --dataset-dir /content/sfo-2class --output-dir /content/drive/MyDrive/sfo/runs
sfod evaluate --detector yolo --dataset-dir /content/sfo-2class --weights /content/drive/MyDrive/sfo/runs/.../best.pt --output-dir /content/drive/MyDrive/sfo/runs/.../eval/standalone
```

This is the preferred runtime interface for the project.

## Dataset

The dataset is built from public sources and filtered to the small-object regime.

Included sources include:
- DUT-Anti-UAV for drones
- SOD4SB / MVA2023 for birds
- additional drone sources depending on the training setup

The dataset is assembled as COCO-style annotations with two classes:
- `drone`
- `bird`

Important design choices:
- the project operates on relative object scale, not absolute px size
- oversized images are padded on a neutral grey canvas instead of being rescaled
- object sizes are preserved in image space; only the canvas grows

## Detection setup

The code is built around a detector abstraction, so the evaluation pipeline stays identical even when the underlying model changes. In practice, this means the project can compare:
- YOLO-based detectors
- D-FINE-style detectors
- other model families added through the same interface

This is a major design win: the dataset, scoring, and output reporting are decoupled from the detector implementation details.

## Metrics

The evaluation logic is intentionally different from a generic COCO metric implementation.

It uses:
- relative-size bins instead of absolute pixel bins
- per-class reporting rather than a single mean score
- cross-class confusion accounting
- adaptive IoU thresholds based on object scale

This makes the evaluation much more appropriate for tiny objects, where a fixed small IoU threshold can be dominated by annotation uncertainty and image rescaling effects.

## Getting started

### 1. Install dependencies

```bash
python -m venv .venv
. .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare the dataset

```bash
python -m dataset_preparation analyze --raw data
python -m dataset_preparation build --raw data --out dataset/sfo-2class
```

### 3. Train a detector

```bash
sfod train --detector yolo --dataset-dir /content/sfo-2class --output-dir /content/drive/MyDrive/sfo/runs
```

### 4. Evaluate a checkpoint

```bash
sfod evaluate \
  --detector yolo \
  --dataset-dir /content/sfo-2class \
  --weights /content/drive/MyDrive/sfo/runs/.../train/weights/best.pt \
  --output-dir /content/drive/MyDrive/sfo/runs/.../eval/standalone
```

### 5. Run inference on a video

```bash
sfod video \
  --source example_video_birds.MP4 \
  --weights /path/to/best.pt \
  --output-dir runs/video_output \
  --conf 0.001 \
  --iou 0.7 \
  --max-det 300 \
  --imgsz 640
```

You can also run the same workflow via the direct script:

```bash
python run_video.py \
  --source example_video_birds.MP4 \
  --weights /path/to/best.pt \
  --output-dir runs/video_output
```

This saves the annotated video to the chosen output directory and prints the generated file paths.

## Notes on notebooks

The `notebooks/` folder is optional. It is useful for exploratory analysis, visual validation, and experiment tracking, but it is not required for the training and evaluation pipeline itself.

## Summary

This project is a research-oriented detection pipeline for very small airborne objects. Its main strengths are:
- strong data-engineering discipline
- detector-agnostic evaluation
- small-object-aware metrics
- a clean separation between dataset preparation, model training, and scoring

If you want to work with it productively, the right mental model is: this is a benchmark and research tool for small-object detection, not a general-purpose app or web service.
