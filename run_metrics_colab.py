"""
=============================================================================
 HOW TO USE THIS IN GOOGLE COLAB
=============================================================================

WHAT THIS DOES
    Scores your detector on the test split and reports, for drone and bird
    SEPARATELY (never averaged together):
      - AP, AP50, AP50-95, precision, recall, F1
      - the same numbers per object-size bin
      - how many birds the model called drones, and drones it called birds,
        in total and per size bin

WHAT YOU NEED FIRST
    1. The file `metrics.py` (from this repo, folder `detection/`).
       It only needs numpy, pandas and matplotlib. No torch, no ultralytics.
    2. The test annotations: `instances_test.json` (COCO).
    3. Your model's predictions on the test split (see FORMAT below).

STEP 1 - put the file in Colab
    Drag `metrics.py` into the Colab file browser so it lands in `/content`,
    or run this in a cell:

        !wget -O /content/metrics.py <raw github url of detection/metrics.py>

STEP 2 - get predictions
    Run your model over the test images first. Use a LOW confidence threshold
    (0.001). If you filter at 0.25 the precision-recall curve is cut off and AP
    comes out lower than it really is. Filtering happens later, automatically.

STEP 3 - paste this whole file into a cell and run it
    Edit the paths in the SETTINGS block, then run. It prints the tables and
    saves everything to OUTPUT_DIR.

=============================================================================
 FORMAT OF THE PREDICTIONS
=============================================================================

One row per predicted box. A pandas DataFrame, a CSV, or a JSON list:

    image_id           class   x1      y1      x2      y2      score
    bird_test_00001    bird    1018.0  1247.0  1040.0  1272.0  0.87
    bird_test_00001    drone   512.0   300.0   530.0   318.0   0.11
    drone_train_00042  drone   88.0    640.0   119.0   671.0   0.93

    image_id  file name without the extension ("bird_test_00001.jpg" ->
              "bird_test_00001"). This is what links a box to its image.
    class     "drone" or "bird". A number is fine too: 0 = drone, 1 = bird
              (column named `class_id`), or COCO ids 1 = drone, 2 = bird
              (column named `category_id`).
    x1 y1     top-left corner, in pixels of the ORIGINAL image
    x2 y2     bottom-right corner, in pixels of the ORIGINAL image
    score     confidence, 0 to 1

    Instead of x1/y1/x2/y2 you may give a `bbox` column holding COCO-style
    [x, y, width, height]. Both are accepted.

If you used this repo's `Detector.predict()`, you already have exactly this
table -- go straight to OPTION A below.

=============================================================================
"""

from pathlib import Path

import pandas as pd

# In Colab the file sits next to the notebook, so a plain import works.
from metrics import MetricsConfig, evaluate, load_coco_ground_truth, load_predictions

# =============================================================================
# SETTINGS - edit these
# =============================================================================

ANNOTATIONS = Path("/content/sfo-2class/annotations/instances_test.json")
PREDICTIONS = Path("/content/predictions_test.csv")
OUTPUT_DIR = Path("/content/metrics_test")

CLASS_NAMES = ("drone", "bird")  # order matters: 0 = drone, 1 = bird

# Reference resolution of the metric. LEAVE THIS AT 640 even if you trained at
# a different size. Object size enters the metric as (relative scale x this
# number), so changing it changes the thresholds and your AP stops being
# comparable with every other run. It is a property of the metric, not of your
# model.
METRIC_IMAGE_SIZE = 640

# How strict the box overlap must be for a detection to count as correct.
#   "dynamic" - the threshold adapts to object size (recommended here)
#   0.5       - the classic fixed COCO threshold
# Our objects are 2-16 px wide at this resolution. A fixed 0.5 demands a 6 px
# object be placed within 1.1 px, and a 2 px object within 0.37 px, which is
# finer than the annotation itself resolves. "dynamic" runs from 0.30 on the
# smallest objects up to 0.50 once they reach about 11 px.
IOU_THRESHOLD = "dynamic"

# The confidence at which precision, recall and the confusion counts are
# reported. "best_f1" picks it automatically, per class. Use a number (0.25)
# to pin it to a threshold you plan to ship.
OPERATING_POINT = "best_f1"

# Size bins, as a fraction of the image: longest box side / longest image side.
# 0.003 = 0.3%. This is the same binning the dataset was built with.
RATIO_BINS = (0.0, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.025)

# =============================================================================
# 1. LOAD THE GROUND TRUTH
# =============================================================================

images, ground_truth = load_coco_ground_truth(ANNOTATIONS, CLASS_NAMES)

print(f"ground truth: {len(images)} images, {len(ground_truth)} boxes")
print(ground_truth["class"].value_counts().to_string())

# =============================================================================
# 2. LOAD THE PREDICTIONS - pick one option
# =============================================================================

# ---- OPTION A: you already have a DataFrame in this notebook ----------------
# For example straight out of this repo's detector:
#
#     _, predictions = detector.predict(dataset.images_dir("test"), conf=0.001)
#
# predictions = predictions

# ---- OPTION B: a CSV or JSON file ------------------------------------------
predictions = load_predictions(PREDICTIONS, CLASS_NAMES)

# ---- OPTION C: raw Ultralytics results, if you ran YOLO yourself -----------
# rows = []
# for result in model.predict(source=TEST_IMAGES_DIR, imgsz=640,  # your training size
#                             conf=0.001, stream=True, verbose=False):
#     image_id = Path(result.path).stem
#     for box, score, class_id in zip(result.boxes.xyxy.cpu().numpy(),
#                                     result.boxes.conf.cpu().numpy(),
#                                     result.boxes.cls.cpu().numpy().astype(int)):
#         rows.append({"image_id": image_id, "class": CLASS_NAMES[class_id],
#                      "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
#                      "score": float(score)})
# predictions = pd.DataFrame(rows)

print(f"\npredictions: {len(predictions)} boxes")
print(predictions["class"].value_counts().to_string())

# =============================================================================
# 3. SCORE
# =============================================================================

config = MetricsConfig(
    class_names=CLASS_NAMES,
    ratio_bins=RATIO_BINS,
    imgsz=METRIC_IMAGE_SIZE,
    iou_threshold=IOU_THRESHOLD,
    operating_point=OPERATING_POINT,
)

report = evaluate(images, ground_truth, predictions, config)

report.print_summary()
report.save(OUTPUT_DIR)
report.plot(OUTPUT_DIR / "figures")

# =============================================================================
# 4. HOW TO READ THE OUTPUT
# =============================================================================

print(f"""
-----------------------------------------------------------------------------
Saved to {OUTPUT_DIR}

  per_class.csv           one row per class. `ap` is the headline number.
                          ap50 / ap50_95 are there to compare with papers.
  per_bin.csv             the same numbers split by object size. This is where
                          you see the model fall apart on the smallest objects.
  confusion.csv           birds called drones, drones called birds, and misses.
  confusion_per_bin.csv   the same, per size bin.
  pr_curves.csv           the full precision-recall curve for each class.
  figures/                PR curve, AP per bin, recall per bin, confusion.

Reading the tables:

  `bin` is the object size as a share of the image, and `px_at_imgsz` is
  roughly how many pixels that is once the image is resized to {METRIC_IMAGE_SIZE}.
  A 0.3-0.5% object is about 2-3 pixels across at {METRIC_IMAGE_SIZE} - if AP is near
  zero there, the object was probably too small to survive the resize, which
  is a resolution problem, not a model problem.

  In confusion.csv, `correct` + `confused_as_*` + `missed` = n_ground_truth.
  A box is `confused_as_X` when the right class missed it but class X put a
  box on it. High bird -> drone means false alarms in a drone alerting system.

  Nothing here is averaged across drone and bird. There is no single mAP on
  purpose: the two classes behave very differently and the mean hides it.
-----------------------------------------------------------------------------
""")

# Quick per-class summary for logging or a results table.
print("headline:", {key: round(value, 4) for key, value in report.headline().items()})
