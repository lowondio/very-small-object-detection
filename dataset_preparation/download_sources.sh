#!/usr/bin/env bash
# Fetch the two raw sources into data/. Idempotent: skips what is already there.
#
#   bash dataset_preparation/download_sources.sh [DEST]     # DEST defaults to ./data
#
# Files are pulled one id at a time. `gdown --folder` on these folders gets
# throttled by Drive ("cannot retrieve public link ... many accesses"), and the
# 62 GB drone2021 subset of SOD4SB is not needed -- only the 2023 folder is.

set -euo pipefail
DEST="${1:-data}"
GDOWN="${GDOWN:-gdown}"

fetch() {  # fetch <file-id> <path> <human size>
  if [[ -s "$2" ]]; then echo "  have $2"; return; fi
  echo "  get  $2  ($3)"
  "$GDOWN" "$1" -O "$2"
}

# --- DUT-Anti-UAV: PASCAL VOC, images + xml bundled per split, ~1.35 GB ------
mkdir -p "$DEST/dut"
fetch 1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj "$DEST/dut/train.zip" "745 MB, 5200 imgs"
fetch 1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n "$DEST/dut/val.zip"   "372 MB, 2600 imgs"
fetch 1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0 "$DEST/dut/test.zip"  "271 MB, 2200 imgs"
for split in train val test; do
  [[ -d "$DEST/dut/$split/xml" ]] || unzip -q -o "$DEST/dut/$split.zip" -d "$DEST/dut"
done

# --- SOD4SB / MVA2023 birds: COCO, annotations separate from images, ~9.4 GB -
mkdir -p "$DEST/birds/annotations"
fetch 1uhI2WfiHoMqmuE3FqqWWOUfkZ2UZNWSl "$DEST/birds/annotations/split_train_coco.json" "5 MB"
fetch 1ty2NFkTWYQdm4q6iZH__lHlMIq4LbfWh "$DEST/birds/annotations/split_val_coco.json"   "0.6 MB"
if [[ ! -d "$DEST/birds/images" ]]; then
  fetch 1hBhkbaIzyntGqPIWmEUFGSsLVADQwTWr "$DEST/birds/images.zip" "9.4 GB, 9759 imgs"
  unzip -q -o "$DEST/birds/images.zip" -d "$DEST/birds" && rm -f "$DEST/birds/images.zip"
fi

echo
echo "raw sources ready under $DEST/"
find "$DEST" -maxdepth 2 -type d | sort
# If gdown is throttled on a big file:
#   curl -L "https://drive.google.com/uc?export=download&id=<ID>&confirm=t" -o out.zip
