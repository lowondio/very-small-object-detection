#!/usr/bin/env bash
# Package a built dataset as a single archive, tracked with git LFS.
#
#   bash dataset_preparation/package.sh [DATASET_DIR] [OUT_DIR]
#
# One file, images and labels together: dist/<name>.tar
#
# Not gzipped. The payload is ~99% JPEG, which is already compressed, so gzip
# would spend minutes to save well under a percent.
#
# The archive is ~9.3 GB, past GitHub's 100 MB per-file limit, so it goes in
# via git LFS (see .gitattributes). Note that LFS bandwidth and storage are
# metered per account -- a file this size needs a paid data pack.

set -euo pipefail
SRC="${1:-dataset/sfo-2class}"
OUT="${2:-dist}"
NAME="$(basename "$SRC")"

[[ -d "$SRC" ]] || { echo "no such dataset: $SRC" >&2; exit 1; }
mkdir -p "$OUT"

echo "==> archiving $SRC"
tar -cf "$OUT/$NAME.tar" -C "$(dirname "$SRC")" "$NAME"

echo "==> checksum"
( cd "$OUT" && shasum -a 256 "$NAME.tar" > "$NAME.tar.sha256" )

echo
ls -lh "$OUT"
echo
echo "track with LFS, once per clone:"
echo "  git lfs install && git lfs track 'dist/*.tar' && git add .gitattributes"
echo "extract:  tar -xf $NAME.tar"
echo "verify:   shasum -a 256 -c $NAME.tar.sha256"
