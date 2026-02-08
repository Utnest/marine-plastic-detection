#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR="${1:-datasets/images}"
LABEL_DIR="${2:-datasets/labels}"
REQUIRE_NON_EMPTY="${3:-true}"

echo "Checking dataset structure..."
echo "Image directory: ${IMAGE_DIR}"
echo "Label directory: ${LABEL_DIR}"

if [[ ! -d "${IMAGE_DIR}" ]]; then
  echo "::error::Image directory does not exist: ${IMAGE_DIR}"
  exit 1
fi

if [[ ! -d "${LABEL_DIR}" ]]; then
  echo "::error::Label directory does not exist: ${LABEL_DIR}"
  exit 1
fi

python3 - "${IMAGE_DIR}" "${LABEL_DIR}" "${REQUIRE_NON_EMPTY}" <<'PY'
import sys
from pathlib import Path

image_dir = Path(sys.argv[1]).resolve()
label_dir = Path(sys.argv[2]).resolve()
require_non_empty = sys.argv[3].strip().lower() == "true"

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

images = sorted(
    p for p in image_dir.rglob("*")
    if p.is_file() and p.suffix.lower() in image_exts
)

if not images:
    print(f"::error::No images found in {image_dir}")
    sys.exit(1)

image_bases = set()
missing_labels = []
empty_annotations = []
invalid_rows = []

for img in images:
    rel = img.relative_to(image_dir)
    base = rel.with_suffix("")
    image_bases.add(base.as_posix())

    label_path = (label_dir / rel).with_suffix(".txt")
    if not label_path.exists():
        missing_labels.append(label_path)
        continue

    raw_lines = label_path.read_text(encoding="utf-8").splitlines()
    lines = [ln.strip() for ln in raw_lines if ln.strip()]

    if require_non_empty and not lines:
        empty_annotations.append(label_path)
        continue

    for idx, row in enumerate(lines, start=1):
        parts = row.split()
        if len(parts) != 5:
            invalid_rows.append(
                (label_path, idx, f"expected 5 columns, found {len(parts)}")
            )
            continue

        cls_raw, x_raw, y_raw, w_raw, h_raw = parts

        try:
            cls_float = float(cls_raw)
            if not cls_float.is_integer() or cls_float < 0:
                raise ValueError()
        except ValueError:
            invalid_rows.append((label_path, idx, "class id must be an integer >= 0"))
            continue

        try:
            x, y, w, h = map(float, (x_raw, y_raw, w_raw, h_raw))
        except ValueError:
            invalid_rows.append((label_path, idx, "coordinates must be numeric"))
            continue

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            invalid_rows.append(
                (label_path, idx, "x_center and y_center must be in [0, 1]")
            )
            continue

        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            invalid_rows.append((label_path, idx, "width and height must be in (0, 1]"))
            continue

label_files = sorted(p for p in label_dir.rglob("*.txt") if p.is_file())
orphan_labels = []
for label in label_files:
    rel = label.relative_to(label_dir)
    base = rel.with_suffix("").as_posix()
    if base not in image_bases:
        orphan_labels.append(label)

has_errors = False

if missing_labels:
    has_errors = True
    print("::error::Missing label files:")
    for path in missing_labels[:100]:
        print(f"  - {path}")
    if len(missing_labels) > 100:
        print(f"  ...and {len(missing_labels) - 100} more")

if empty_annotations:
    has_errors = True
    print("::error::Empty annotation files:")
    for path in empty_annotations[:100]:
        print(f"  - {path}")
    if len(empty_annotations) > 100:
        print(f"  ...and {len(empty_annotations) - 100} more")

if invalid_rows:
    has_errors = True
    print("::error::Invalid YOLO rows:")
    for path, row_num, reason in invalid_rows[:100]:
        print(f"  - {path}:{row_num} -> {reason}")
    if len(invalid_rows) > 100:
        print(f"  ...and {len(invalid_rows) - 100} more")

if orphan_labels:
    has_errors = True
    print("::error::Orphan label files (no matching image):")
    for path in orphan_labels[:100]:
        print(f"  - {path}")
    if len(orphan_labels) > 100:
        print(f"  ...and {len(orphan_labels) - 100} more")

if has_errors:
    sys.exit(1)

print(f"Dataset OK: {len(images)} images, {len(label_files)} label files validated.")
PY
