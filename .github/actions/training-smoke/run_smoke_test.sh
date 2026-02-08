#!/usr/bin/env bash
set -euo pipefail

DATA_CONFIG="${1:-datasets/data.yaml}"
TASK="${2:-detect}"
MODEL="${3:-yolov8n.pt}"
EPOCHS="${4:-1}"
MAX_IMAGES="${5:-5}"
IMGSZ="${6:-640}"
PROJECT_DIR="${7:-ci-smoke-runs}"
RUN_NAME="${8:-smoke}"

if [[ ! -f "${DATA_CONFIG}" ]]; then
  echo "::error::Dataset YAML not found: ${DATA_CONFIG}"
  exit 1
fi

SMOKE_DIR=".ci/smoke-dataset"
mkdir -p "${SMOKE_DIR}"

SMOKE_YAML=$(python3 - "${DATA_CONFIG}" "${MAX_IMAGES}" "${SMOKE_DIR}" <<'PY'
import sys
from pathlib import Path
import yaml

data_config = Path(sys.argv[1]).resolve()
max_images = int(sys.argv[2])
smoke_dir = Path(sys.argv[3]).resolve()
smoke_dir.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(data_config.read_text(encoding="utf-8"))
if not isinstance(cfg, dict):
    raise SystemExit("::error::Dataset YAML must be a mapping")

cfg_dir = data_config.parent
root = cfg.get("path")
if root:
    root = Path(root)
    if not root.is_absolute():
        root = (cfg_dir / root).resolve()
else:
    root = cfg_dir

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def resolve_text_list(path_like: str):
    if any(ch in path_like for ch in "*?[]"):
        return sorted(
            f for f in root.glob(path_like) if f.is_file() and f.suffix.lower() in image_exts
        )

    p = Path(path_like)
    if not p.is_absolute():
        p = (root / p).resolve()
    if not p.exists():
        return []
    if p.is_file():
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out = []
        for ln in lines:
            lp = Path(ln)
            if not lp.is_absolute():
                lp = (p.parent / lp).resolve()
            if lp.exists() and lp.suffix.lower() in image_exts:
                out.append(lp)
        return out
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in image_exts)
    return []

train_spec = cfg.get("train")
if train_spec is None:
    raise SystemExit("::error::Dataset YAML missing required 'train' key")

train_images = []
if isinstance(train_spec, str):
    train_images = resolve_text_list(train_spec)
elif isinstance(train_spec, list):
    for entry in train_spec:
        if isinstance(entry, str):
            train_images.extend(resolve_text_list(entry))
else:
    raise SystemExit("::error::Unsupported 'train' value in dataset YAML")

train_images = sorted(dict.fromkeys(train_images))
if not train_images:
    raise SystemExit("::error::No train images resolved from dataset YAML")

selected = train_images[:max_images]
if len(selected) < max_images:
    print(
        f"::warning::Requested {max_images} images but found {len(selected)}; continuing with available images.",
        file=sys.stderr,
    )

train_txt = smoke_dir / "train.txt"
val_txt = smoke_dir / "val.txt"
train_txt.write_text("\n".join(str(p) for p in selected) + "\n", encoding="utf-8")
val_txt.write_text("\n".join(str(p) for p in selected) + "\n", encoding="utf-8")

smoke_cfg = dict(cfg)
smoke_cfg["path"] = str(root)
smoke_cfg["train"] = str(train_txt)
smoke_cfg["val"] = str(val_txt)
smoke_cfg.pop("test", None)

smoke_yaml = smoke_dir / "data-smoke.yaml"
smoke_yaml.write_text(yaml.safe_dump(smoke_cfg, sort_keys=False), encoding="utf-8")
print(str(smoke_yaml))
PY
)

echo "Smoke dataset config: ${SMOKE_YAML}"

YOLO_CMD=(python3 -m ultralytics)
if command -v yolo >/dev/null 2>&1; then
  YOLO_CMD=(yolo)
fi

FULL_RUN_NAME="${RUN_NAME}-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}"
echo "Starting YOLO smoke training..."
"${YOLO_CMD[@]}" "${TASK}" train \
  data="${SMOKE_YAML}" \
  model="${MODEL}" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch=2 \
  device=cpu \
  workers=0 \
  project="${PROJECT_DIR}" \
  name="${FULL_RUN_NAME}"

RUN_DIR="${PROJECT_DIR}/${FULL_RUN_NAME}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "::error::Training did not produce run directory: ${RUN_DIR}"
  exit 1
fi

if [[ ! -f "${RUN_DIR}/weights/last.pt" && ! -f "${RUN_DIR}/weights/best.pt" ]]; then
  echo "::error::Training completed without saving weights in ${RUN_DIR}/weights"
  exit 1
fi

echo "Smoke training succeeded and weights were saved."
