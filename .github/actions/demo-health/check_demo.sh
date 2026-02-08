#!/usr/bin/env bash
set -euo pipefail

APP_FILE="${1:-streamlit_app/app.py}"
REQUIREMENTS_FILE="${2:-streamlit_app/requirements.txt}"
FALLBACK_REQUIREMENTS_FILE="${3:-requirements.txt}"
INFERENCE_FUNCTIONS="${4:-run_inference,predict,infer}"
IMAGE_SIZE="${5:-640}"

if [[ ! -f "${APP_FILE}" ]]; then
  echo "::error::App file not found: ${APP_FILE}"
  exit 1
fi

python3 -m pip install --upgrade pip

if [[ -f "${REQUIREMENTS_FILE}" ]]; then
  python3 -m pip install -r "${REQUIREMENTS_FILE}"
elif [[ -f "${FALLBACK_REQUIREMENTS_FILE}" ]]; then
  python3 -m pip install -r "${FALLBACK_REQUIREMENTS_FILE}"
else
  echo "::warning::No requirements file found. Installing minimal defaults."
  python3 -m pip install streamlit ultralytics pillow
fi

python3 - "${APP_FILE}" "${INFERENCE_FUNCTIONS}" "${IMAGE_SIZE}" <<'PY'
import importlib.util
import sys
from pathlib import Path

from PIL import Image

app_file = Path(sys.argv[1]).resolve()
fn_names = [name.strip() for name in sys.argv[2].split(",") if name.strip()]
img_size = int(sys.argv[3])

dummy_path = Path(".ci/dummy-input.jpg").resolve()
dummy_path.parent.mkdir(parents=True, exist_ok=True)
Image.new("RGB", (img_size, img_size), color=(40, 120, 200)).save(dummy_path)

spec = importlib.util.spec_from_file_location("streamlit_demo_app", app_file)
if spec is None or spec.loader is None:
    raise SystemExit(f"::error::Unable to import app module from {app_file}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def try_call(fn, *candidates):
    last_error = None
    for arg in candidates:
        try:
            fn(arg)
            return True
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return False

for fn_name in fn_names:
    fn = getattr(module, fn_name, None)
    if callable(fn):
        image_obj = Image.open(dummy_path)
        ok = try_call(fn, str(dummy_path), image_obj)
        if ok:
            print(f"Demo health check passed via function '{fn_name}'.")
            raise SystemExit(0)

model = getattr(module, "model", None)
if model is not None and hasattr(model, "predict") and callable(model.predict):
    model.predict(str(dummy_path))
    print("Demo health check passed via module-level model.predict.")
    raise SystemExit(0)

raise SystemExit(
    "::error::Could not execute app inference. Expose one of: "
    f"{', '.join(fn_names)} or a module-level 'model.predict'."
)
PY
