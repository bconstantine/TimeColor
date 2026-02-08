#!/usr/bin/env bash
set -euo pipefail

# Repo root (works regardless of where script is run from)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

# Locate diffusers package root in the CURRENT active env
DIFFUSERS_ROOT="$($PY - <<'PY'
import diffusers, pathlib
print(pathlib.Path(diffusers.__file__).resolve().parent)
PY
)"

echo "[TimeColor] Using python: $($PY -c 'import sys; print(sys.executable)')"
echo "[TimeColor] diffusers root: ${DIFFUSERS_ROOT}"
echo "[TimeColor] diffusers version: $($PY -c 'import diffusers; print(diffusers.__version__)')"

SRC1="${ROOT}/diffusers_attention_modified/attention_processor.py"
SRC2="${ROOT}/diffusers_attention_modified/cogvideox_transformer_3d.py"

DST1="${DIFFUSERS_ROOT}/models/attention_processor.py"
DST2="${DIFFUSERS_ROOT}/models/transformers/cogvideox_transformer_3d.py"

# Sanity checks
[[ -f "$SRC1" ]] || { echo "Missing $SRC1"; exit 1; }
[[ -f "$SRC2" ]] || { echo "Missing $SRC2"; exit 1; }
[[ -f "$DST1" ]] || { echo "Missing target $DST1"; exit 1; }
[[ -f "$DST2" ]] || { echo "Missing target $DST2"; exit 1; }

# Backup originals
STAMP="$($PY - <<'PY'
from datetime import datetime
print(datetime.now().strftime("%Y%m%d_%H%M%S"))
PY
)"
BACKUP_DIR="${ROOT}/.diffusers_backup/${STAMP}/models/transformers"
mkdir -p "$BACKUP_DIR"

cp -v "$DST1" "${ROOT}/.diffusers_backup/${STAMP}/models/attention_processor.py"
cp -v "$DST2" "$BACKUP_DIR/cogvideox_transformer_3d.py"

# Patch (overwrite)
install -m 644 "$SRC1" "$DST1"
install -m 644 "$SRC2" "$DST2"

# Verify imports point to the patched locations
$PY - <<'PY'
from diffusers.models import attention_processor
from diffusers.models.transformers import cogvideox_transformer_3d
print("[TimeColor] Patched OK:")
print(" -", attention_processor.__file__)
print(" -", cogvideox_transformer_3d.__file__)
PY

echo "[TimeColor] Done. Backup saved under: ${ROOT}/.diffusers_backup/${STAMP}"
