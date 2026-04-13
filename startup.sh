#!/bin/bash
# startup.sh — Checks Network Volume for cached model, downloads if missing,
# then starts the Python handler which launches llama-server.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/runpod-volume/models}"
MODEL_FILE="${MODEL_FILE:-gemma-4-31B-it-abliterated-Q4_K_M.gguf}"
MODEL_REPO="${MODEL_REPO:-LiconStudio/Gemma-4-31B-it-abliterated-GGUF}"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"

echo "=========================================="
echo " gemma4-runpod startup"
echo "=========================================="
echo "MODEL_DIR  : ${MODEL_DIR}"
echo "MODEL_FILE : ${MODEL_FILE}"
echo "MODEL_REPO : ${MODEL_REPO}"
echo "MODEL_PATH : ${MODEL_PATH}"
echo "=========================================="

# ------------------------------------------------------------------
# 1. Ensure model directory exists (Network Volume may not pre-exist)
# ------------------------------------------------------------------
if [ ! -d "${MODEL_DIR}" ]; then
    echo "[startup] Creating model directory: ${MODEL_DIR}"
    mkdir -p "${MODEL_DIR}"
fi

# ------------------------------------------------------------------
# 2. Check if model is already cached on the Network Volume
# ------------------------------------------------------------------
if [ -f "${MODEL_PATH}" ]; then
    MODEL_SIZE=$(stat -c%s "${MODEL_PATH}" 2>/dev/null || echo "0")
    echo "[startup] Model found in cache (${MODEL_SIZE} bytes): ${MODEL_PATH}"
else
    echo "[startup] Model NOT found in cache. Downloading from HuggingFace..."
    echo "[startup]   Repo : ${MODEL_REPO}"
    echo "[startup]   File : ${MODEL_FILE}"

    python3 - <<'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download

repo  = os.environ["MODEL_REPO"]
fname = os.environ["MODEL_FILE"]
dest  = os.environ["MODEL_DIR"]

try:
    path = hf_hub_download(
        repo_id=repo,
        filename=fname,
        local_dir=dest,
        local_dir_use_symlinks=False,
    )
    print(f"[startup] Download complete: {path}")
except Exception as e:
    print(f"[startup] ERROR during download: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

    echo "[startup] Model download finished."
fi

# ------------------------------------------------------------------
# 3. Hand off to the Python handler (which starts llama-server)
# ------------------------------------------------------------------
echo "[startup] Starting Python handler..."
exec python3 /app/handler.py
