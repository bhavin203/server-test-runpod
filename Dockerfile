FROM runpod/pytorch:3.10-2.1.0-cu121

# Use bash so heredoc works
SHELL ["/bin/bash","-lc"]

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/app/.insightface

# Minimal libs OpenCV needs + git (pip sometimes wants it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep pip fresh so wheels are used (no source builds)
RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- PRELOAD INSIGHTFACE WEIGHTS (IMPORTANT: keep the heredoc as real new lines) ---
RUN python - <<'PY'
import os
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
os.makedirs(os.environ.get("INSIGHTFACE_HOME","/app/.insightface"), exist_ok=True)
fa = FaceAnalysis(name="buffalo_l")
fa.prepare(ctx_id=-1, det_size=(640,640))   # CPU, just to trigger downloads
get_model("inswapper_128.onnx", download=True, download_zip=True)
print("Preloaded InsightFace models.")
PY

# If your handler file is named rp_handler.py, keep this; otherwise change to handler.py
COPY rp_handler.py .
CMD ["python", "-u", "rp_handler.py"]
