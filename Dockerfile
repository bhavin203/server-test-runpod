FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/app/.insightface

# System libs for OpenCV + safe build deps in case a wheel is missing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep pip toolchain fresh so manylinux wheels are preferred
RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Preload InsightFace weights so cold starts are fast
SHELL ["/bin/bash","-lc"]
RUN python - <<'PY'
import os
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
os.makedirs(os.environ.get("INSIGHTFACE_HOME","/app/.insightface"), exist_ok=True)
fa = FaceAnalysis(name="buffalo_l")
fa.prepare(ctx_id=-1, det_size=(640,640))   # CPU download only
get_model("inswapper_128.onnx", download=True, download_zip=True)
print("Preloaded InsightFace models.")
PY

# Your handler filename — keep this consistent with your repo
COPY rp_handler.py .

CMD ["python", "-u", "rp_handler.py"]
