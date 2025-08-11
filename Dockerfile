# CUDA + PyTorch base that already has GPU libs
FROM runpod/pytorch:3.10-2.1.0-cu121

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/app/.insightface

# OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Keep pip fresh so it can pick prebuilt wheels
RUN python -m pip install --upgrade pip

# 2) Install core numeric deps FIRST (helps resolver choose wheels)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) (Optional but recommended) Preload InsightFace models at build time
RUN python - <<'PY'\n\
import os\n\
from insightface.app import FaceAnalysis\n\
from insightface.model_zoo import get_model\n\
os.makedirs(os.environ.get("INSIGHTFACE_HOME","/app/.insightface"), exist_ok=True)\n\
fa = FaceAnalysis(name="buffalo_l")\n\
# CPU prepare to trigger model downloads into INSIGHTFACE_HOME\n\
fa.prepare(ctx_id=-1, det_size=(640,640))\n\
get_model("inswapper_128.onnx", download=True, download_zip=True)\n\
print("Preloaded InsightFace models.")\n\
PY

# 4) Your handler
COPY rp_handler.py .

CMD ["python", "-u", "rp_handler.py"]
