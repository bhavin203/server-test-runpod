FROM runpod/pytorch:3.10-2.1.0-cu121
SHELL ["/bin/bash","-lc"]
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 INSIGHTFACE_HOME=/app/.insightface
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python - <<'PY'\nimport os\nfrom insightface.app import FaceAnalysis\nfrom insightface.model_zoo import get_model\nos.makedirs(os.environ.get("INSIGHTFACE_HOME","/app/.insightface"), exist_ok=True)\nfa=FaceAnalysis(name="buffalo_l"); fa.prepare(ctx_id=-1, det_size=(640,640))\nget_model("inswapper_128.onnx", download=True, download_zip=True)\nprint("Preloaded InsightFace models.")\nPY
COPY handler.py .
CMD ["python", "-u", "rp_handler.py"]
