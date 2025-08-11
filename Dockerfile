FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/app/.insightface

# OpenCV + basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep pip toolchain fresh so wheels are preferred
RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Your serverless handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
