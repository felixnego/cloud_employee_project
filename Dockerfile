# Single image for the whole pipeline: ffmpeg + CPU-only ONNX models.
# Deliberately no PyTorch — every model here is ONNX Runtime or CTranslate2.
# Measured: 2.84GB total, of which 681MB is baked-in model weights. The
# PyTorch equivalent (ultralytics + Py-Feat + EasyOCR) adds ~2.5GB on top,
# and several of those pieces have no native arm64 wheels.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/opt/models \
    VIDEO_DIR=/data/videos \
    WAREHOUSE_DIR=/warehouse \
    CACHE_DIR=/cache \
    OMP_NUM_THREADS=4 \
    MPLCONFIGDIR=/tmp/mpl

# ffmpeg/ffprobe do all demuxing and decoding; libglib is an OpenCV runtime dep.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Models are a separate layer so iterating on code never re-downloads them.
COPY docker/download_models.sh /tmp/download_models.sh
RUN bash /tmp/download_models.sh

WORKDIR /app
COPY pipeline/ /app/pipeline/
COPY transforms/ /app/transforms/

ENTRYPOINT ["python", "-m", "pipeline.run"]
CMD ["all"]
