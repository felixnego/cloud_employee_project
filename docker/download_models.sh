#!/usr/bin/env bash
# Bake every model into the image at build time so that extraction runs are
# fully offline, reproducible, and don't depend on a third party staying up.
# All models are ONNX or CTranslate2 — there is no PyTorch in this image.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/opt/models}"
mkdir -p "$MODEL_DIR/clip"

get() {  # get <url> <dest>
  echo "  -> $(basename "$2")"
  curl -fsSL --retry 3 --retry-delay 2 -o "$2" "$1"
}

echo "[models] face detection (YuNet, OpenCV Zoo)"
get "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    "$MODEL_DIR/face_detection_yunet.onnx"

echo "[models] facial expression (FER+, ONNX Model Zoo)"
get "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx" \
    "$MODEL_DIR/emotion_ferplus.onnx"

echo "[models] object detection (YOLOX-nano, OpenCV Zoo)"
get "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx" \
    "$MODEL_DIR/object_detection_yolox.onnx"

echo "[models] zero-shot shot labelling (CLIP ViT-B/32, int8)"
get "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/model_quantized.onnx" \
    "$MODEL_DIR/clip/model_quantized.onnx"
get "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json" \
    "$MODEL_DIR/clip/tokenizer.json"

echo "[models] speech-to-text (faster-whisper small, int8)"
python - <<'PY'
import os
from faster_whisper import WhisperModel
WhisperModel("small", device="cpu", compute_type="int8",
             download_root=os.path.join(os.environ["MODEL_DIR"], "whisper"))
print("  -> faster-whisper small cached")
PY

echo "[models] done:"
du -sh "$MODEL_DIR"
