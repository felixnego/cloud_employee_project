"""Single place for every path, sampling rate and model constant.

Changing the grain of the whole dataset is a one-line edit here, which is the
point: sampling decisions are configuration, not something buried in extractors.
"""
from __future__ import annotations

import os
from pathlib import Path


def _path(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default)).resolve()


# ---------------------------------------------------------------- locations --
VIDEO_DIR = _path("VIDEO_DIR", "video_data")
WAREHOUSE_DIR = _path("WAREHOUSE_DIR", "warehouse")
CACHE_DIR = _path("CACHE_DIR", "cache")
MODEL_DIR = _path("MODEL_DIR", "/opt/models")
TRANSFORM_DIR = _path("TRANSFORM_DIR", "transforms")

RAW_DIR = WAREHOUSE_DIR / "raw"
MART_DIR = WAREHOUSE_DIR / "marts"
DUCKDB_PATH = WAREHOUSE_DIR / "warehouse.duckdb"

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}

# ----------------------------------------------------------------- sampling --
# 4 Hz is the native grain of the dataset. These are short, fast-cut social
# ads: at 1 Hz a 400 ms product reveal disappears entirely. Seconds are then
# published as a friendly rollup -- you can always aggregate up, never down.
FRAME_HZ = 4.0
FRAME_HEIGHT = 720          # frames are downscaled for speed; keeps OCR legible
AUDIO_SR = 16_000           # what Whisper wants anyway

# Per-model sampling. Cheap models see every frame; expensive ones subsample.
VISUAL_HZ = FRAME_HZ        # OpenCV: trivially cheap
FACE_HZ = FRAME_HZ          # YuNet + FER+: small ONNX models
OBJECT_HZ = 2.0             # YOLOX-nano: ~80ms/frame on CPU
OCR_HZ = 1.0                # RapidOCR: slowest per frame; on-screen text is slow-moving
CLIP_FRAMES_PER_SHOT = 3    # CLIP runs per shot, not per frame

# ------------------------------------------------------------------- models --
WHISPER_MODEL = "small"     # 'base' is ~3x faster but noticeably worse on
                            # voiceover mixed under a music bed
WHISPER_COMPUTE = "int8"
SCENE_THRESHOLD = 27.0      # PySceneDetect ContentDetector default; tuned for cuts
FACE_SCORE_THRESHOLD = 0.70
OBJECT_SCORE_THRESHOLD = 0.40
OBJECT_NMS_THRESHOLD = 0.50
OCR_SCORE_THRESHOLD = 0.50

YUNET_ONNX = MODEL_DIR / "face_detection_yunet.onnx"
FERPLUS_ONNX = MODEL_DIR / "emotion_ferplus.onnx"
YOLOX_ONNX = MODEL_DIR / "object_detection_yolox.onnx"
CLIP_ONNX = MODEL_DIR / "clip" / "model_quantized.onnx"
CLIP_TOKENIZER = MODEL_DIR / "clip" / "tokenizer.json"
WHISPER_CACHE = MODEL_DIR / "whisper"

# FER+ emits these 8 classes, in this order.
EMOTION_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]

# ------------------------------------------------- zero-shot label taxonomy --
# CLIP is run as several independent, mutually-exclusive questions rather than
# one 30-way softmax. Softmaxing within an axis gives calibrated, comparable
# scores per axis; one flat softmax over mixed concepts does not.
# `label` is what an analyst sees; `prompt` is what CLIP scores.
#
# Two axes were removed after testing them against real frames (see README,
# "What we tested and rejected"):
#
#   * A "title card" option on any axis absorbs almost every frame regardless
#     of wording -- CLIP ViT-B/32 has a very strong prior toward text-card
#     prompts. "Is this a title card?" is answered far better by the OCR
#     feature (`max_text_height_frac`, `n_text_detections`).
#   * A "mood" axis never identified an upbeat beauty ad as upbeat under any
#     wording tried. Single-frame affect is beyond this model, and a
#     confidently-wrong mood label is worse than no mood label.
CLIP_AXES: dict[str, list[tuple[str, str]]] = {
    "shot_scale": [
        ("extreme_close_up", "an extreme close-up shot of a detail"),
        ("close_up",         "a close-up shot of a person's face"),
        ("medium",           "a medium shot showing a person's upper body"),
        ("wide",             "a wide establishing shot of a place"),
    ],
    # Concrete "a photo taken ..." wording beats abstract scene descriptions.
    # No title-card option here on purpose -- see the note above.
    "setting": [
        ("indoor",  "a photo taken inside a building or room"),
        ("outdoor", "a photo taken outside in a street, field or stadium"),
        ("studio",  "a photo of a person in a photography studio with a plain backdrop"),
    ],
    "subject": [
        ("product",       "a close-up of a consumer product being shown"),
        ("single_person", "a single person speaking directly to the camera"),
        ("two_people",    "two people interacting with each other"),
        ("group",         "a large group or crowd of people"),
        ("hands",         "a pair of hands demonstrating or holding something"),
        ("place",         "a place or landscape with no people in it"),
    ],
    "style": [
        ("ugc_selfie",   "a vertical selfie-style phone video filmed at arm's length"),
        ("cinematic",    "a professionally filmed cinematic advertisement shot"),
        ("documentary",  "a candid documentary-style observational shot"),
        ("animation",    "an animated or motion-graphics sequence"),
    ],
}

# COCO-80, in the order YOLOX was trained on. Static list rather than a
# download, so the image has one fewer network dependency.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
