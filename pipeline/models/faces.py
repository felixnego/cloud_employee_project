"""Face detection (YuNet) + facial expression (FER+), both ONNX.

This pair replaces MediaPipe/DeepFace/Py-Feat: it needs no PyTorch, has
arm64 wheels, and -- unlike MediaPipe blendshapes -- FER+ returns the eight
*named* emotion categories an analyst can actually reason about.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from ._onnx import session, softmax


class FaceDetector:
    """YuNet. Returns (n, 15): bbox(4) + 5 landmarks(10) + score(1)."""

    def __init__(self, score_threshold: float = config.FACE_SCORE_THRESHOLD):
        self._det = cv2.FaceDetectorYN.create(
            model=str(config.YUNET_ONNX), config="", input_size=(320, 320),
            score_threshold=score_threshold, nms_threshold=0.3, top_k=500,
        )

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(bgr)
        return np.empty((0, 15), dtype=np.float32) if faces is None else faces


class ExpressionClassifier:
    """FER+ on a 64x64 grayscale face crop -> 8 emotion probabilities."""

    def __init__(self):
        self._sess = session(config.FERPLUS_ONNX)
        self._input = self._sess.get_inputs()[0].name
        self._output = self._sess.get_outputs()[0].name

    def predict(self, bgr_face: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        tensor = gray.astype(np.float32).reshape(1, 1, 64, 64)
        logits = self._sess.run([self._output], {self._input: tensor})[0]
        return softmax(logits.reshape(-1))

    @staticmethod
    def labels() -> list[str]:
        return list(config.EMOTION_LABELS)
