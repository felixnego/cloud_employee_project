"""YOLOX-nano object detection (ONNX, OpenCV Zoo).

Chosen over YOLOv8 because every public YOLOv8 ONNX export sits behind an
auth-gated Hugging Face repo, while `ultralytics` itself drags in PyTorch.
YOLOX-nano is COCO-80, 34 MB, and downloads from a stable URL.

The decode step is hand-written: YOLOX exports a raw (1, n_anchors, 85)
tensor with no NMS baked in, so grid/stride decoding happens here.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from ._onnx import session

_PAD_VALUE = 114  # YOLOX's letterbox fill


class ObjectDetector:
    def __init__(self,
                 score_threshold: float = config.OBJECT_SCORE_THRESHOLD,
                 nms_threshold: float = config.OBJECT_NMS_THRESHOLD):
        self._sess = session(config.YOLOX_ONNX)
        self._input = self._sess.get_inputs()[0].name
        shape = self._sess.get_inputs()[0].shape
        # Static shapes expected: [1, 3, H, W]
        self._h = int(shape[2]) if isinstance(shape[2], int) else 640
        self._w = int(shape[3]) if isinstance(shape[3], int) else 640
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.classes = config.COCO_CLASSES

    # ---------------------------------------------------------- pre/post ----
    def _letterbox(self, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = bgr.shape[:2]
        ratio = min(self._h / h, self._w / w)
        resized = cv2.resize(bgr, (int(round(w * ratio)), int(round(h * ratio))),
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self._h, self._w, 3), _PAD_VALUE, dtype=np.float32)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        # YOLOX takes BGR 0-255 float32 in CHW; no mean/std normalisation.
        return canvas.transpose(2, 0, 1)[None, ...], ratio

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        """(1, n, 85) grid-relative -> (n, 85) in letterbox pixel space."""
        out = raw[0].copy()
        grids, expanded = [], []
        for stride in (8, 16, 32):
            hsize, wsize = self._h // stride, self._w // stride
            yv, xv = np.meshgrid(np.arange(hsize), np.arange(wsize), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            expanded.append(np.full((hsize * wsize, 1), stride))
        grid = np.concatenate(grids, 0)
        stride_arr = np.concatenate(expanded, 0)
        if grid.shape[0] != out.shape[0]:   # unexpected export; bail loudly
            raise ValueError(
                f"YOLOX anchor mismatch: model emitted {out.shape[0]} anchors, "
                f"strides (8,16,32) at {self._h}x{self._w} imply {grid.shape[0]}"
            )
        out[:, :2] = (out[:, :2] + grid) * stride_arr
        out[:, 2:4] = np.exp(out[:, 2:4]) * stride_arr
        return out

    # --------------------------------------------------------------- API ----
    def detect(self, bgr: np.ndarray) -> list[dict]:
        h, w = bgr.shape[:2]
        tensor, ratio = self._letterbox(bgr)
        raw = self._sess.run(None, {self._input: tensor})[0]
        pred = self._decode(np.asarray(raw, dtype=np.float32))

        obj_conf = pred[:, 4]
        cls_scores = pred[:, 5:]
        cls_ids = cls_scores.argmax(1)
        scores = obj_conf * cls_scores[np.arange(len(cls_ids)), cls_ids]

        keep = scores >= self.score_threshold
        if not keep.any():
            return []
        boxes_cxcywh = pred[keep, :4] / ratio      # back to source pixels
        scores, cls_ids = scores[keep], cls_ids[keep]

        # cxcywh -> xywh for cv2 NMS
        xywh = np.empty_like(boxes_cxcywh)
        xywh[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        xywh[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        xywh[:, 2:] = boxes_cxcywh[:, 2:]

        idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(),
                               self.score_threshold, self.nms_threshold)
        if idx is None or len(idx) == 0:
            return []

        results = []
        for i in np.asarray(idx).reshape(-1):
            x, y, bw, bh = xywh[i]
            cid = int(cls_ids[i])
            results.append({
                "class_id": cid,
                "class_name": self.classes[cid] if cid < len(self.classes) else f"id_{cid}",
                "score": float(scores[i]),
                # Normalised so rows are resolution-independent.
                "x0": float(np.clip(x / w, 0, 1)),
                "y0": float(np.clip(y / h, 0, 1)),
                "x1": float(np.clip((x + bw) / w, 0, 1)),
                "y1": float(np.clip((y + bh) / h, 0, 1)),
                "area_frac": float(np.clip(bw * bh / (w * h), 0, 1)),
            })
        return results
