"""CLIP ViT-B/32 (int8 ONNX) for zero-shot shot labelling.

Runs the full CLIPModel graph, which emits `image_embeds` and `text_embeds`
directly, so the label vocabulary is encoded once and cached; later calls only
pay for the vision tower. Falls back to `logits_per_image` if an export
doesn't expose the embedding outputs.

Tokenisation uses the `tokenizers` Rust package reading the repo's
tokenizer.json -- no transformers, no PyTorch.
"""
from __future__ import annotations

import cv2
import numpy as np
from tokenizers import Tokenizer

from .. import config
from ._onnx import session

_CONTEXT_LEN = 77
_LOGIT_SCALE = 100.0                      # CLIP's trained temperature
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_SIDE = 224


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> CLIP's (3, 224, 224): shortest-side resize + centre crop."""
    h, w = bgr.shape[:2]
    scale = _SIDE / min(h, w)
    resized = cv2.resize(bgr, (max(_SIDE, int(round(w * scale))),
                               max(_SIDE, int(round(h * scale)))),
                         interpolation=cv2.INTER_CUBIC)
    rh, rw = resized.shape[:2]
    top, left = (rh - _SIDE) // 2, (rw - _SIDE) // 2
    crop = resized[top:top + _SIDE, left:left + _SIDE]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class ZeroShotLabeller:
    """Scores a frame against the fixed label vocabulary in config.CLIP_AXES."""

    def __init__(self, axes: dict[str, list[tuple[str, str]]] | None = None):
        self.axes = axes or config.CLIP_AXES
        # Flat prompt list plus the slice of it belonging to each axis.
        self.axis_slices: dict[str, slice] = {}
        self.labels: list[str] = []
        prompts: list[str] = []
        for axis, pairs in self.axes.items():
            start = len(prompts)
            for label, prompt in pairs:
                self.labels.append(label)
                prompts.append(prompt)
            self.axis_slices[axis] = slice(start, len(prompts))
        self.prompts = prompts

        self._sess = session(config.CLIP_ONNX)
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self._outputs = [o.name for o in self._sess.get_outputs()]

        tok = Tokenizer.from_file(str(config.CLIP_TOKENIZER))
        tok.enable_truncation(max_length=_CONTEXT_LEN)
        tok.enable_padding(length=_CONTEXT_LEN,
                           pad_id=tok.token_to_id("<|endoftext|>") or 0,
                           pad_token="<|endoftext|>")
        self._tok = tok
        self._text_embeds: np.ndarray | None = None

    # ------------------------------------------------------------- feeds ----
    def _tokenize(self, texts: list[str]) -> dict[str, np.ndarray]:
        enc = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed: dict[str, np.ndarray] = {}
        if "input_ids" in self._inputs:
            feed["input_ids"] = ids
        if "attention_mask" in self._inputs:
            feed["attention_mask"] = mask
        return feed

    def _run(self, texts: list[str], pixel_values: np.ndarray) -> dict[str, np.ndarray]:
        feed = self._tokenize(texts)
        feed["pixel_values"] = pixel_values.astype(np.float32)
        raw = self._sess.run(None, feed)
        return dict(zip(self._outputs, raw))

    # --------------------------------------------------------------- API ----
    def similarity(self, frames: list[np.ndarray]) -> np.ndarray:
        """-> (n_frames, n_prompts) scaled cosine similarity."""
        if not frames:
            return np.empty((0, len(self.prompts)), dtype=np.float32)
        pixels = np.stack([preprocess(f) for f in frames])

        if "image_embeds" in self._outputs and "text_embeds" in self._outputs:
            if self._text_embeds is None:
                # First call encodes the whole vocabulary and caches it.
                out = self._run(self.prompts, pixels[:1])
                self._text_embeds = _l2(np.asarray(out["text_embeds"]))
                img = _l2(np.asarray(out["image_embeds"]))
                if len(frames) > 1:
                    rest = self._run(self.prompts[:1], pixels[1:])
                    img = np.concatenate([img, _l2(np.asarray(rest["image_embeds"]))])
            else:
                # Vocabulary already cached: one cheap prompt keeps the graph happy.
                out = self._run(self.prompts[:1], pixels)
                img = _l2(np.asarray(out["image_embeds"]))
            return _LOGIT_SCALE * img @ self._text_embeds.T

        # Fallback: no embedding outputs, so re-encode the vocabulary per batch.
        key = next(k for k in self._outputs if "logits_per_image" in k)
        return np.asarray(self._run(self.prompts, pixels)[key], dtype=np.float32)

    def score_axes(self, frames: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Per-axis probabilities, averaged over the supplied frames.

        Softmax is applied *within* each axis so scores are calibrated against
        genuine alternatives ("indoor vs outdoor vs studio"), not against
        unrelated concepts from another axis.
        """
        sim = self.similarity(frames)
        result: dict[str, np.ndarray] = {}
        for axis, sl in self.axis_slices.items():
            block = sim[:, sl]
            block = block - block.max(axis=1, keepdims=True)
            probs = np.exp(block)
            probs /= probs.sum(axis=1, keepdims=True)
            result[axis] = probs.mean(axis=0)
        return result
