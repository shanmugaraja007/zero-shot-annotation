"""CLIP classification of mask proposals.

SAM2 says where the objects are. CLIP says what they are, using nothing but the
text you hand it, which is what makes the whole thing zero shot: adding a new
class means adding a string, not collecting and labelling a dataset.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

from .config import CLIPConfig

log = logging.getLogger(__name__)


class Classifier:
    """Prompt ensembled CLIP head over cropped mask proposals."""

    def __init__(self, cfg: CLIPConfig, device: str):
        self.cfg = cfg
        self.device = device
        self._model = None
        self._processor = None
        self._text_features: torch.Tensor | None = None
        self._labels: list[str] = []

    def _load(self):
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        log.info("loading CLIP %s on %s", self.cfg.model_id, self.device)
        self._model = CLIPModel.from_pretrained(self.cfg.model_id).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.cfg.model_id)

    @torch.no_grad()
    def set_prompts(self, labels: list[str], templates: list[str]) -> None:
        """Embed the class vocabulary once, ahead of any image.

        Each label is embedded under several templates and the results are
        averaged. Prompt ensembling is worth a few points of accuracy for free
        and costs nothing at inference time, because this runs once.
        """
        self._load()
        self._labels = list(labels)

        per_label = []
        for label in labels:
            prompts = [t.format(label) for t in templates]
            inputs = self._processor(text=prompts, return_tensors="pt",
                                     padding=True).to(self.device)
            feats = self._model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            per_label.append(feats.mean(dim=0))

        text = torch.stack(per_label)
        self._text_features = text / text.norm(dim=-1, keepdim=True)
        log.info("embedded %d classes over %d templates", len(labels), len(templates))

    def _crop(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
        h, w = image.shape[:2]
        x, y, bw, bh = bbox
        pad_x = int(bw * self.cfg.context_pad)
        pad_y = int(bh * self.cfg.context_pad)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + bw + pad_x)
        y1 = min(h, y + bh + pad_y)
        return Image.fromarray(image[y0:y1, x0:x1])

    @torch.no_grad()
    def classify(self, image: np.ndarray,
                 bboxes: list[tuple[int, int, int, int]]) -> list[tuple[str, float]]:
        """Return (label, confidence) for every proposal box."""
        if self._text_features is None:
            raise RuntimeError("call set_prompts() before classify()")
        if not bboxes:
            return []

        self._load()
        crops = [self._crop(image, b) for b in bboxes]

        results: list[tuple[str, float]] = []
        for start in range(0, len(crops), self.cfg.batch_size):
            batch = crops[start:start + self.cfg.batch_size]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            probs = (self.cfg.logit_scale * feats @ self._text_features.T).softmax(dim=-1)
            conf, idx = probs.max(dim=-1)
            for c, i in zip(conf.tolist(), idx.tolist()):
                results.append((self._labels[i], float(c)))

        return results
