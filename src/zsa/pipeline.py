"""End to end zero shot annotation: image in, labelled masks out.

SAM2 proposes class agnostic masks; CLIP assigns the zero shot semantic label.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .classifier import Classifier
from .config import PipelineConfig
from .segmenter import MaskProposal, Segmenter

log = logging.getLogger(__name__)

UNLABELLED = "unlabelled"


@dataclass
class Annotation:
    """One labelled region."""

    label: str
    score: float
    bbox: tuple[int, int, int, int]
    area: int
    mask: np.ndarray

    def to_dict(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "area": self.area,
        }


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


class ZeroShotAnnotator:
    """SAM2 proposes, CLIP disposes."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.segmenter = Segmenter(cfg.sam, self.device)
        self.classifier = Classifier(cfg.clip, self.device)
        self._prompts_ready = False

    def _ensure_prompts(self) -> None:
        if self._prompts_ready:
            return
        if not self.cfg.prompts:
            raise ValueError("no prompts configured; nothing to label against")
        self.classifier.set_prompts(self.cfg.prompts, self.cfg.templates)
        self._prompts_ready = True

    def _suppress(self, annotations: list[Annotation]) -> list[Annotation]:
        """Drop lower scoring masks that substantially overlap a better one.

        SAM2 happily returns a whole tractor and its front wheel as separate
        proposals. Without this you annotate the same pixels twice.
        """
        ordered = sorted(annotations, key=lambda a: a.score, reverse=True)
        kept: list[Annotation] = []
        for cand in ordered:
            if all(_iou(cand.mask, k.mask) < self.cfg.nms_iou for k in kept):
                kept.append(cand)
        dropped = len(annotations) - len(kept)
        if dropped:
            log.info("suppressed %d overlapping masks", dropped)
        return kept

    def annotate_array(self, image: np.ndarray) -> list[Annotation]:
        """Annotate an RGB uint8 array."""
        self._ensure_prompts()

        proposals: list[MaskProposal] = self.segmenter.propose(image)
        if not proposals:
            return []

        scored = self.classifier.classify(image, [p.bbox for p in proposals])

        annotations: list[Annotation] = []
        for prop, (label, conf) in zip(proposals, scored):
            if conf < self.cfg.min_score:
                label = UNLABELLED
            annotations.append(
                Annotation(
                    label=label,
                    score=conf,
                    bbox=prop.bbox,
                    area=prop.area,
                    mask=prop.mask,
                )
            )

        annotations = self._suppress(annotations)
        annotations = [a for a in annotations if a.label != UNLABELLED]

        log.info("kept %d annotations across %d classes",
                 len(annotations), len({a.label for a in annotations}))
        return annotations

    def annotate_image(self, path: str | Path) -> list[Annotation]:
        """Annotate an image file."""
        image = np.array(Image.open(path).convert("RGB"))
        log.info("annotating %s (%dx%d)", path, image.shape[1], image.shape[0])
        return self.annotate_array(image)
