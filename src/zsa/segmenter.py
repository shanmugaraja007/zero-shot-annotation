"""Class agnostic mask proposals from SAM2.

SAM2 does not classify anything. It returns every region it thinks is an
object, with no idea what any of them are; the semantic label comes from CLIP
later. This module wraps the automatic mask generator and applies the mask area
thresholds that keep the proposal count workable before CLIP has to score them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import SAMConfig

log = logging.getLogger(__name__)


@dataclass
class MaskProposal:
    """One class agnostic region proposal."""

    mask: np.ndarray          # bool, (H, W)
    bbox: tuple[int, int, int, int]   # x, y, w, h
    area: int
    predicted_iou: float
    stability_score: float


class Segmenter:
    """Thin wrapper over SAM2AutomaticMaskGenerator."""

    def __init__(self, cfg: SAMConfig, device: str):
        self.cfg = cfg
        self.device = device
        self._generator = None

    def _load(self):
        if self._generator is not None:
            return self._generator

        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2_hf

        log.info("loading SAM2 %s on %s", self.cfg.model_id, self.device)
        model = build_sam2_hf(self.cfg.model_id, device=self.device)
        self._generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=self.cfg.points_per_side,
            pred_iou_thresh=self.cfg.pred_iou_thresh,
            stability_score_thresh=self.cfg.stability_score_thresh,
            min_mask_region_area=self.cfg.min_mask_region_area,
        )
        return self._generator

    def propose(self, image: np.ndarray) -> list[MaskProposal]:
        """Return filtered region proposals for an RGB uint8 image."""
        generator = self._load()
        raw = generator.generate(image)

        # Mask area thresholds, as a fraction of the frame.
        h, w = image.shape[:2]
        frame_area = float(h * w)
        lo = self.cfg.min_area_frac * frame_area
        hi = self.cfg.max_area_frac * frame_area

        proposals: list[MaskProposal] = []
        for record in raw:
            area = int(record["area"])
            if area < lo or area > hi:
                continue
            x, y, bw, bh = (int(v) for v in record["bbox"])
            proposals.append(
                MaskProposal(
                    mask=record["segmentation"].astype(bool),
                    bbox=(x, y, bw, bh),
                    area=area,
                    predicted_iou=float(record.get("predicted_iou", 0.0)),
                    stability_score=float(record.get("stability_score", 0.0)),
                )
            )

        proposals.sort(key=lambda p: p.area, reverse=True)
        log.info("SAM2 proposed %d regions, %d survived filtering",
                 len(raw), len(proposals))
        return proposals
