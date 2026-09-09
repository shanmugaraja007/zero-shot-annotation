"""Configuration objects for the zero shot annotation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SAMConfig:
    """Settings for the SAM2 automatic mask generator."""

    model_id: str = "facebook/sam2.1-hiera-small"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.80
    stability_score_thresh: float = 0.90
    min_mask_region_area: int = 900
    # Masks covering more than this fraction of the frame are dropped. They are
    # almost always the sky or the road surface swallowing the whole scene.
    max_area_frac: float = 0.55
    # Masks smaller than this fraction are dropped as texture noise.
    min_area_frac: float = 0.0015


@dataclass
class CLIPConfig:
    """Settings for the CLIP classification head."""

    model_id: str = "openai/clip-vit-base-patch32"
    # Crops are expanded by this factor before classification. A tight crop of a
    # traffic sign is ambiguous; a little context makes it obvious.
    context_pad: float = 0.15
    batch_size: int = 32
    # Softmax temperature. CLIP ships 100.0 as its logit scale.
    logit_scale: float = 100.0


@dataclass
class PipelineConfig:
    """Top level configuration."""

    prompts: list[str] = field(default_factory=list)
    templates: list[str] = field(
        default_factory=lambda: [
            "a photo of a {}",
            "a close up photo of a {}",
            "a photo of a {} in a field",
            "a cropped photo of a {}",
        ]
    )
    # Below this score a mask is written out as "unlabelled" rather than being
    # forced into the closest class. Silence beats a confident wrong label.
    min_confidence: float = 0.28
    # Two masks overlapping by more than this keep only the higher scoring one.
    nms_iou: float = 0.70
    device: str = "auto"
    sam: SAMConfig = field(default_factory=SAMConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        sam = SAMConfig(**raw.pop("sam", {}))
        clip = CLIPConfig(**raw.pop("clip", {}))
        return cls(sam=sam, clip=clip, **raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
