"""Zero shot image annotation with CLIP and SAM2."""

from .config import CLIPConfig, PipelineConfig, SAMConfig
from .pipeline import Annotation, ZeroShotAnnotator

__version__ = "0.1.0"
__all__ = [
    "Annotation",
    "CLIPConfig",
    "PipelineConfig",
    "SAMConfig",
    "ZeroShotAnnotator",
]
