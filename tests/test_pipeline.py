"""Tests for the parts that do not need model weights.

Covers config loading, the IoU helper, mask suppression and every export
format. Segmenter and Classifier are deliberately not covered here: they are
thin wrappers over SAM2 and CLIP, and testing them means downloading weights.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from zsa import PipelineConfig, ZeroShotAnnotator
from zsa import export
from zsa.pipeline import Annotation, _iou, _resolve_device


def _ann(label, conf, bbox, mask):
    return Annotation(label=label, confidence=conf, bbox=bbox,
                      area=int(mask.sum()), mask=mask)


def test_config_loads_agriculture():
    cfg = PipelineConfig.load("configs/agriculture.yaml")
    assert "crop field" in cfg.prompts
    assert cfg.sam.points_per_side == 32
    assert cfg.clip.model_id.startswith("openai/clip")
    assert 0.0 < cfg.min_confidence < 1.0


def test_config_loads_driving():
    cfg = PipelineConfig.load("configs/driving.yaml")
    assert "pedestrian" in cfg.prompts
    assert cfg.sam.max_area_frac < 1.0


@pytest.mark.parametrize("requested,expected", [("cuda", "cuda"), ("cpu", "cpu")])
def test_resolve_device_passthrough(requested, expected):
    assert _resolve_device(requested) == expected


def test_iou():
    a = np.zeros((10, 10), bool)
    a[:5, :5] = True
    assert _iou(a, a.copy()) == 1.0

    disjoint = np.zeros((10, 10), bool)
    disjoint[5:, 5:] = True
    assert _iou(a, disjoint) == 0.0

    half = np.zeros((10, 10), bool)
    half[:5, :10] = True
    assert _iou(a, half) == pytest.approx(0.5)


def test_suppress_drops_overlapping_lower_score():
    big = np.zeros((64, 80), bool)
    big[10:40, 10:40] = True
    inner = np.zeros((64, 80), bool)
    inner[12:38, 12:38] = True          # heavy overlap with big
    far = np.zeros((64, 80), bool)
    far[45:60, 50:75] = True            # no overlap

    anns = [
        _ann("tree", 0.91, (10, 10, 30, 30), big),
        _ann("bush", 0.55, (12, 12, 26, 26), inner),
        _ann("sky", 0.77, (50, 45, 25, 15), far),
    ]

    cfg = PipelineConfig(prompts=["tree", "bush", "sky"], nms_iou=0.70)
    annotator = ZeroShotAnnotator.__new__(ZeroShotAnnotator)
    annotator.cfg = cfg

    kept = ZeroShotAnnotator._suppress(annotator, anns)
    assert {a.label for a in kept} == {"tree", "sky"}


def test_exports(tmp_path):
    image = (np.random.rand(64, 80, 3) * 255).astype(np.uint8)
    mask = np.zeros((64, 80), bool)
    mask[10:40, 10:40] = True
    anns = [_ann("tree", 0.91, (10, 10, 30, 30), mask)]
    classes = ["tree", "sky"]

    out = export.to_json(anns, "field.jpg", (80, 64), tmp_path / "a.json")
    payload = json.loads(out.read_text())
    assert payload["width"] == 80 and len(payload["annotations"]) == 1
    assert payload["annotations"][0]["label"] == "tree"

    out = export.to_coco([("field.jpg", (80, 64), anns)], classes, tmp_path / "c.json")
    coco = json.loads(out.read_text())
    assert len(coco["categories"]) == 2
    assert coco["annotations"][0]["bbox"] == [10, 10, 30, 30]

    out = export.to_masks(anns, (80, 64), classes, tmp_path / "m.png")
    arr = np.array(Image.open(out))
    assert arr.shape == (64, 80)
    assert arr.max() > 0                # something was painted
    assert arr.min() == 0               # background survives

    out = export.to_overlay(anns, image, tmp_path / "o.jpg")
    assert out.stat().st_size > 500


def test_colour_is_stable_per_label():
    assert export.colour_for("tree") == export.colour_for("tree")
