"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .export import to_coco, to_json, to_masks, to_overlay
from .pipeline import ZeroShotAnnotator

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images_in(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="zsa",
        description="Zero shot image annotation with CLIP and SAM2.",
    )
    p.add_argument("input", type=Path,
                   help="image file or directory of images")
    p.add_argument("-c", "--config", type=Path, default=Path("configs/agriculture.yaml"),
                   help="pipeline config YAML")
    p.add_argument("-o", "--output", type=Path, default=Path("runs/latest"),
                   help="output directory")
    p.add_argument("--prompts", nargs="+", default=None,
                   help="override the class vocabulary from the config")
    p.add_argument("--min-confidence", type=float, default=None,
                   help="override the confidence floor")
    p.add_argument("--no-overlay", action="store_true",
                   help="skip writing the visual overlay")
    p.add_argument("--coco", action="store_true",
                   help="also write a single COCO file for the whole run")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = PipelineConfig.load(args.config)
    if args.prompts:
        cfg.prompts = args.prompts
    if args.min_confidence is not None:
        cfg.min_confidence = args.min_confidence

    images = _images_in(args.input)
    if not images:
        print(f"no images found under {args.input}", file=sys.stderr)
        return 1

    out_dir = args.output
    (out_dir / "json").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    if not args.no_overlay:
        (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    annotator = ZeroShotAnnotator(cfg)
    records = []

    for i, path in enumerate(images, start=1):
        image = np.array(Image.open(path).convert("RGB"))
        h, w = image.shape[:2]
        annotations = annotator.annotate_array(image)

        stem = path.stem
        to_json(annotations, path, (w, h), out_dir / "json" / f"{stem}.json")
        to_masks(annotations, (w, h), cfg.prompts, out_dir / "masks" / f"{stem}.png")
        if not args.no_overlay:
            to_overlay(annotations, image, out_dir / "overlays" / f"{stem}.jpg")

        records.append((path, (w, h), annotations))
        print(f"[{i}/{len(images)}] {path.name}: {len(annotations)} annotations")

    if args.coco:
        dest = to_coco(records, cfg.prompts, out_dir / "annotations_coco.json")
        print(f"wrote {dest}")

    total = sum(len(r[2]) for r in records)
    print(f"done: {total} annotations across {len(images)} images -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
