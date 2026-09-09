"""Writing annotations out in formats other tools actually read."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Stable palette so the same class keeps the same colour across a whole run.
PALETTE = [
    (125, 211, 160), (240, 200, 106), (107, 168, 224), (224, 115, 107),
    (186, 148, 226), (120, 214, 214), (232, 158, 96), (156, 200, 108),
]


def colour_for(label: str) -> tuple[int, int, int]:
    return PALETTE[hash(label) % len(PALETTE)]


def to_json(annotations, image_path: str | Path, size: tuple[int, int],
            out_path: str | Path) -> Path:
    """Write a plain JSON record of the annotations."""
    payload = {
        "image": str(Path(image_path).name),
        "width": size[0],
        "height": size[1],
        "annotations": [a.to_dict() for a in annotations],
    }
    out = Path(out_path)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def to_coco(records, categories: list[str], out_path: str | Path) -> Path:
    """Write a COCO detection file across a whole run.

    `records` is an iterable of (image_path, (width, height), annotations).
    COCO because every annotation tool and training script already reads it.
    """
    cat_index = {name: i + 1 for i, name in enumerate(sorted(categories))}
    images, anns = [], []
    ann_id = 1

    for img_id, (path, (w, h), annotations) in enumerate(records, start=1):
        images.append({
            "id": img_id,
            "file_name": Path(path).name,
            "width": w,
            "height": h,
        })
        for a in annotations:
            x, y, bw, bh = a.bbox
            anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_index[a.label],
                "bbox": [x, y, bw, bh],
                "area": int(a.area),
                "iscrowd": 0,
                "score": round(float(a.score), 4),
            })
            ann_id += 1

    payload = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": i, "name": n} for n, i in cat_index.items()],
    }
    out = Path(out_path)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def to_masks(annotations, size: tuple[int, int], classes: list[str],
             out_path: str | Path) -> Path:
    """Write a single channel index mask, 0 = background."""
    w, h = size
    index = {name: i + 1 for i, name in enumerate(sorted(classes))}
    canvas = np.zeros((h, w), dtype=np.uint8)
    # Smallest last so a small object drawn inside a large one stays visible.
    for a in sorted(annotations, key=lambda a: a.area, reverse=True):
        canvas[a.mask] = index[a.label]
    out = Path(out_path)
    Image.fromarray(canvas).save(out)
    return out


def to_overlay(annotations, image: np.ndarray, out_path: str | Path,
               alpha: float = 0.45) -> Path:
    """Render a human checkable overlay: tinted masks plus labels."""
    base = Image.fromarray(image).convert("RGBA")
    tint = Image.new("RGBA", base.size, (0, 0, 0, 0))

    for a in annotations:
        rgb = colour_for(a.label)
        layer = np.zeros((*a.mask.shape, 4), dtype=np.uint8)
        layer[a.mask] = (*rgb, int(alpha * 255))
        tint = Image.alpha_composite(tint, Image.fromarray(layer))

    out_img = Image.alpha_composite(base, tint)
    draw = ImageDraw.Draw(out_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for a in annotations:
        x, y, bw, bh = a.bbox
        rgb = colour_for(a.label)
        draw.rectangle([x, y, x + bw, y + bh], outline=(*rgb, 255), width=2)
        caption = f"{a.label} {a.score:.2f}"
        tw = draw.textlength(caption, font=font)
        ty = max(0, y - 18)
        draw.rectangle([x, ty, x + tw + 8, ty + 18], fill=(*rgb, 235))
        draw.text((x + 4, ty + 2), caption, fill=(10, 10, 10, 255), font=font)

    out = Path(out_path)
    out_img.convert("RGB").save(out, quality=92)
    return out
