from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image
from .utils import clamp

@dataclass(frozen=True)
class CropResult:
    bbox_px: Dict[str, int]
    out_path: Path

class Cropper:
    """
    Crops normalized bbox regions from a page PNG.
    """

    def norm_box_to_pixels(self, box: Dict[str, float], w: int, h: int) -> Tuple[int, int, int, int]:
        x0 = int(clamp(float(box["x0"]), 0, 1) * w)
        y0 = int(clamp(float(box["y0"]), 0, 1) * h)
        x1 = int(clamp(float(box["x1"]), 0, 1) * w)
        y1 = int(clamp(float(box["y1"]), 0, 1) * h)

        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

        # Avoid zero-size crops
        if x1 - x0 < 5:
            x1 = min(w, x0 + 5)
        if y1 - y0 < 5:
            y1 = min(h, y0 + 5)

        return x0, y0, x1, y1

    def crop_to_png(self, page_png: Path, bbox_norm: Dict[str, float], out_path: Path) -> CropResult:
        img = Image.open(page_png).convert("RGB")
        w, h = img.size
        x0, y0, x1, y1 = self.norm_box_to_pixels(bbox_norm, w, h)
        crop = img.crop((x0, y0, x1, y1))
        crop.save(out_path)
        return CropResult(
            bbox_px={"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            out_path=out_path
        )
