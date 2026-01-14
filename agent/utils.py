from __future__ import annotations
import base64
import re
from pathlib import Path

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def slugify(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-\.()]+", "", s)
    s = s.replace(" ", "_")
    s = s[:max_len] if len(s) > max_len else s
    return s or "section"

def image_to_data_url(png_path: Path) -> str:
    data = png_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
