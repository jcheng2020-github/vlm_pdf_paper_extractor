from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

@dataclass
class SectionRecord:
    name: str
    file: str

@dataclass
class PageItemRecord:
    type: str              # "table" | "figure"
    caption: str
    confidence: float
    bbox_norm: dict
    bbox_px: dict
    image: str

@dataclass
class PageRecord:
    page: int
    page_png: str
    items: list[PageItemRecord]
    error: str | None = None

@dataclass
class PdfManifest:
    pdf: str
    output_dir: str

    # NEW: document-level metadata (from VLM text extraction)
    title: str | None
    authors: list[str] | None

    # Section text outputs
    sections: list[SectionRecord]

    # Page object outputs
    pages: list[PageRecord]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        import json
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
