from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .pdf_reader import PdfReader

# pdfminer.six
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine, LTChar, LTAnno

@dataclass
class LineItem:
    text: str
    page_index: int
    y_top: float      # for sorting within page (optional)
    avg_font_size: float

_heading_re = re.compile(
    r"^(\d+(\.\d+)*)\s+.+|^(abstract|introduction|methods?|materials and methods|results?|discussion|conclusion|references)\b",
    re.IGNORECASE
)

class SectionExtractor:
    """
    Section extraction using pdfminer.six.

    Strategy:
    1) Extract text lines per page with font-size estimates.
    2) Estimate "body font size" as the most frequent size.
    3) Treat lines as headings if:
       - matches common section names / numbering, OR
       - font size is significantly larger than body size, OR
       - short ALL-CAPS line
    4) Accumulate text lines into sections.
    """

    def extract(self, reader: PdfReader) -> Dict[str, str]:
        pdf_path = reader.pdf_path
        lines = self._extract_lines_pdfminer(pdf_path)

        if not lines:
            # If scanned / no text layer
            return {"FrontMatter": ""}

        body_size = self._guess_body_font_size([ln.avg_font_size for ln in lines])

        sections: Dict[str, List[str]] = {}
        current = "FrontMatter"
        sections[current] = []

        for ln in lines:
            t = ln.text.strip()
            if not t:
                continue

            if self._is_heading_candidate(t, ln.avg_font_size, body_size):
                current = t
                sections.setdefault(current, [])
                continue

            sections[current].append(t)

        # Join + light cleanup
        out: Dict[str, str] = {}
        for name, parts in sections.items():
            text = "\n".join(parts)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            out[name] = text
        return out

    # -------------------------
    # pdfminer extraction
    # -------------------------

    def _extract_lines_pdfminer(self, pdf_path: Path) -> List[LineItem]:
        """
        Extract line-like text with estimated average font size.

        We iterate layout elements (LTTextContainer -> LTTextLine).
        For each LTTextLine, we compute average font size over LTChar.
        """
        results: List[LineItem] = []

        for page_index, layout in enumerate(extract_pages(str(pdf_path))):
            page_lines: List[LineItem] = []

            for element in layout:
                if not isinstance(element, LTTextContainer):
                    continue

                for obj in element:
                    if not isinstance(obj, LTTextLine):
                        continue

                    text = obj.get_text().strip()
                    if not text:
                        continue

                    avg_size = self._avg_font_size(obj)
                    # y1 is top in pdfminer coordinate system
                    y_top = float(getattr(obj, "y1", 0.0))

                    page_lines.append(LineItem(
                        text=text,
                        page_index=page_index,
                        y_top=y_top,
                        avg_font_size=avg_size
                    ))

            # Sort lines top-to-bottom on each page for stability
            page_lines.sort(key=lambda x: (-x.y_top, x.text))
            results.extend(page_lines)

        return results

    def _avg_font_size(self, text_line: LTTextLine) -> float:
        sizes: List[float] = []
        for ch in text_line:
            if isinstance(ch, LTChar):
                sizes.append(float(ch.size))
            elif isinstance(ch, LTAnno):
                # whitespace/newlines, ignore
                pass
        if not sizes:
            # Fallback if font sizes unavailable
            return 10.0
        return sum(sizes) / len(sizes)

    # -------------------------
    # Heading heuristics
    # -------------------------

    def _guess_body_font_size(self, sizes: List[float]) -> float:
        if not sizes:
            return 10.0

        buckets: dict[int, int] = {}
        for sz in sizes:
            # bucket to nearest 0.5 pt
            b = int(round(sz * 2))
            buckets[b] = buckets.get(b, 0) + 1

        best_bucket = max(buckets.items(), key=lambda kv: kv[1])[0]
        return best_bucket / 2.0

    def _is_heading_candidate(self, text: str, size: float, body_size: float) -> bool:
        t = text.strip()

        if len(t) < 3 or len(t) > 160:
            return False

        # Regex for common headings / numbering
        if _heading_re.match(t):
            return True

        # Bigger than body text
        if size >= body_size + 1.5 and re.search(r"[A-Za-z]", t):
            return True

        # ALL CAPS short lines can be headings
        if t.isupper() and len(t) <= 70 and re.search(r"[A-Z]", t):
            return True

        return False
