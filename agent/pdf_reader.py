from __future__ import annotations
from pathlib import Path
import fitz  # PyMuPDF

class PdfReader:
    """
    Uses PyMuPDF for page rendering and (optionally) other PDF utilities.
    For text extraction in the pdfminer.six option, we will not use get_text_dict().
    """

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(str(pdf_path))

    def __len__(self) -> int:
        return len(self.doc)

    def close(self) -> None:
        self.doc.close()

    def render_page_to_png(self, page_index: int, out_path: Path, dpi: int = 200) -> None:
        page = self.doc[page_index]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(out_path))
