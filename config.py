from dataclasses import dataclass

@dataclass(frozen=True)
class AgentConfig:
    model: str = "gpt-5.2"
    dpi: int = 200
    max_pdfs: int | None = None

    # Vision object crops
    min_confidence: float = 0.30

    # Debug knobs
    keep_page_pngs: bool = True

    # Text extraction backend: "vlm" recommended when PDFs are messy
    text_backend: str = "vlm"   # options: "vlm" (you can keep "pdfminer"/"pymupdf" later)

    # VLM text extraction settings
    vlm_text_model: str | None = None   # if None, use `model`
    vlm_max_pages: int | None = None    # None = all pages (can be expensive)
    vlm_pages_per_call: int = 6         # batch page images per request
