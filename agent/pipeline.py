from __future__ import annotations
from pathlib import Path

from openai import OpenAI

from .pdf_reader import PdfReader
from .vision_detector import VisionDetector
from .cropper import Cropper
from .manifest import PdfManifest, SectionRecord, PageRecord, PageItemRecord
from .utils import ensure_dir, slugify
from .vlm_text_extractor import VlmTextExtractor
from .progress import ConsoleProgress


class PDFVlmAgent:
    """
    Orchestrates:
      - page renders
      - VLM-based text extraction (title/authors/sections) with batch progress
      - table/figure detection + crops
      - manifest writing
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        dpi: int,
        min_confidence: float = 0.30,
        text_backend: str = "vlm",
        vlm_text_model: str | None = None,
        vlm_pages_per_call: int = 6,
        vlm_max_pages: int | None = None,
    ):
        self.client = client
        self.model = model
        self.dpi = dpi
        self.min_confidence = min_confidence

        self.text_backend = text_backend
        self.vlm_text_model = vlm_text_model or model
        self.vlm_pages_per_call = vlm_pages_per_call
        self.vlm_max_pages = vlm_max_pages

        self.detector = VisionDetector(client, model)
        self.cropper = Cropper()
        self.text_extractor_vlm = VlmTextExtractor(
            client=client,
            model=self.vlm_text_model,
            pages_per_call=self.vlm_pages_per_call
        )

    def process_pdf(self, pdf_path: Path, out_root: Path, progress: ConsoleProgress | None = None) -> PdfManifest:
        pdf_name = pdf_path.stem
        out_dir = out_root / pdf_name

        pages_dir = out_dir / "pages"
        tables_dir = out_dir / "tables"
        figures_dir = out_dir / "figures"
        section_dir_vlm = out_dir / "section_text_vlm"

        ensure_dir(pages_dir)
        ensure_dir(tables_dir)
        ensure_dir(figures_dir)
        ensure_dir(section_dir_vlm)

        reader = PdfReader(pdf_path)

        title: str | None = None
        authors: list[str] | None = None
        section_records: list[SectionRecord] = []
        page_records: list[PageRecord] = []

        def log(msg: str) -> None:
            if progress:
                progress.step(msg)

        try:
            # 1) Render pages
            n_pages = len(reader)
            log(f"Step 1/4: Render pages -> PNG (pages={n_pages}, dpi={self.dpi})")
            page_pngs: list[Path] = []
            for pi in range(n_pages):
                page_png = pages_dir / f"page_{pi+1:03d}.png"
                reader.render_page_to_png(pi, page_png, dpi=self.dpi)
                page_pngs.append(page_png)
            log("Step 1/4 done: Page PNGs rendered.")

            # 2) VLM text extraction
            if self.text_backend.lower() == "vlm":
                max_pages = self.vlm_max_pages if self.vlm_max_pages is not None else n_pages
                max_pages = min(max_pages, n_pages)

                log(
                    f"Step 2/4: VLM text extraction (model={self.vlm_text_model}, "
                    f"pages_per_call={self.vlm_pages_per_call}, using_pages={max_pages}/{n_pages})"
                )

                vlm_pages = page_pngs[:max_pages]
                text_result = self.text_extractor_vlm.extract_from_pages(vlm_pages, progress=progress)

                title = text_result.title
                authors = text_result.authors

                log("Writing title.txt, authors.txt, section_text_vlm/*.txt, text_manifest.json ...")
                (out_dir / "title.txt").write_text(title or "", encoding="utf-8")
                (out_dir / "authors.txt").write_text("\n".join(authors or []), encoding="utf-8")

                for sec in text_result.sections:
                    sec_name = sec["name"]
                    sec_text = sec["text"]
                    fname = slugify(sec_name) + ".txt"
                    fpath = section_dir_vlm / fname
                    fpath.write_text(sec_text, encoding="utf-8")
                    section_records.append(SectionRecord(name=sec_name, file=str(fpath)))

                (out_dir / "text_manifest.json").write_text(
                    __import__("json").dumps(text_result.to_dict(), indent=2),
                    encoding="utf-8"
                )
                log(f"Step 2/4 done: Sections written={len(section_records)}")
            else:
                log("Step 2/4 skipped: text_backend != 'vlm'")

            # 3) Detect + crop tables/figures per page
            log(f"Step 3/4: Detect & crop tables/figures (min_conf={self.min_confidence})")
            total_items = 0

            for pi, page_png in enumerate(page_pngs):
                try:
                    det = self.detector.detect(page_png)
                    items = det.get("items", [])
                except Exception as e:
                    page_records.append(PageRecord(
                        page=pi + 1,
                        page_png=str(page_png),
                        items=[],
                        error=str(e)
                    ))
                    continue

                page_items: list[PageItemRecord] = []
                idx_table = 0
                idx_fig = 0

                for item in items:
                    conf = float(item.get("confidence", 0.0))
                    if conf < self.min_confidence:
                        continue

                    obj_type = item["type"]
                    bbox = item["bbox"]
                    caption = item.get("caption", "")

                    if obj_type == "table":
                        idx_table += 1
                        out_img = tables_dir / f"table_p{pi+1:03d}_{idx_table:02d}.png"
                    else:
                        idx_fig += 1
                        out_img = figures_dir / f"figure_p{pi+1:03d}_{idx_fig:02d}.png"

                    crop_res = self.cropper.crop_to_png(page_png, bbox, out_img)

                    page_items.append(PageItemRecord(
                        type=obj_type,
                        caption=caption,
                        confidence=conf,
                        bbox_norm=bbox,
                        bbox_px=crop_res.bbox_px,
                        image=str(out_img),
                    ))
                    total_items += 1

                page_records.append(PageRecord(
                    page=pi + 1,
                    page_png=str(page_png),
                    items=page_items,
                    error=None
                ))

            log(f"Step 3/4 done: Cropped items saved={total_items}")

            # 4) Save manifest
            log("Step 4/4: Write manifest.json")
            manifest = PdfManifest(
                pdf=str(pdf_path),
                output_dir=str(out_dir),
                title=title,
                authors=authors,
                sections=section_records,
                pages=page_records,
            )
            manifest.save(out_dir / "manifest.json")
            log("Step 4/4 done: manifest.json written.")

            return manifest

        finally:
            reader.close()
