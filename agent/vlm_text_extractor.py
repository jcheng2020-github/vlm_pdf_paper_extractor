from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .utils import image_to_data_url
from .progress import ConsoleProgress


VLM_TEXT_SCHEMA: Dict[str, Any] = {
    "name": "paper_text_structure",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "authors": {"type": "array", "items": {"type": "string"}},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["name", "text"]
                }
            }
        },
        "required": ["title", "authors", "sections"]
    }
}


@dataclass
class VlmTextResult:
    title: str
    authors: List[str]
    sections: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "authors": self.authors, "sections": self.sections}


class VlmTextExtractor:
    """
    VLM-based text extraction:
      - title
      - authors
      - section blocks

    Carry-over section context:
      Pass last section name from previous batch into next batch prompt so the model
      continues long sections even when the header isn't repeated.

    Progress reporting:
      If `progress` is provided, prints batch-by-batch logs.
    """

    def __init__(self, client: OpenAI, model: str, pages_per_call: int = 6):
        self.client = client
        self.model = model
        self.pages_per_call = max(1, pages_per_call)

    def extract_from_pages(self, page_pngs: List[Path], progress: ConsoleProgress | None = None) -> VlmTextResult:
        if not page_pngs:
            return VlmTextResult(title="", authors=[], sections=[])

        # Phase A: title + authors from first 2 pages max
        if progress:
            progress.step("VLM meta extraction (title + authors) from first 1–2 pages...")
        meta_pages = page_pngs[:2]
        meta = self._extract_meta(meta_pages)

        # Phase B: sections batched with carry-over
        if progress:
            progress.step(f"VLM section extraction in batches (pages_per_call={self.pages_per_call})...")
        sections = self._extract_sections_batched(page_pngs, progress=progress)

        # Merge duplicate section names by concatenation
        merged_sections: List[Dict[str, str]] = []
        seen: Dict[str, int] = {}
        for sec in sections:
            name = (sec.get("name") or "").strip() or "Unknown"
            text = (sec.get("text") or "").strip()
            if not text:
                continue

            if name in seen:
                merged_sections[seen[name]]["text"] += "\n\n" + text
            else:
                seen[name] = len(merged_sections)
                merged_sections.append({"name": name, "text": text})

        return VlmTextResult(
            title=(meta.get("title") or "").strip(),
            authors=[a.strip() for a in (meta.get("authors") or []) if a and a.strip()],
            sections=merged_sections
        )

    # -----------------------
    # Meta extraction
    # -----------------------

    def _extract_meta(self, page_pngs: List[Path]) -> Dict[str, Any]:
        content = [{"type": "input_text", "text": self._meta_prompt()}]
        for p in page_pngs:
            content.append({"type": "input_image", "image_url": image_to_data_url(p)})

        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "paper_meta",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["title", "authors"]
                    },
                    "strict": True
                }
            }
        )
        return json.loads(resp.output_text.strip())

    # -----------------------------------------
    # Section extraction (batched + carry-over)
    # -----------------------------------------

    def _extract_sections_batched(self, page_pngs: List[Path], progress: ConsoleProgress | None) -> List[Dict[str, str]]:
        all_sections: List[Dict[str, str]] = []
        prev_section: Optional[str] = None

        total_batches = (len(page_pngs) + self.pages_per_call - 1) // self.pages_per_call

        for b, i in enumerate(range(0, len(page_pngs), self.pages_per_call), start=1):
            batch = page_pngs[i:i + self.pages_per_call]

            # page indices for logging (1-based)
            page_start = i + 1
            page_end = i + len(batch)

            if progress:
                progress.vlm_batch_start(
                    batch_idx=b,
                    batch_total=total_batches,
                    page_start=page_start,
                    page_end=page_end,
                    prev_section=prev_section
                )

            t0 = time.time()
            batch_sections = self._extract_sections(batch, prev_section=prev_section)
            dt = time.time() - t0

            all_sections.extend(batch_sections)

            new_prev = self._last_section_name(batch_sections) or prev_section
            if progress:
                progress.vlm_batch_done(
                    batch_idx=b,
                    extracted_sections=len(batch_sections),
                    new_prev_section=new_prev,
                    seconds=dt
                )

            prev_section = new_prev

        return all_sections

    def _extract_sections(self, page_pngs: List[Path], prev_section: Optional[str]) -> List[Dict[str, str]]:
        content = [{"type": "input_text", "text": self._sections_prompt(prev_section)}]
        for p in page_pngs:
            content.append({"type": "input_image", "image_url": image_to_data_url(p)})

        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": VLM_TEXT_SCHEMA["name"],
                    "schema": VLM_TEXT_SCHEMA["schema"],
                    "strict": True
                }
            }
        )

        data = json.loads(resp.output_text.strip())
        return data.get("sections", [])

    @staticmethod
    def _last_section_name(sections: List[Dict[str, str]]) -> Optional[str]:
        for sec in reversed(sections):
            name = (sec.get("name") or "").strip()
            if name:
                return name
        return None

    # -----------------------
    # Prompts
    # -----------------------

    @staticmethod
    def _meta_prompt() -> str:
        return (
            "From these academic paper pages, extract:\n"
            "1) The full paper title (as shown on the first page)\n"
            "2) The list of author names in order\n\n"
            "Return JSON with keys: title (string), authors (array of strings).\n"
            "Do not include affiliations or emails in authors; only names."
        )

    @staticmethod
    def _sections_prompt(prev_section: Optional[str]) -> str:
        carry = ""
        if prev_section and prev_section.strip():
            carry = (
                f"\nCarry-over context:\n"
                f"- The previous batch ended inside the section named: \"{prev_section.strip()}\".\n"
                f"- At the start of THESE pages, treat text as continuing \"{prev_section.strip()}\" "
                f"UNLESS you clearly see a new TOP-LEVEL section heading (e.g., Results, Discussion).\n"
                f"- If the section heading is not repeated, still label the continued text under "
                f"\"{prev_section.strip()}\" until a new top-level heading appears.\n"
            )

        return (
            "You are extracting structured text from academic paper page images.\n\n"
            "Task:\n"
            "- Identify TOP-LEVEL section headings (Abstract, Introduction, Methods, Results, Discussion, Conclusion, References, etc.).\n"
            "- Extract the full readable text under each section.\n"
            "- Preserve true reading order (including multi-column layouts).\n"
            "- Do NOT invent content. If text is unreadable, omit it.\n"
            "- Ignore headers/footers/page numbers.\n"
            "- Keep the text plain (no markdown), but keep paragraph breaks.\n"
            f"{carry}\n"
            "Output MUST be valid JSON matching this schema:\n"
            "{\n"
            '  "title": string,\n'
            '  "authors": string[],\n'
            '  "sections": [ { "name": string, "text": string } ]\n'
            "}\n\n"
            "Notes:\n"
            "- For this call, you may set title to \"\" and authors to [] if not visible.\n"
            "- Use consistent section names across pages (e.g., always \"Methods\").\n"
        )
