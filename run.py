from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

from config import AgentConfig
from agent.pipeline import PDFVlmAgent
from agent.utils import ensure_dir
from agent.progress import ConsoleProgress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder containing PDFs")
    ap.add_argument("--output", required=True, help="Output folder")

    ap.add_argument("--model", default=None, help="OpenAI model id (vision capable)")
    ap.add_argument("--dpi", type=int, default=None, help="Render DPI for pages")
    ap.add_argument("--min_conf", type=float, default=None, help="Min confidence for crops")

    ap.add_argument("--text_backend", default=None, help="Text backend: vlm (recommended)")
    ap.add_argument("--vlm_text_model", default=None, help="Model for VLM text extraction (defaults to --model)")
    ap.add_argument("--vlm_pages_per_call", type=int, default=None, help="How many pages per VLM call")
    ap.add_argument("--vlm_max_pages", type=int, default=None, help="Limit pages used for VLM text extraction")

    args = ap.parse_args()

    cfg = AgentConfig(
        model=args.model or AgentConfig.model,
        dpi=args.dpi or AgentConfig.dpi,
        min_confidence=args.min_conf or AgentConfig.min_confidence,
        text_backend=args.text_backend or AgentConfig.text_backend,
        vlm_text_model=args.vlm_text_model or AgentConfig.vlm_text_model,
        vlm_pages_per_call=args.vlm_pages_per_call or AgentConfig.vlm_pages_per_call,
        vlm_max_pages=args.vlm_max_pages if args.vlm_max_pages is not None else AgentConfig.vlm_max_pages,
    )

    in_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    ensure_dir(out_dir)

    pdfs = sorted([p for p in in_dir.glob("*.pdf") if p.is_file()])
    if not pdfs:
        raise SystemExit(f"No PDFs found in: {in_dir}")

    client = OpenAI()

    agent = PDFVlmAgent(
        client=client,
        model=cfg.model,
        dpi=cfg.dpi,
        min_confidence=cfg.min_confidence,
        text_backend=cfg.text_backend,
        vlm_text_model=cfg.vlm_text_model,
        vlm_pages_per_call=cfg.vlm_pages_per_call,
        vlm_max_pages=cfg.vlm_max_pages,
    )

    prog = ConsoleProgress(total_papers=len(pdfs))
    prog.run_start()

    run_manifests = []
    completed = 0

    for idx, pdf_path in enumerate(pdfs, start=1):
        prog.paper_start(idx=idx, pdf_name=pdf_path.name)

        try:
            manifest = agent.process_pdf(pdf_path, out_dir, progress=prog)
            run_manifests.append(manifest.to_dict())
            prog.paper_done()
        except Exception as e:
            prog.paper_fail(str(e))
            # Continue to next paper rather than stopping entire run
        finally:
            completed += 1
            prog.run_status(completed)

    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifests, indent=2), encoding="utf-8")
    prog.run_done()
    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
