from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


def _fmt_seconds(s: float) -> str:
    s = max(0.0, float(s))
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {sec:02d}s"
    if m > 0:
        return f"{m:d}m {sec:02d}s"
    return f"{sec:d}s"


def _ts() -> str:
    # local time stamp for CLI readability
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class PaperTiming:
    start: float
    end: Optional[float] = None

    @property
    def elapsed(self) -> float:
        return (self.end or time.time()) - self.start


class ConsoleProgress:
    """
    Clean CLI progress reporter with:
      - per-paper step messages
      - per-batch VLM messages
      - overall run progress + ETA

    Usage:
      prog = ConsoleProgress(total_papers=N)
      prog.run_start()
      prog.paper_start(i, N, pdf_name)
      prog.step("Rendering pages...")
      prog.vlm_batch_start(batch_idx, batch_total, page_start, page_end)
      prog.vlm_batch_done(...)
      prog.paper_done(...)
      prog.run_done(...)
    """

    def __init__(self, total_papers: int):
        self.total_papers = int(total_papers)
        self.run_start_time = time.time()
        self.paper_times: list[float] = []
        self.current_paper: Optional[str] = None
        self.current_paper_timing: Optional[PaperTiming] = None

    # ---------------------
    # Run-level
    # ---------------------

    def run_start(self) -> None:
        print(f"[{_ts()}] Run start | Total papers: {self.total_papers}")

    def run_done(self) -> None:
        elapsed = time.time() - self.run_start_time
        avg = (sum(self.paper_times) / len(self.paper_times)) if self.paper_times else 0.0
        print(f"[{_ts()}] Run complete | Elapsed: {_fmt_seconds(elapsed)} | Avg/paper: {_fmt_seconds(avg)}")

    def run_status(self, completed: int) -> None:
        completed = int(completed)
        remaining = max(0, self.total_papers - completed)
        elapsed = time.time() - self.run_start_time
        avg = (sum(self.paper_times) / len(self.paper_times)) if self.paper_times else 0.0
        eta = avg * remaining if avg > 0 else 0.0
        print(
            f"[{_ts()}] Batch status | Done: {completed}/{self.total_papers} | Remaining: {remaining} | "
            f"Elapsed: {_fmt_seconds(elapsed)} | Avg/paper: {_fmt_seconds(avg)} | ETA: {_fmt_seconds(eta)}"
        )

    # ---------------------
    # Paper-level
    # ---------------------

    def paper_start(self, idx: int, pdf_name: str) -> None:
        self.current_paper = pdf_name
        self.current_paper_timing = PaperTiming(start=time.time())
        print(f"\n[{_ts()}] === Paper {idx}/{self.total_papers}: {pdf_name} ===")

    def paper_done(self) -> float:
        if not self.current_paper_timing:
            return 0.0
        self.current_paper_timing.end = time.time()
        elapsed = self.current_paper_timing.elapsed
        self.paper_times.append(elapsed)
        print(f"[{_ts()}] ✅ Paper done | {self.current_paper} | Time: {_fmt_seconds(elapsed)}")
        self.current_paper = None
        self.current_paper_timing = None
        return elapsed

    def paper_fail(self, err: str) -> None:
        print(f"[{_ts()}] ❌ Paper failed | {self.current_paper} | Error: {err}")

    # ---------------------
    # Step-level (within paper)
    # ---------------------

    def step(self, msg: str) -> None:
        prefix = self.current_paper or "Paper"
        print(f"[{_ts()}] [{prefix}] {msg}")

    # ---------------------
    # VLM batch-level (within paper)
    # ---------------------

    def vlm_batch_start(self, batch_idx: int, batch_total: int, page_start: int, page_end: int, prev_section: str | None) -> None:
        prefix = self.current_paper or "Paper"
        carry = f' | carry="{prev_section}"' if prev_section else ""
        print(
            f"[{_ts()}] [{prefix}] VLM batch {batch_idx}/{batch_total} | pages {page_start}-{page_end}{carry}"
        )

    def vlm_batch_done(self, batch_idx: int, extracted_sections: int, new_prev_section: str | None, seconds: float) -> None:
        prefix = self.current_paper or "Paper"
        nxt = f' | next_carry="{new_prev_section}"' if new_prev_section else ""
        print(
            f"[{_ts()}] [{prefix}] VLM batch {batch_idx} done | sections_in_batch={extracted_sections} | "
            f"time={_fmt_seconds(seconds)}{nxt}"
        )
