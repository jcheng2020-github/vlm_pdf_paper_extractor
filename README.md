
# VLM PDF Agent – Section, Figure, and Table Extraction

This project implements a **Vision-Language-Model (VLM) powered PDF processing agent** that automatically:

* Reads academic PDFs (batch mode: multiple PDFs in one folder)
* Extracts **paper title** and **author list**
* Extracts **section-level text** into separate `.txt` files (VLM from page images)
* Renders **page images**
* Detects **figures and tables** using an OpenAI vision model
* Crops figures and tables into individual `.png` files
* Produces **structured JSON manifests** for downstream ML or NLP pipelines
* Prints **detailed step-by-step progress logs**, including **batch-by-batch** VLM extraction and **ETA**

The system is **modular, object-oriented, and easy to debug**, designed for large-scale academic paper processing.

---

## Features

* 📄 **Section-level text extraction** (VLM from page images)
* 🧾 **Title + authors extraction**
* 🖼 **High-resolution page rendering**
* 📊 **Vision-based figure & table detection**
* ✂️ **Accurate bounding-box cropping**
* 🧾 **Structured JSON manifests**
* 🧩 **Clean OOP & multi-module design**
* 🧭 **Clean console progress** (per step, per VLM batch, per paper, overall ETA)

---

## Project Structure

```

vlm_pdf_agent/
├── README.md
├── requirements.txt
├── run.py
├── config.py
├── agent/
│   ├── **init**.py
│   ├── pipeline.py
│   ├── pdf_reader.py
│   ├── vision_detector.py
│   ├── cropper.py
│   ├── manifest.py
│   ├── vlm_text_extractor.py
│   ├── progress.py
│   └── utils.py

````

> Note: `section_extractor.py` is optional/legacy if you previously tried non-VLM text extraction.

---

## Installation

### 1️⃣ Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
````

---

### 2️⃣ Install dependencies via `requirements.txt`

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Set OpenAI API key

```bash
export OPENAI_API_KEY="your_api_key_here"
```

(Windows PowerShell)

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
```

---

## Usage

### Basic run (Bash / macOS / Linux)

Uses **VLM-based text extraction** (default) to extract:

* paper title
* author list
* section-level text from page images
* figures and tables
* detailed progress logs (step-by-step + VLM batch-by-batch + ETA)

```bash
python run.py --input input_pdfs --output output
```

---

### Advanced options (Bash / macOS / Linux)

```bash
python run.py \
  --input /path/to/the/folder/of/input_pdfs \
  --output /path/to/the/folder/of/output \
  --model gpt-5.2 \
  --dpi 300 \
  --min_conf 0.4 \
  --text_backend vlm \
  --vlm_pages_per_call 6 \
  --vlm_max_pages 12
```

**Notes**

* `--text_backend vlm` enables vision-based text extraction (recommended)
* `--vlm_pages_per_call` controls how many page images are sent per VLM request (batch size)
* `--vlm_max_pages` limits pages used for text extraction (cost control). Detection/cropping still uses all pages.

---

### Basic run (PowerShell)

```powershell
python run.py --input "C:\path\to\the\folder\of\input_pdfs" --output "C:\path\to\the\folder\of\output"
```

---

### Advanced options (PowerShell)

> PowerShell uses the backtick <code>`</code> for line continuation (not ``)

```powershell
python run.py `
  --input "C:\path\to\the\folder\of\input_pdfs" `
  --output "C:\path\to\the\folder\of\output" `
  --model gpt-5.2 `
  --dpi 300 `
  --min_conf 0.4 `
  --text_backend vlm `
  --vlm_pages_per_call 6 `
  --vlm_max_pages 12
```

---

### One-line alternative (PowerShell)

```powershell
python run.py --input "C:\path\to\the\folder\of\input_pdfs" --output "C:\path\to\the\folder\of\output" --model gpt-5.2 --dpi 300 --min_conf 0.4 --text_backend vlm --vlm_pages_per_call 6 --vlm_max_pages 12
```

---

## Progress Logs

This project prints **clean console logs** (no tqdm), including:

* per-paper start/end timing
* step-level progress (render → VLM meta → VLM section batches → detect/crop → manifest)
* VLM section extraction **batch-by-batch** with carry-over context
* overall batch status with **Done / Remaining / Avg per paper / ETA**

Example (illustrative):

```
[2026-01-14 10:05:12] Run start | Total papers: 20

[2026-01-14 10:05:12] === Paper 1/20: paperA.pdf ===
[2026-01-14 10:05:12] [paperA.pdf] Step 1/4: Render pages -> PNG (pages=12, dpi=300)
[2026-01-14 10:05:20] [paperA.pdf] Step 2/4: VLM text extraction (model=gpt-5.2, pages_per_call=6, using_pages=12/12)
[2026-01-14 10:05:20] [paperA.pdf] VLM batch 1/2 | pages 1-6
[2026-01-14 10:05:40] [paperA.pdf] VLM batch 1 done | sections_in_batch=3 | time=20s | next_carry="Methods"
[2026-01-14 10:05:40] [paperA.pdf] VLM batch 2/2 | pages 7-12 | carry="Methods"
[2026-01-14 10:05:56] [paperA.pdf] VLM batch 2 done | sections_in_batch=2 | time=16s | next_carry="Results"
[2026-01-14 10:06:10] [paperA.pdf] Step 3/4: Detect & crop tables/figures (min_conf=0.4)
[2026-01-14 10:06:45] [paperA.pdf] Step 4/4: Write manifest.json
[2026-01-14 10:06:46] ✅ Paper done | paperA.pdf | Time: 1m 34s
[2026-01-14 10:06:46] Batch status | Done: 1/20 | Remaining: 19 | Elapsed: 1m 34s | Avg/paper: 1m 34s | ETA: 29m 46s
```

---

## What this run produces

For each PDF in the input folder:

* `title.txt` — paper title
* `authors.txt` — author list
* `section_text_vlm/*.txt` — one file per section (filename = section name)
* `text_manifest.json` — structured VLM text output (`title`, `authors`, `sections`)
* `pages/*.png` — rendered pages
* `tables/*.png` — cropped tables
* `figures/*.png` — cropped figures
* `manifest.json` — full per-paper output record (includes title/authors/sections + page items)

A global summary is also generated:

* `run_manifest.json` — list of per-paper manifests for the whole run

---

## Recommended settings

| Scenario                 | Suggested flags                 |
| ------------------------ | ------------------------------- |
| Small papers (≤10 pages) | default                         |
| Long papers              | `--vlm_max_pages 10–15`         |
| Dense layouts            | `--dpi 300`                     |
| Reduce false crops       | `--min_conf 0.5`                |
| Fewer VLM calls          | increase `--vlm_pages_per_call` |

---

## Notes for Windows users

* Ensure your virtual environment is activated:

  ```powershell
  .\.venv\Scripts\Activate
  ```

* If `python` is not recognized, try:

  ```powershell
  py run.py --input input_pdfs --output output
  ```

---

## Meaning of `--input` and `--output`

### `--input input_pdfs`

* **Type:** Path to a **folder (directory)**
* **Purpose:** Specifies the directory that contains the PDF files to be processed
* **Behavior:**

  * The agent scans this folder for `*.pdf`
  * **All PDFs inside the folder** are processed
  * Subfolders are **not** scanned (non-recursive by default)

✅ **Correct examples**

```powershell
--input input_pdfs
--input C:\Users\Alice\Documents\papers
--input ./data/pdfs
```

❌ **Incorrect example**

```powershell
--input paper1.pdf   # ❌ single file paths are not supported
```

> If you need single-PDF processing, place the PDF alone in a folder.

---

### `--output output`

* **Type:** Path to a **folder (directory)**
* **Purpose:** Specifies where all extracted results will be written
* **Behavior:**

  * A **subfolder is created per PDF**, named after the PDF filename (without `.pdf`)
  * Existing output folders are reused (files may be overwritten)

Example:

```powershell
--output output
```

Creates:

```
output/
├── paper1/
│   ├── section_text_vlm/
│   ├── pages/
│   ├── tables/
│   ├── figures/
│   ├── title.txt
│   ├── authors.txt
│   ├── text_manifest.json
│   └── manifest.json
├── paper2/
│   └── ...
└── run_manifest.json
```

---

### Summary Table

| Argument   | Expects         | Description                                |
| ---------- | --------------- | ------------------------------------------ |
| `--input`  | **Folder path** | Directory containing one or more PDF files |
| `--output` | **Folder path** | Directory where all results are written    |

---

### Common Mistakes

❌ Passing a file instead of a folder
❌ Using a folder without any PDFs
❌ Forgetting write permissions for the output directory

---

### Best Practice

Use **absolute paths** if unsure:

```powershell
python run.py `
  --input C:\data\pdfs `
  --output C:\data\results
```

---

## Output Structure

For each PDF (`paper1.pdf`):

```
output/paper1/
├── title.txt
├── authors.txt
├── text_manifest.json
├── section_text_vlm/
│   ├── Abstract.txt
│   ├── Introduction.txt
│   ├── Methods.txt
│   └── Results.txt
├── pages/
│   ├── page_001.png
│   ├── page_002.png
├── tables/
│   ├── table_p002_01.png
├── figures/
│   ├── figure_p003_01.png
└── manifest.json
```

A global summary is also generated:

```
output/run_manifest.json
```

---

## Manifest Contents (Simplified)

```json
{
  "pdf": "input_pdfs/paper1.pdf",
  "output_dir": "output/paper1",
  "title": "Paper Title Here",
  "authors": ["Author A", "Author B"],
  "sections": [
    { "name": "Methods", "file": "output/paper1/section_text_vlm/Methods.txt" }
  ],
  "pages": [
    {
      "page": 2,
      "page_png": "output/paper1/pages/page_002.png",
      "items": [
        {
          "type": "table",
          "caption": "Table 1. Dataset statistics",
          "confidence": 0.92,
          "image": "output/paper1/tables/table_p002_01.png",
          "bbox_norm": { "x0": 0.1, "y0": 0.4, "x1": 0.9, "y1": 0.7 },
          "bbox_px": { "x0": 120, "y0": 480, "x1": 1080, "y1": 840 }
        }
      ]
    }
  ]
}
```

---

## Architecture Overview

### Main Orchestrator

* **`PDFVlmAgent`**

  * Stateless, reusable pipeline
  * Coordinates page rendering, VLM text extraction, vision detection, cropping, and manifests

### Core Components

| Module                  | Responsibility                                                          |
| ----------------------- | ----------------------------------------------------------------------- |
| `pdf_reader.py`         | PDF loading & page rendering (PyMuPDF)                                  |
| `vlm_text_extractor.py` | VLM extraction of title/authors/sections (batched + carry-over context) |
| `vision_detector.py`    | OpenAI VLM detection of tables/figures (JSON schema)                    |
| `cropper.py`            | Bounding-box cropping                                                   |
| `manifest.py`           | Structured output serialization                                         |
| `progress.py`           | Clean console progress logging + ETA                                    |
| `utils.py`              | Helpers (paths, slugify, image data URLs)                               |

---

## Debugging & Development Tips

* Inspect `pages/*.png` to verify rendering quality
* Increase `--dpi` if small text is missed
* Increase `--min_conf` to suppress false-positive crops
* Limit cost with `--vlm_max_pages` and tune `--vlm_pages_per_call`
* If a page fails detection, check `PageRecord.error` in `manifest.json`

---

## Typical Use Cases

* Systematic literature reviews
* ML dataset documentation extraction
* Neuroscience / medical AI surveys
* Paper-to-database ingestion
* Vision-language research pipelines

---

## Roadmap / Extensions

* OCR tables → CSV
* Caption-to-section linking
* Multiprocessing for large corpora
* Integration with RAG pipelines
* Canonicalization of section names (e.g., “Materials and Methods” → “Methods”)

---

## License

MIT (or adapt as needed)

```

