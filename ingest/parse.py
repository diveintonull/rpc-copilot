"""Parse raw corpus documents into Markdown for downstream chunking.

Two parsing paths, because the sources differ in quality:

* Text-clean PDFs (e.g. GDPR) -> pymupdf4llm, fast and lossless.
* Chinese GB/T standard PDFs (e.g. GB/T 22239, GB/T 35273) embed fonts with no
  ToUnicode map, so pymupdf produces mojibake. These need OCR and are handled by
  MinerU instead (see ingest/parse_mineru.py). This script SKIPS them so a re-run
  never clobbers the good OCR output in data/parsed/.

For HTML sources (e.g. the PRC laws) convert with markdownify instead.

Run: uv run python -m ingest.parse
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf4llm

RAW = Path("data/raw")
OUT = Path("data/parsed")

# Stems (filename without extension) that must go through OCR / MinerU, not pymupdf.
OCR_REQUIRED = {
    "GBT+22239-2019",
    "GBT+35273-2020",
}


def count_headings(md: str) -> int:
    """Number of Markdown ATX headings — a quick proxy for structure retention."""
    return len(re.findall(r"^#+ ", md, flags=re.MULTILINE))


def parse_pdf(pdf: Path) -> Path:
    md = pymupdf4llm.to_markdown(str(pdf))
    dest = OUT / f"{pdf.stem}.md"
    dest.write_text(md, encoding="utf-8")
    print(f"OK    {pdf.name} -> {dest.name}  ({count_headings(md)} headings)")
    return dest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAW}/ — put the downloaded documents there first.")
        return

    for pdf in pdfs:
        if pdf.stem in OCR_REQUIRED:
            print(f"SKIP  {pdf.name} -> handled by MinerU (uv run python -m ingest.parse_mineru)")
            continue
        parse_pdf(pdf)


if __name__ == "__main__":
    main()
