"""OCR-parse the Chinese GB/T standard PDFs with MinerU.

pymupdf can't read these (embedded fonts with no ToUnicode map -> mojibake), so
we run MinerU's `pipeline` OCR backend and copy the resulting Markdown into
data/parsed/, overwriting any garbled pymupdf output.

Notes:
* First run downloads ~300 MB of models from ModelScope (China-friendly mirror).
* CPU-only, so a full standard takes several to ~20 minutes.
* Rich output (md + page images + layout json) is kept under data/mineru/.

Run: uv run python -m ingest.parse_mineru
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ingest.parse import OCR_REQUIRED, OUT, RAW, count_headings

WORK = Path("data/mineru")  # MinerU's rich output (md + images + json)


def run_mineru(pdf: Path) -> Path:
    env = {**os.environ, "MINERU_MODEL_SOURCE": "modelscope"}
    subprocess.run(
        [
            "mineru", "-p", str(pdf), "-o", str(WORK),
            "-b", "pipeline", "-m", "ocr", "-l", "ch",
        ],
        check=True,
        env=env,
    )
    produced = WORK / pdf.stem / "ocr" / f"{pdf.stem}.md"
    if not produced.exists():
        raise FileNotFoundError(f"MinerU did not produce {produced}")
    return produced


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [RAW / f"{stem}.pdf" for stem in sorted(OCR_REQUIRED)]

    for pdf in targets:
        if not pdf.exists():
            print(f"MISS  {pdf.name} not found in {RAW}/ — skipping")
            continue
        produced = run_mineru(pdf)
        dest = OUT / f"{pdf.stem}.md"
        shutil.copyfile(produced, dest)
        md = dest.read_text(encoding="utf-8")
        print(f"OK    {pdf.name} -> {dest.name}  ({count_headings(md)} headings)")


if __name__ == "__main__":
    main()
