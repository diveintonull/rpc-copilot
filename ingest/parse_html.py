"""Convert HTML law sources (PRC statutes from cac.gov.cn) into Markdown.

These pages are clean UTF-8, so no OCR is needed — but the statute text is
wrapped in a lot of site chrome (nav / header / footer). We isolate the article
body ('main-content' on cac.gov.cn) with BeautifulSoup, then convert only that
to Markdown.

Note: only ASCII is printed to stdout — the Windows console is GBK and crashes
on printing Chinese (incl. the U+200B zero-width space in some titles). File I/O
is always UTF-8.

Run: uv run python -m ingest.parse_html
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify as md

from ingest.parse import OUT, RAW

# CSS class of the article-body container, tried in priority order.
CONTENT_SELECTORS = ["main-content", "TRS_Editor", "content"]


def extract_body(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for cls in CONTENT_SELECTORS:
        el = soup.find(class_=cls)
        if el:
            return str(el)
    return str(soup.body or soup)  # fallback: whole <body>


def clean(text: str) -> str:
    text = text.replace("​", "").replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse runs of blank lines
    return text.strip() + "\n"


def count_articles(text: str) -> int:
    """Distinct 第X条 markers — the structure proxy for a statute (vs # headings)."""
    return len(set(re.findall(r"第[一二三四五六七八九十百千]+条", text)))


def parse_html(src: Path) -> Path:
    body = extract_body(src.read_text(encoding="utf-8"))
    text = clean(md(body, heading_style="ATX"))
    dest = OUT / f"{src.stem}.md"
    dest.write_text(text, encoding="utf-8")
    print(f"OK    {src.name} -> {dest.name}  ({count_articles(text)} articles, {len(text)} chars)")
    return dest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    htmls = sorted(RAW.glob("*.html"))
    if not htmls:
        print(f"No HTML files found in {RAW}/ — download the statute pages there first.")
        return
    for src in htmls:
        parse_html(src)


if __name__ == "__main__":
    main()
