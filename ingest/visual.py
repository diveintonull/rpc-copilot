"""Prepare governed PDF pages for multimodal retrieval.

Each PDF page is rendered as one image and described by a small, versioned
manifest.  The page image is the retrieval input; OCR/PDF text is retained for
human-readable evidence cards, citations, and debugging.

CLI:
  python -m ingest.visual
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ingest.index import DOCUMENT_SPECS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
MINERU_ROOT = PROJECT_ROOT / "data" / "mineru"
VISUAL_ROOT = PROJECT_ROOT / "data" / "visual"
VISUAL_PAGES_ROOT = VISUAL_ROOT / "pages"
VISUAL_MANIFEST = VISUAL_ROOT / "manifest.json"
MANIFEST_VERSION = 1
DEFAULT_PAGE_DPI = 144


@dataclass(frozen=True, slots=True)
class VisualPage:
    """One rendered, provenance-aware document page."""

    visual_id: str
    document_id: str
    source_id: str
    version: str
    title: str
    page_number: int
    image_relpath: str
    text: str
    content_hash: str

    def image_path(self, pages_root: Path = VISUAL_PAGES_ROOT) -> Path:
        return pages_root / Path(self.image_relpath)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "VisualPage":
        return cls(
            visual_id=str(payload["visual_id"]),
            document_id=str(payload["document_id"]),
            source_id=str(payload["source_id"]),
            version=str(payload["version"]),
            title=str(payload["title"]),
            page_number=int(payload["page_number"]),
            image_relpath=str(payload["image_relpath"]),
            text=str(payload.get("text", "")),
            content_hash=str(payload["content_hash"]),
        )


PageRenderer = Callable[[Path, Path, int, bool], list[Path]]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_directory_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._+-]+", "-", value).strip("-.")
    return normalized or "document"


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        result: list[str] = []
        for nested in value.values():
            result.extend(_flatten_strings(nested))
        return result
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        result = []
        for nested in value:
            result.extend(_flatten_strings(nested))
        return result
    return []


def _record_text(record: Mapping[str, Any]) -> str:
    """Return readable content from a MinerU content-list record."""
    record_type = str(record.get("type", ""))
    fields = {
        "text": ("text",),
        "header": ("text",),
        "footer": ("text",),
        "page_number": ("text",),
        "list": ("list_items",),
        "table": ("table_caption", "table_body", "table_footnote"),
        "image": ("image_caption", "image_footnote"),
    }.get(record_type, ("text",))
    parts: list[str] = []
    for field in fields:
        parts.extend(_flatten_strings(record.get(field)))
    return "\n".join(part for part in parts if part)


def load_mineru_page_text(pdf_stem: str, mineru_root: Path = MINERU_ROOT) -> dict[int, str]:
    """Load MinerU OCR/layout content grouped by one-based page number."""
    path = mineru_root / pdf_stem / "ocr" / f"{pdf_stem}_content_list.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"MinerU content list must be an array: {path}")

    grouped: dict[int, list[str]] = defaultdict(list)
    for record in payload:
        if not isinstance(record, Mapping):
            continue
        page_index = record.get("page_idx")
        if not isinstance(page_index, int) or page_index < 0:
            continue
        text = _record_text(record)
        if text:
            grouped[page_index + 1].append(text)
    return {page: "\n".join(parts) for page, parts in grouped.items()}


def extract_pdf_page_text(pdf_path: Path) -> dict[int, str]:
    """Extract selectable PDF text as a fallback for non-OCR documents."""
    import fitz

    document = fitz.open(pdf_path)
    try:
        return {
            index + 1: page.get_text("text").strip()
            for index, page in enumerate(document)
        }
    finally:
        document.close()


def render_pdf_pages(
    pdf_path: Path,
    target_directory: Path,
    dpi: int = DEFAULT_PAGE_DPI,
    force: bool = False,
) -> list[Path]:
    """Render every PDF page to a stable PNG path."""
    if dpi < 72:
        raise ValueError("page DPI must be at least 72")
    import fitz

    target_directory.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    paths: list[Path] = []
    try:
        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)
        for page_index, page in enumerate(document):
            target = target_directory / f"page-{page_index + 1:04d}.png"
            if force or not target.is_file():
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(target)
            paths.append(target)
    finally:
        document.close()
    return paths


def _document_spec(pdf_path: Path) -> dict[str, Any]:
    spec = DOCUMENT_SPECS.get(pdf_path.stem)
    if spec is None:
        return {
            "source_id": pdf_path.stem,
            "version": "unversioned",
            "title": pdf_path.stem,
        }
    return spec


def prepare_visual_pages(
    *,
    raw_root: Path = RAW_ROOT,
    mineru_root: Path = MINERU_ROOT,
    visual_root: Path = VISUAL_ROOT,
    dpi: int = DEFAULT_PAGE_DPI,
    force: bool = False,
    renderer: PageRenderer = render_pdf_pages,
) -> list[VisualPage]:
    """Render governed PDFs and write a deterministic page manifest."""
    pages_root = visual_root / "pages"
    manifest_path = visual_root / "manifest.json"
    visual_root.mkdir(parents=True, exist_ok=True)
    pages: list[VisualPage] = []
    previous_hashes: dict[str, str] = {}
    previous_dpi: int | None = None
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_dpi = previous.get("dpi")
            previous_dpi = raw_dpi if isinstance(raw_dpi, int) else None
            raw_hashes = previous.get("document_hashes", {})
            if isinstance(raw_hashes, Mapping):
                previous_hashes = {
                    str(key): str(value) for key, value in raw_hashes.items()
                }
        except (AttributeError, json.JSONDecodeError, OSError, TypeError):
            previous_hashes = {}
    document_hashes: dict[str, str] = {}

    for pdf_path in sorted(raw_root.glob("*.pdf")):
        spec = _document_spec(pdf_path)
        source_id = str(spec["source_id"])
        version = str(spec["version"])
        document_id = f"{source_id}@{version}"
        source_hash = _sha256_bytes(pdf_path.read_bytes())
        document_hashes[pdf_path.stem] = source_hash
        target_directory = pages_root / _safe_directory_name(document_id)
        rendered = renderer(
            pdf_path,
            target_directory,
            dpi,
            force
            or previous_dpi != dpi
            or previous_hashes.get(pdf_path.stem) != source_hash,
        )
        page_text = load_mineru_page_text(pdf_path.stem, mineru_root)
        if not page_text:
            page_text = extract_pdf_page_text(pdf_path)
        for page_number, image_path in enumerate(rendered, start=1):
            text = page_text.get(page_number, "").strip()
            image_relpath = image_path.relative_to(pages_root).as_posix()
            visual_id = f"{document_id}#page={page_number}"
            content_hash = _sha256_bytes(
                f"{source_hash}\0{page_number}\0{text}".encode("utf-8")
            )
            pages.append(
                VisualPage(
                    visual_id=visual_id,
                    document_id=document_id,
                    source_id=source_id,
                    version=version,
                    title=str(spec["title"]),
                    page_number=page_number,
                    image_relpath=image_relpath,
                    text=text,
                    content_hash=content_hash,
                )
            )

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dpi": dpi,
        "document_hashes": document_hashes,
        "pages": [page.to_payload() for page in pages],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pages


def load_visual_manifest(path: Path = VISUAL_MANIFEST) -> list[VisualPage]:
    """Load and validate the generated page manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("unsupported visual manifest version")
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        raise ValueError("visual manifest pages must be an array")
    return [VisualPage.from_payload(item) for item in raw_pages if isinstance(item, Mapping)]


def main() -> None:
    pages = prepare_visual_pages()
    print(f"prepared {len(pages)} visual pages in {VISUAL_PAGES_ROOT}")


if __name__ == "__main__":
    main()
