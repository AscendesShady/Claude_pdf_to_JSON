"""PDF text/table extraction: page-aware, skips scanned pages, strips boilerplate and PII."""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz

from config import PipelineConfig

logger = logging.getLogger(__name__)

_PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[AWS_KEY]"),
    (re.compile(r"\b(sk|pk)-[A-Za-z0-9]{16,}\b"), "[SECRET]"),
]
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RUN_RE = re.compile(r"[ \t]{2,}")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass
class PageBlock:
    page_num: int  # 1-indexed
    text: str
    tables: list[str] = field(default_factory=list)


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG_RE.sub(" ", text)
    for pattern, repl in _PII_PATTERNS:
        text = pattern.sub(repl, text)
    text = _WHITESPACE_RUN_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _truncate_at_boilerplate(text: str, boilerplate_titles: tuple[str, ...]) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip().strip(":").lower()
        if len(stripped) < 40 and stripped in boilerplate_titles:
            return "\n".join(lines[:i]).strip()
    return text


def _extract_tables(page: "fitz.Page") -> tuple[list[str], list["fitz.Rect"]]:
    table_strings: list[str] = []
    bboxes: list[fitz.Rect] = []
    try:
        finder = page.find_tables()
    except Exception as exc:  # pragma: no cover - defensive against malformed pages
        logger.debug("Table detection failed on page %s: %s", page.number + 1, exc)
        return table_strings, bboxes

    for table in finder.tables:
        try:
            rows = table.extract()
        except Exception as exc:  # pragma: no cover
            logger.debug("Table extraction failed on page %s: %s", page.number + 1, exc)
            continue
        if not rows or len(rows) < 2:
            continue
        lines = [" | ".join((cell or "").strip() for cell in row) for row in rows]
        table_strings.append("\n".join(lines))
        bboxes.append(fitz.Rect(table.bbox))
    return table_strings, bboxes


def _extract_body_text(page: "fitz.Page", table_bboxes: list["fitz.Rect"]) -> str:
    blocks = page.get_text("blocks")
    parts = []
    for block in blocks:
        rect = fitz.Rect(block[:4])
        block_text = block[4]
        if any(rect.intersects(tb) for tb in table_bboxes):
            continue
        parts.append(block_text)
    return "\n".join(parts)


def count_pages(pdf_path: str | Path) -> int:
    """Cheap page count for progress reporting; does not extract any text."""
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def extract_pdf(pdf_path: str | Path, config: PipelineConfig) -> list[PageBlock]:
    """Extract cleaned per-page text and tables from a PDF, skipping scanned pages."""
    pdf_path = Path(pdf_path)
    pages: list[PageBlock] = []
    skipped_scanned = 0

    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_num = page.number + 1
            table_strings, table_bboxes = _extract_tables(page)
            raw_text = _extract_body_text(page, table_bboxes)

            if len(raw_text.strip()) < config.min_page_text_chars and not table_strings:
                skipped_scanned += 1
                logger.info(
                    "Skipping page %d of %s: no extractable text (likely scanned)",
                    page_num, pdf_path.name,
                )
                continue

            cleaned = _clean_text(raw_text)
            cleaned = _truncate_at_boilerplate(cleaned, config.boilerplate_section_titles)
            cleaned_tables = [_clean_text(t) for t in table_strings]

            if not cleaned.strip() and not cleaned_tables:
                continue

            pages.append(PageBlock(page_num=page_num, text=cleaned, tables=cleaned_tables))

    logger.info(
        "Extracted %d usable pages from %s (%d skipped as scanned/empty)",
        len(pages), pdf_path.name, skipped_scanned,
    )
    return pages
