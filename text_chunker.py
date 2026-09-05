"""Hierarchical fallback chunking: hard sections -> paragraphs -> sentences -> lines."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from config import PipelineConfig
from pdf_extractor import PageBlock

logger = logging.getLogger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass
class Chunk:
    chunk_id: str
    source_doc: str
    page_start: int
    page_end: int
    text: str
    is_table: bool
    token_count: int


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
_ABBR_RE = re.compile(
    r"\b(mr|mrs|ms|dr|prof|sr|jr|vs|etc|fig|no|approx|st|ave|eq|sec|ch|vol|pp|p|e\.g|i\.e)\.\s*$",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^(chapter|section)\s+\d+", re.IGNORECASE | re.MULTILINE)
_ALLCAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 \-:]{4,59}$", re.MULTILINE)


def split_into_sentences(text: str) -> list[str]:
    raw = _SENTENCE_SPLIT_RE.split(text.strip())
    sentences: list[str] = []
    i = 0
    while i < len(raw):
        current = raw[i]
        while i + 1 < len(raw) and _ABBR_RE.search(current):
            i += 1
            current = current + " " + raw[i]
        if current.strip():
            sentences.append(current.strip())
        i += 1
    return sentences


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def _inject_hard_cuts(text: str) -> str:
    def _mark(m: re.Match) -> str:
        return "\n\n\n" + m.group(0)

    text = _HEADING_RE.sub(_mark, text)
    text = _ALLCAPS_HEADING_RE.sub(_mark, text)
    return text


def _atomize_paragraph(paragraph: str, target_tokens: int) -> list[str]:
    """Break an oversized paragraph into sentences, then lines, as a last resort."""
    if count_tokens(paragraph) <= target_tokens:
        return [paragraph]

    sentences = split_into_sentences(paragraph)
    if len(sentences) <= 1:
        lines = _split_lines(paragraph)
        return lines if len(lines) > 1 else [paragraph]

    atoms: list[str] = []
    for sent in sentences:
        if count_tokens(sent) <= target_tokens:
            atoms.append(sent)
        else:
            lines = _split_lines(sent)
            atoms.extend(lines if len(lines) > 1 else [sent])
    return atoms


def _take_overlap(
    units: list[tuple[int, str, int]], overlap_tokens: int
) -> list[tuple[int, str, int]]:
    result: list[tuple[int, str, int]] = []
    total = 0
    for page_num, text, tokens in reversed(units):
        if result and total + tokens > overlap_tokens:
            break
        result.insert(0, (page_num, text, tokens))
        total += tokens
    return result


def _pack_units(
    atoms: list[tuple[int, str, int]], target_tokens: int, overlap_tokens: int
) -> list[tuple[int, int, str]]:
    packed: list[tuple[int, int, str]] = []
    current: list[tuple[int, str, int]] = []
    current_tokens = 0

    for page_num, text, tokens in atoms:
        if current and current_tokens + tokens > target_tokens:
            packed.append((current[0][0], current[-1][0], " ".join(u[1] for u in current)))
            current = _take_overlap(current, overlap_tokens)
            current_tokens = sum(u[2] for u in current)
        current.append((page_num, text, tokens))
        current_tokens += tokens

    if current:
        packed.append((current[0][0], current[-1][0], " ".join(u[1] for u in current)))
    return packed


def _split_into_hard_sections(
    pages: list[PageBlock], hard_cut_pattern: str
) -> list[list[tuple[int, str]]]:
    # Hard-cut boundaries must be split BEFORE paragraph splitting: paragraph splitting
    # (on a single blank line) would otherwise consume the very blank-line run that marks
    # a hard cut, leaving no trace of it in the resulting paragraph strings to detect.
    hard_cut_re = re.compile(hard_cut_pattern)
    sections: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []

    for page in pages:
        marked = _inject_hard_cuts(page.text)
        pieces = hard_cut_re.split(marked)
        for idx, piece in enumerate(pieces):
            if idx > 0 and current:
                sections.append(current)
                current = []
            for para in _split_paragraphs(piece):
                if para:
                    current.append((page.page_num, para))

    if current:
        sections.append(current)
    return sections


def chunk_document(pages: list[PageBlock], source_doc: str, config: PipelineConfig) -> list[Chunk]:
    target = config.target_chunk_tokens
    overlap = config.chunk_overlap_tokens()
    doc_stem = Path(source_doc).stem
    chunks: list[Chunk] = []
    counter = 0

    for page in pages:
        for table_text in page.tables:
            tokens = count_tokens(table_text)
            if tokens < config.min_chunk_tokens:
                continue
            if tokens > config.max_chunk_tokens:
                logger.info(
                    "Discarding oversized table on page %d of %s (%d tokens)",
                    page.page_num, source_doc, tokens,
                )
                continue
            counter += 1
            chunks.append(Chunk(
                chunk_id=f"{doc_stem}_c{counter:05d}",
                source_doc=source_doc,
                page_start=page.page_num,
                page_end=page.page_num,
                text=table_text,
                is_table=True,
                token_count=tokens,
            ))

    for section in _split_into_hard_sections(pages, config.hard_cut_pattern):
        atoms: list[tuple[int, str, int]] = []
        for page_num, para in section:
            for atom_text in _atomize_paragraph(para, target):
                atoms.append((page_num, atom_text, count_tokens(atom_text)))

        for page_start, page_end, text in _pack_units(atoms, target, overlap):
            tokens = count_tokens(text)
            if tokens < config.min_chunk_tokens:
                continue
            if tokens > config.max_chunk_tokens:
                logger.info(
                    "Discarding oversized chunk spanning pages %d-%d of %s (%d tokens)",
                    page_start, page_end, source_doc, tokens,
                )
                continue
            counter += 1
            chunks.append(Chunk(
                chunk_id=f"{doc_stem}_c{counter:05d}",
                source_doc=source_doc,
                page_start=page_start,
                page_end=page_end,
                text=text,
                is_table=False,
                token_count=tokens,
            ))

    logger.info("Built %d chunks from %s", len(chunks), source_doc)
    return chunks
