"""Pydantic validation, dedup, and size/length filtering of generated records."""
from __future__ import annotations

import logging
import re

from pydantic import BaseModel, ValidationError, field_validator

from config import PipelineConfig
from records import RawRecord
from text_chunker import count_tokens

logger = logging.getLogger(__name__)


class ValidatedRecord(BaseModel):
    system: str
    question: str
    answer: str
    source_doc: str
    page_start: int
    page_end: int
    chunk_id: str
    is_table: bool = False

    @field_validator("system", "question", "answer", "source_doc", "chunk_id")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def validate_one(
    rec: RawRecord, config: PipelineConfig, seen_pairs: set[tuple[str, str]]
) -> tuple[ValidatedRecord | None, str | None]:
    """Validate a single record. Returns (record, None) on success or (None, reason) on rejection."""
    try:
        v = ValidatedRecord(
            system=rec.system,
            question=rec.question,
            answer=rec.answer,
            source_doc=rec.source_doc,
            page_start=rec.page_start,
            page_end=rec.page_end,
            chunk_id=rec.chunk_id,
            is_table=rec.is_table,
        )
    except ValidationError as exc:
        return None, f"failed validation ({exc})"

    q_norm = _normalize(v.question)
    a_norm = _normalize(v.answer)
    if not q_norm or not a_norm:
        return None, "empty after normalization"

    key = (q_norm, a_norm)
    if key in seen_pairs:
        return None, "duplicate question/answer pair"

    answer_tokens = count_tokens(v.answer)
    if answer_tokens < 3:
        return None, "answer too short"
    if rec.chunk_token_count and answer_tokens > rec.chunk_token_count * config.length_ratio_limit:
        return None, "answer implausibly long vs source chunk"

    total_tokens = count_tokens(v.system) + count_tokens(v.question) + count_tokens(v.answer)
    if total_tokens > config.max_seq_tokens:
        return None, f"exceeds max_seq_tokens ({total_tokens} > {config.max_seq_tokens})"

    seen_pairs.add(key)
    return v, None


def run_quality_gate(records: list[RawRecord], config: PipelineConfig) -> list[ValidatedRecord]:
    """Batch entry point (used for standalone testing); the live pipeline gates inline
    in dataset_builder.build_records via validate_one so QC counts update as they happen."""
    validated: list[ValidatedRecord] = []
    seen_pairs: set[tuple[str, str]] = set()

    for rec in records:
        v, reason = validate_one(rec, config, seen_pairs)
        if v is None:
            logger.info("Discarding record from %s: %s", rec.chunk_id, reason)
            continue
        validated.append(v)

    logger.info("Quality gate: %d/%d records passed", len(validated), len(records))
    return validated
