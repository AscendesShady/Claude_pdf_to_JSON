"""Shared record types, split out to avoid a circular import between
dataset_builder (produces RawRecord) and quality_gate (validates RawRecord)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RawRecord:
    system: str
    question: str
    answer: str
    source_doc: str
    page_start: int
    page_end: int
    chunk_id: str
    chunk_token_count: int
    is_table: bool
