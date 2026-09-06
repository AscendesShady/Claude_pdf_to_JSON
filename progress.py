"""A single event type used to report live pipeline progress up to the GUI."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProgressEvent:
    stage: str  # "extract" | "generate"
    current_task: str = ""
    pages_done: int = 0
    pages_total: int = 0
    chunks_created: int = 0
    examples_generated: int = 0
    qc_passed: int = 0
    qc_rejected: int = 0
    tokens_used: int = 0
    done: int = 0
    total: int = 0
