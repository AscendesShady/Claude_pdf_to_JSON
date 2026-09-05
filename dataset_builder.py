"""Turns chunks into canonical Q&A records by prompting the local Ollama model,
validating and deduplicating each candidate inline so QC counts update live."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import ollama_client
import quality_gate
from config import PipelineConfig
from progress import ProgressEvent
from prompts import (
    GROUNDING_RULES,
    PARAPHRASE_TEMPLATE,
    QA_GENERATION_TEMPLATE,
    SYSTEM_PROMPT_VARIANTS,
    TABLE_QA_GENERATION_TEMPLATE,
)
from quality_gate import ValidatedRecord
from records import RawRecord
from text_chunker import Chunk

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class GenerationStats:
    examples_generated: int = 0
    qc_passed: int = 0
    qc_rejected: int = 0


def _insufficient(answer: str) -> bool:
    lowered = answer.strip().lower()
    return "insufficient context" in lowered or "not enough information" in lowered


def _generate_primary(chunk: Chunk, config: PipelineConfig, system: str) -> Optional[dict]:
    template = TABLE_QA_GENERATION_TEMPLATE if chunk.is_table else QA_GENERATION_TEMPLATE
    prompt = template.format(grounding_rules=GROUNDING_RULES, context=chunk.text)
    return ollama_client.generate_json(config, prompt, system)


def _generate_paraphrase(question: str, answer: str, config: PipelineConfig, system: str) -> Optional[dict]:
    prompt = PARAPHRASE_TEMPLATE.format(grounding_rules=GROUNDING_RULES, question=question, answer=answer)
    return ollama_client.generate_json(config, prompt, system)


def build_records(
    chunks: list[Chunk],
    config: PipelineConfig,
    progress_cb: Optional[ProgressCallback] = None,
    stop_event: Optional[threading.Event] = None,
) -> tuple[list[ValidatedRecord], GenerationStats]:
    validated: list[ValidatedRecord] = []
    seen_pairs: set[tuple[str, str]] = set()
    stats = GenerationStats()
    total = len(chunks)

    def _accept_or_reject(candidate: RawRecord) -> None:
        stats.examples_generated += 1
        v, reason = quality_gate.validate_one(candidate, config, seen_pairs)
        if v is not None:
            validated.append(v)
            stats.qc_passed += 1
        else:
            stats.qc_rejected += 1
            logger.info("Rejected record from %s: %s", candidate.chunk_id, reason)

    for i, chunk in enumerate(chunks):
        if stop_event is not None and stop_event.is_set():
            logger.info("Stop requested; halting dataset generation at chunk %d/%d", i, total)
            break

        system = SYSTEM_PROMPT_VARIANTS[i % len(SYSTEM_PROMPT_VARIANTS)]
        result = _generate_primary(chunk, config, system)

        question = ""
        answer = ""
        if result:
            question = str(result.get("question", "")).strip()
            answer = str(result.get("answer", "")).strip()

        if not result:
            logger.info("Skipping chunk %s: generation failed", chunk.chunk_id)
        elif not question or not answer:
            logger.info("Skipping chunk %s: missing question/answer field", chunk.chunk_id)
        elif _insufficient(answer):
            logger.info("Skipping chunk %s: model reported insufficient context", chunk.chunk_id)
        else:
            _accept_or_reject(RawRecord(
                system=system, question=question, answer=answer,
                source_doc=chunk.source_doc, page_start=chunk.page_start, page_end=chunk.page_end,
                chunk_id=chunk.chunk_id, chunk_token_count=chunk.token_count, is_table=chunk.is_table,
            ))

            for _ in range(config.variants_per_chunk):
                if stop_event is not None and stop_event.is_set():
                    break
                variant = _generate_paraphrase(question, answer, config, system)
                if not variant:
                    continue
                v_question = str(variant.get("question", "")).strip()
                if not v_question or v_question.lower() == question.lower():
                    continue
                _accept_or_reject(RawRecord(
                    system=system, question=v_question, answer=answer,
                    source_doc=chunk.source_doc, page_start=chunk.page_start, page_end=chunk.page_end,
                    chunk_id=chunk.chunk_id, chunk_token_count=chunk.token_count, is_table=chunk.is_table,
                ))

        if progress_cb is not None:
            progress_cb(ProgressEvent(
                stage="generate",
                current_task=f"chunk {i + 1}/{total} (pages {chunk.page_start}-{chunk.page_end})",
                done=i + 1,
                total=total,
                examples_generated=stats.examples_generated,
                qc_passed=stats.qc_passed,
                qc_rejected=stats.qc_rejected,
            ))

    logger.info(
        "Generated %d candidate records from %d chunks: %d passed QC, %d rejected",
        stats.examples_generated, total, stats.qc_passed, stats.qc_rejected,
    )
    return validated, stats
