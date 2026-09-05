"""Orchestrates extraction -> chunking -> generation+QC (live) -> split -> export."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import dataset_builder
import excel_writer
import jsonl_writer
import pdf_extractor
import splitter
import text_chunker
from config import PipelineConfig
from progress import ProgressEvent

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class PipelineResult:
    total_pages: int
    total_chunks: int
    total_examples_generated: int
    total_qc_passed: int
    total_qc_rejected: int
    split_counts: dict[str, int]
    output_dir: str
    review_workbook_path: str


def run_pipeline(
    pdf_paths: list[str],
    config: PipelineConfig,
    progress_cb: Optional[ProgressCallback] = None,
    stop_event: Optional[threading.Event] = None,
) -> PipelineResult:
    stop_event = stop_event or threading.Event()

    def emit(**kwargs) -> None:
        if progress_cb:
            progress_cb(ProgressEvent(**kwargs))

    pages_total = sum(pdf_extractor.count_pages(p) for p in pdf_paths)
    pages_done = 0
    all_chunks: list[text_chunker.Chunk] = []

    for i, path in enumerate(pdf_paths):
        if stop_event.is_set():
            logger.info("Stop requested; halting extraction")
            break

        name = Path(path).name
        emit(stage="extract", current_task=f"Extracting {name} ({i + 1}/{len(pdf_paths)})",
             pages_done=pages_done, pages_total=pages_total, chunks_created=len(all_chunks))

        logger.info("Extracting %s (%d/%d)", name, i + 1, len(pdf_paths))
        pages = pdf_extractor.extract_pdf(path, config)
        pages_done += len(pages)

        chunks = text_chunker.chunk_document(pages, name, config)
        all_chunks.extend(chunks)

        emit(stage="extract", current_task=f"Chunked {name}: {len(chunks)} chunks",
             pages_done=pages_done, pages_total=pages_total, chunks_created=len(all_chunks))

    logger.info("Total chunks across %d document(s): %d", len(pdf_paths), len(all_chunks))

    def _gen_progress(event: ProgressEvent) -> None:
        event.pages_done = pages_done
        event.pages_total = pages_total
        event.chunks_created = len(all_chunks)
        if progress_cb:
            progress_cb(event)

    validated, stats = dataset_builder.build_records(
        all_chunks, config, progress_cb=_gen_progress, stop_event=stop_event
    )
    splits = splitter.split_records(validated, config)
    counts = jsonl_writer.write_jsonl_outputs(splits, config.output_dir)
    review_path = excel_writer.write_review_workbook(splits, config.output_dir)

    return PipelineResult(
        total_pages=pages_done,
        total_chunks=len(all_chunks),
        total_examples_generated=stats.examples_generated,
        total_qc_passed=stats.qc_passed,
        total_qc_rejected=stats.qc_rejected,
        split_counts=counts,
        output_dir=config.output_dir,
        review_workbook_path=review_path,
    )
