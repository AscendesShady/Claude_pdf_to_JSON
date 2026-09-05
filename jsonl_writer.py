"""Writes validated records out as ChatML and Alpaca JSONL files, per split."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from quality_gate import ValidatedRecord

logger = logging.getLogger(__name__)


def _to_alpaca(r: ValidatedRecord) -> dict:
    return {
        "instruction": r.question,
        "input": "",
        "output": r.answer,
        "metadata": {
            "source_doc": r.source_doc,
            "page_start": r.page_start,
            "page_end": r.page_end,
            "chunk_id": r.chunk_id,
        },
    }


def _to_chatml(r: ValidatedRecord) -> dict:
    return {
        "messages": [
            {"role": "system", "content": r.system},
            {"role": "user", "content": r.question},
            {"role": "assistant", "content": r.answer},
        ],
        "metadata": {
            "source_doc": r.source_doc,
            "page_start": r.page_start,
            "page_end": r.page_end,
            "chunk_id": r.chunk_id,
        },
    }


def write_jsonl_outputs(
    splits: dict[str, list[ValidatedRecord]], output_dir: str
) -> dict[str, int]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for split_name, records in splits.items():
        alpaca_path = out_dir / f"{split_name}_alpaca.jsonl"
        chatml_path = out_dir / f"{split_name}_chatml.jsonl"
        with alpaca_path.open("w", encoding="utf-8") as fa, chatml_path.open("w", encoding="utf-8") as fc:
            for r in records:
                fa.write(json.dumps(_to_alpaca(r), ensure_ascii=False) + "\n")
                fc.write(json.dumps(_to_chatml(r), ensure_ascii=False) + "\n")
        counts[split_name] = len(records)
        logger.info("Wrote %d records to %s and %s", len(records), alpaca_path.name, chatml_path.name)

    return counts
