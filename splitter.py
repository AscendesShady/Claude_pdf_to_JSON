"""Splits validated records into train/val/test, grouped by source document to avoid leakage."""
from __future__ import annotations

import logging
import random

from config import PipelineConfig
from quality_gate import ValidatedRecord

logger = logging.getLogger(__name__)


def split_records(
    records: list[ValidatedRecord], config: PipelineConfig
) -> dict[str, list[ValidatedRecord]]:
    docs = sorted({r.source_doc for r in records})
    rng = random.Random(config.split_seed)

    if len(docs) >= 3:
        rng.shuffle(docs)
        n = len(docs)
        n_test = max(1, round(n * config.test_ratio))
        n_val = max(1, round(n * config.val_ratio))
        n_train = max(1, n - n_val - n_test)
        train_docs = set(docs[:n_train])
        val_docs = set(docs[n_train:n_train + n_val])
        test_docs = set(docs[n_train + n_val:])

        splits: dict[str, list[ValidatedRecord]] = {"train": [], "val": [], "test": []}
        for r in records:
            if r.source_doc in train_docs:
                splits["train"].append(r)
            elif r.source_doc in val_docs:
                splits["val"].append(r)
            else:
                splits["test"].append(r)

        logger.info(
            "Document-level split: %d train docs, %d val docs, %d test docs",
            len(train_docs), len(val_docs), len(test_docs),
        )
        return splits

    logger.warning(
        "Only %d source document(s) available; falling back to a chunk-level shuffle split "
        "(document-level leakage prevention needs at least 3 documents).", len(docs)
    )
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n < 10:
        return {"train": shuffled, "val": [], "test": []}

    n_test = max(1, round(n * config.test_ratio))
    n_val = max(1, round(n * config.val_ratio))
    n_train = n - n_val - n_test
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }
