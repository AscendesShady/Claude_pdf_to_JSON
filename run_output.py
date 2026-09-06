"""Per-run output folders, so a second run never overwrites the first run's dataset.

Each run writes into <output_root>/<timestamp>_<label>/ where the label is derived from
the queued PDFs. Keeping one run's files together also matters for correctness: the
train/val/test split is only coherent within a single run, so mixing splits across runs
would break the document-level leakage guarantee the splitter enforces.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from config import PipelineConfig

logger = logging.getLogger(__name__)

_INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_LABEL_CHARS = 40


def _sanitize(name: str) -> str:
    cleaned = _INVALID_CHARS_RE.sub("_", name)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("._")
    return cleaned[:_MAX_LABEL_CHARS] or "run"


def build_run_label(pdf_paths: list[str]) -> str:
    """First book's name, plus a count when the run covers several PDFs."""
    stems = [Path(p).stem for p in pdf_paths if str(p).strip()]
    if not stems:
        return "run"
    label = _sanitize(stems[0])
    if len(stems) > 1:
        label = f"{label}_plus{len(stems) - 1}"
    return label


def create_run_dir(output_root: str, pdf_paths: list[str]) -> Path:
    """Create (and return) a fresh, non-colliding folder for this run's outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(output_root) / f"{timestamp}_{build_run_label(pdf_paths)}"

    # Same-second collisions are near-impossible, but never silently reuse a folder -
    # not overwriting previous results is the entire point of this module.
    candidate = base
    counter = 2
    while candidate.exists():
        candidate = base.parent / f"{base.name}_{counter}"
        counter += 1

    candidate.mkdir(parents=True)
    logger.info("Writing this run's output to %s", candidate)
    return candidate


def write_run_info(
    run_dir: Path, config: PipelineConfig, pdf_paths: list[str], results: dict
) -> Path:
    """Record the settings and totals behind this dataset, so it stays explainable later."""
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_pdfs": [Path(p).name for p in pdf_paths],
        "config": asdict(config),
        "results": results,
    }
    path = run_dir / "run_info.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote run metadata to %s", path.name)
    return path
