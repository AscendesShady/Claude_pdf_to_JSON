"""Central configuration for the PDF -> QLoRA JSONL pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # --- Ollama / generation ---
    ollama_host: str = "http://localhost:11434"
    model_name: str = "gemma4:12b"
    temperature: float = 0.2
    max_new_tokens: int = 600
    request_timeout_s: int = 240
    variants_per_chunk: int = 1  # extra paraphrased-question variants beyond the primary Q&A

    # --- Chunking ---
    target_chunk_tokens: int = 600
    chunk_overlap_pct: float = 0.15
    min_chunk_tokens: int = 40
    max_chunk_tokens: int = 1200  # oversized chunks beyond this are discarded, never truncated
    hard_cut_pattern: str = r"\n\s*\n\s*\n"  # triple-newline style hard section breaks

    # --- Extraction ---
    min_page_text_chars: int = 20  # below this, a page is treated as scanned/image-only and skipped
    boilerplate_section_titles: tuple[str, ...] = (
        "references", "bibliography", "index", "acknowledgments",
        "acknowledgements", "about the author", "table of contents",
    )

    # --- Quality gate ---
    max_seq_tokens: int = 2048  # records whose rendered prompt+answer exceed this are discarded
    length_ratio_limit: float = 3.0  # discard if output/input token ratio deviates beyond this
    min_answer_tokens: int = 12  # below this an answer is a fragment, not a usable response
    # Strip/reject questions that refer to "the context/passage/document" - at inference time
    # there is no passage, so training on them teaches the model to expect one.
    strip_meta_questions: bool = True

    # --- Split ---
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42

    # --- Output ---
    output_dir: str = "output"

    def chunk_overlap_tokens(self) -> int:
        return int(self.target_chunk_tokens * self.chunk_overlap_pct)


# Shown in the GUI's model dropdown only when Ollama can't be reached (so the box isn't
# empty on first paint). The dropdown is always repopulated from a live `ollama list` /
# `/api/tags` call as soon as Ollama responds - this list has no bearing on what actually runs.
AVAILABLE_MODELS_FALLBACK = [
    "gemma4:12b",
    "gemma4:latest",
    "qwen3.8:latest",
    "qwen3.6:latest",
    "llama3.1:8b",
    "llama3:latest",
    "deepseek-r1:32b",
    "nemotron-3-nano:4b",
]
