"""Thin REST wrapper around a local Ollama server for structured JSON generation."""
from __future__ import annotations

import json
import logging

import requests

from config import PipelineConfig

logger = logging.getLogger(__name__)


def is_reachable(host: str, timeout: float = 3.0) -> bool:
    """Cheap check for whether an Ollama server is up at all, before trying anything else."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=timeout)
        return resp.ok
    except requests.RequestException:
        return False


def list_models(host: str) -> list[str]:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except requests.RequestException as exc:
        logger.warning("Could not list Ollama models at %s: %s", host, exc)
        return []


def _call(config: PipelineConfig, prompt: str, system: str) -> tuple[str, int]:
    """Returns (response_text, tokens_consumed) - Ollama reports prompt/completion counts."""
    payload = {
        "model": config.model_name,
        "prompt": prompt,
        "system": system,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_new_tokens,
        },
    }
    resp = requests.post(
        f"{config.ollama_host}/api/generate", json=payload, timeout=config.request_timeout_s
    )
    resp.raise_for_status()
    data = resp.json()
    tokens = (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
    return data.get("response", ""), tokens


def generate_json(config: PipelineConfig, prompt: str, system: str) -> tuple[dict | None, int]:
    """Call Ollama expecting a JSON object back; retries once with a correction nudge.

    Returns (parsed_object_or_None, tokens_consumed). Tokens accumulate across retries,
    since a failed attempt still costs real inference.
    """
    last_raw = ""
    tokens_used = 0
    for attempt in range(2):
        try:
            last_raw, tokens = _call(config, prompt, system)
            tokens_used += tokens
        except requests.RequestException as exc:
            logger.warning("Ollama request failed (attempt %d/2): %s", attempt + 1, exc)
            continue

        try:
            parsed = json.loads(last_raw)
        except json.JSONDecodeError:
            logger.debug("Invalid JSON from model (attempt %d/2): %r", attempt + 1, last_raw[:200])
            prompt = (
                prompt
                + "\n\nYour previous response was not valid JSON. "
                "Respond with ONLY the JSON object and nothing else."
            )
            continue

        if isinstance(parsed, dict):
            return parsed, tokens_used
        logger.debug("Model returned valid JSON but not an object (attempt %d/2): %r", attempt + 1, last_raw[:200])

    logger.warning("Giving up on this generation after 2 failed attempts")
    return None, tokens_used
