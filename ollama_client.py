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


def _call(config: PipelineConfig, prompt: str, system: str) -> str:
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
    return resp.json().get("response", "")


def generate_json(config: PipelineConfig, prompt: str, system: str) -> dict | None:
    """Call Ollama expecting a JSON object back; retries once with a correction nudge."""
    last_raw = ""
    for attempt in range(2):
        try:
            last_raw = _call(config, prompt, system)
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
            return parsed
        logger.debug("Model returned valid JSON but not an object (attempt %d/2): %r", attempt + 1, last_raw[:200])

    logger.warning("Giving up on this generation after 2 failed attempts")
    return None
