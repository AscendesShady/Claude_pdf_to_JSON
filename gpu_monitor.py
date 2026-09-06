"""Lightweight nvidia-smi polling, used for the GUI's live VRAM readout."""
from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

# Without this, every poll flashes a console window on Windows - unusable once per second.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def read_vram() -> tuple[float, float] | None:
    """Returns (used_mb, total_mb) summed across GPUs, or None if nvidia-smi isn't available."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, creationflags=_NO_WINDOW,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    if proc.returncode != 0:
        return None

    used_total = 0.0
    capacity_total = 0.0
    for row in proc.stdout.strip().splitlines():
        parts = row.split(",")
        if len(parts) != 2:
            continue
        try:
            used_total += float(parts[0].strip())
            capacity_total += float(parts[1].strip())
        except ValueError:
            continue

    if capacity_total <= 0:
        return None
    return used_total, capacity_total
