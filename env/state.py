"""State helpers for the TrustOps environment."""

from __future__ import annotations

from typing import Any

from config import DATA_CHUNK_SIZE, MAX_HISTORY_WINDOW
from utils.helpers import deep_copy


def build_state(task_type: str, task_description: str, max_steps: int) -> dict[str, Any]:
    """Create a fresh internal environment state dictionary."""
    return {
        "task_type": task_type,
        "task_description": task_description,
        "history": [],
        "trajectory": [],
        "integrity_flags": [],
        "steps_remaining": max_steps,
        "done": False,
        "chunk_index": 0,
    }


def build_public_state(
    state: dict[str, Any],
    data_chunk: list[dict[str, Any]],
    available_actions: list[str],
) -> dict[str, Any]:
    """Return sanitized observation only."""
    return {
        "task_description": state["task_description"],
        "data_chunk": deep_copy(data_chunk),
        "history": deep_copy(state["history"][-MAX_HISTORY_WINDOW:]),
        "available_actions": deep_copy(available_actions),
        "steps_remaining": state["steps_remaining"],
    }


def next_chunk_index(current_index: int, total_rows: int, chunk_size: int = DATA_CHUNK_SIZE) -> int:
    """Advance chunk index with wrap-around."""
    if total_rows <= 0 or chunk_size <= 0:
        return 0
    total_chunks = max((total_rows + chunk_size - 1) // chunk_size, 1)
    return (current_index + 1) % total_chunks


def snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Return an immutable-like snapshot copy."""
    return deep_copy(state)
