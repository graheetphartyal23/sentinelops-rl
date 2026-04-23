"""General helper utilities used across modules."""

from __future__ import annotations

import copy
from typing import Any


def deep_copy(value: Any) -> Any:
    """Return a deep copy of a value."""
    return copy.deepcopy(value)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a float into a provided inclusive range."""
    return max(min_value, min(max_value, value))


def normalize_record_list(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create stable ordering for row-comparison in task evaluation."""
    return sorted(records, key=lambda item: (item.get("id"), str(item)))


def chunk_rows(rows: list[dict[str, Any]], chunk_index: int, chunk_size: int) -> list[dict[str, Any]]:
    """Return a stable data window for partial observability."""
    if chunk_size <= 0:
        return []
    start = max(chunk_index, 0) * chunk_size
    end = start + chunk_size
    return deep_copy(rows[start:end])


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert unknown value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
