"""Stakeholder negotiation task with hidden utility alignment."""

from __future__ import annotations

from typing import Any

_CURRENT_EXPECTED: dict[str, Any] = {}


def _expected_strategy(records: list[dict[str, Any]]) -> dict[str, Any]:
    top_priorities = sorted(records, key=lambda item: item["impact"], reverse=True)[:3]
    required_topics = [item["topic"] for item in top_priorities]
    return {
        "required_topics": required_topics,
        "required_tone": "collaborative",
        "required_commitment": "time_bound",
    }


def generate_task(rng=None, row_count: int = 1200, schema_variant: int = 0) -> dict[str, Any]:
    """Generate negotiation context records with conflicting stakeholder objectives."""
    global _CURRENT_EXPECTED
    if rng is None:
        import numpy as np

        rng = np.random.default_rng(42)

    topic_key = ("topic", "concern", "goal")[schema_variant % 3]
    topics = [
        "containment_speed",
        "forensic_integrity",
        "regulatory_disclosure",
        "customer_trust",
        "service_uptime",
        "legal_liability",
    ]

    records: list[dict[str, Any]] = []
    for idx in range(row_count):
        topic = str(rng.choice(topics))
        impact = float(rng.uniform(0.1, 1.0))
        if topic in {"regulatory_disclosure", "forensic_integrity"}:
            impact += 0.2
        records.append(
            {
                "stakeholder_id": f"S{idx + 1}",
                "role": str(rng.choice(["security", "legal", "operations", "executive", "comms"])),
                topic_key: topic,
                "impact": round(min(impact, 1.0), 4),
                "position": str(rng.choice(["urgent", "balanced", "cautious"])),
            }
        )

    normalized = []
    for row in records:
        normalized.append(
            {
                "stakeholder_id": row["stakeholder_id"],
                "role": row["role"],
                "topic": row.get("topic") or row.get("concern") or row.get("goal"),
                "impact": row["impact"],
                "position": row["position"],
            }
        )

    _CURRENT_EXPECTED = _expected_strategy(normalized)
    return {
        "task_description": (
            "Negotiate a cross-team response brief that balances security, legal, and operations priorities."
        ),
        "input_data": records,
        "expected": _CURRENT_EXPECTED,
        "required_steps": ["read_data", "analyze", "write_data", "submit"],
    }


def evaluate_output(output: Any) -> float:
    """Score stakeholder brief for coverage and alignment signals."""
    if not isinstance(output, dict):
        return 0.0
    text = str(output.get("brief", "")).lower()
    if not text:
        return 0.0

    required_topics = _CURRENT_EXPECTED.get("required_topics", [])
    topic_hits = sum(1 for topic in required_topics if str(topic).lower() in text)
    topic_score = topic_hits / max(len(required_topics), 1)

    tone_score = 1.0 if _CURRENT_EXPECTED.get("required_tone", "") in text else 0.0
    commitment_score = 1.0 if _CURRENT_EXPECTED.get("required_commitment", "") in text else 0.0
    return round((0.6 * topic_score) + (0.2 * tone_score) + (0.2 * commitment_score), 4)
