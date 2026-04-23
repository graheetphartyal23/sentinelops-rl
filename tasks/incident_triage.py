"""Incident triage task with large noisy telemetry slices."""

from __future__ import annotations

from typing import Any

from utils.helpers import safe_float

_CURRENT_EXPECTED: dict[str, Any] = {}


def _derive_expected(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, float] = {}
    for row in records:
        attack = str(row.get("attack_vector", "unknown"))
        grouped[attack] = grouped.get(attack, 0.0) + safe_float(row.get("risk_score"), 0.0)

    dominant_vector = max(grouped.items(), key=lambda kv: kv[1])[0] if grouped else "credential_stuffing"
    prioritized_assets = [
        row["asset_id"]
        for row in sorted(records, key=lambda r: safe_float(r.get("risk_score"), 0.0), reverse=True)[:10]
    ]
    patch_actions = sorted(
        {
            str(row.get("mitigation", "rotate_keys"))
            for row in records
            if str(row.get("severity", "")).lower() in {"high", "critical"}
        }
    )[:6]
    return {
        "root_cause": dominant_vector,
        "priority_assets": prioritized_assets,
        "mitigation_plan": patch_actions,
    }


def generate_task(rng=None, row_count: int = 1500, schema_variant: int = 0) -> dict[str, Any]:
    """Generate deterministic security incident telemetry."""
    global _CURRENT_EXPECTED
    if rng is None:
        import numpy as np

        rng = np.random.default_rng(42)

    vector_key = ("attack_vector", "technique", "threat_type")[schema_variant % 3]
    vectors = ["credential_stuffing", "privilege_escalation", "supply_chain", "insider_exfiltration"]
    severities = ["low", "medium", "high", "critical"]
    mitigations = ["rotate_keys", "patch_vuln", "disable_account", "isolate_host", "block_ioc"]

    records: list[dict[str, Any]] = []
    for idx in range(row_count):
        vector = str(rng.choice(vectors, p=[0.35, 0.25, 0.2, 0.2]))
        severity = str(rng.choice(severities, p=[0.2, 0.35, 0.3, 0.15]))
        risk = float(rng.uniform(0.1, 0.95))
        if vector == "credential_stuffing":
            risk += 0.2
        if severity in {"high", "critical"}:
            risk += 0.15
        records.append(
            {
                "event_id": idx + 1,
                "asset_id": f"asset-{int(rng.integers(100, 999))}",
                vector_key: vector,
                "severity": severity,
                "risk_score": round(min(risk, 1.0), 4),
                "mitigation": str(rng.choice(mitigations)),
                "log_excerpt": str(
                    rng.choice(
                        [
                            "Multiple failed MFA challenges",
                            "Unusual privilege grant chain detected",
                            "Package hash mismatch in CI pipeline",
                            "Bulk outbound transfer to unknown domain",
                        ]
                    )
                ),
            }
        )

    normalized = []
    for row in records:
        normalized.append(
            {
                "event_id": row["event_id"],
                "asset_id": row["asset_id"],
                "attack_vector": row.get("attack_vector") or row.get("technique") or row.get("threat_type"),
                "severity": row["severity"],
                "risk_score": row["risk_score"],
                "mitigation": row["mitigation"],
                "log_excerpt": row["log_excerpt"],
            }
        )

    _CURRENT_EXPECTED = _derive_expected(normalized)
    return {
        "task_description": (
            "Triage an active cyber incident. Identify likely root cause, rank affected assets, and propose mitigations."
        ),
        "input_data": records,
        "expected": _CURRENT_EXPECTED,
        "required_steps": ["read_data", "analyze", "write_data", "submit"],
    }


def evaluate_output(output: Any) -> float:
    """Score incident triage output with soft verifiable criteria."""
    if not isinstance(output, dict) or not _CURRENT_EXPECTED:
        return 0.0

    root = str(output.get("root_cause", "")).lower()
    expected_root = str(_CURRENT_EXPECTED.get("root_cause", "")).lower()
    root_score = 1.0 if expected_root in root else 0.0

    expected_assets = set(_CURRENT_EXPECTED.get("priority_assets", []))
    actual_assets = set(output.get("priority_assets", [])) if isinstance(output.get("priority_assets"), list) else set()
    asset_score = len(expected_assets.intersection(actual_assets)) / max(len(expected_assets), 1)

    expected_mit = set(_CURRENT_EXPECTED.get("mitigation_plan", []))
    actual_mit = set(output.get("mitigation_plan", [])) if isinstance(output.get("mitigation_plan"), list) else set()
    mitigation_score = len(expected_mit.intersection(actual_mit)) / max(len(expected_mit), 1)

    return round((0.4 * root_score) + (0.3 * asset_score) + (0.3 * mitigation_score), 4)
