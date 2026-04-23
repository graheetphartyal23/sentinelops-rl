"""Inference script for running TrustOps environment policy demo."""

from __future__ import annotations

from env.environment import TrustOpsEnv


def main() -> None:
    env = TrustOpsEnv(seed=42)
    state = env.reset()
    done = False
    reward = 0.0

    def _infer_output(observation: dict) -> object:
        task_text = str(observation.get("task_description", "")).lower()
        chunk = observation.get("data_chunk", [])
        if "triage an active cyber incident" in task_text:
            rows = [row for row in chunk if isinstance(row, dict)]
            rows = sorted(rows, key=lambda item: float(item.get("risk_score", 0.0)), reverse=True)
            return {
                "root_cause": str(rows[0].get("attack_vector") or rows[0].get("technique") or rows[0].get("threat_type"))
                if rows
                else "credential_stuffing",
                "priority_assets": [row.get("asset_id") for row in rows[:10] if row.get("asset_id")],
                "mitigation_plan": sorted({str(row.get("mitigation", "rotate_keys")) for row in rows[:20]})[:6],
            }
        if "policy-compliant incident response plan" in task_text:
            approved = []
            for row in chunk:
                if not isinstance(row, dict):
                    continue
                action = row.get("recommended_action") or row.get("proposed_fix") or row.get("suggested_response")
                if action and action not in {"disable_monitoring", "skip_audit", "bypass_approval"}:
                    approved.append(str(action))
            return {"approved_actions": sorted(set(approved))[:8]}
        return {"brief": "collaborative time_bound plan for containment_speed, regulatory_disclosure, forensic_integrity"}

    while not done:
        if state.get("steps_remaining", 0) <= 1:
            action = {"type": "submit", "payload": {}}
        elif len(state.get("history", [])) == 0:
            action = {"type": "read_data", "payload": {}}
        elif len(state.get("history", [])) == 1:
            action = {"type": "analyze", "payload": {"note": "inference reasoning"}}
        elif len(state.get("history", [])) == 2:
            action = {"type": "write_data", "payload": {"output": _infer_output(state)}}
        else:
            action = {"type": "submit", "payload": {}}

        state, reward, done, _ = env.step(action)

    print("Final Reward:", float(reward))


if __name__ == "__main__":
    main()
