"""Structured logging for actions, cheating signals, and rewards."""

from __future__ import annotations

from typing import Any


class TrustOpsLogger:
    """In-memory logger for a single environment run."""

    def __init__(self) -> None:
        self.action_logs: list[dict[str, Any]] = []
        self.cheat_logs: list[dict[str, Any]] = []
        self.reward_logs: list[dict[str, Any]] = []
        self.trajectory_summaries: list[dict[str, Any]] = []

    def reset_episode(self) -> None:
        """Clear episode-specific records."""
        self.action_logs.clear()
        self.cheat_logs.clear()
        self.reward_logs.clear()
        self.trajectory_summaries.clear()

    def log_action(self, step_index: int, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Record an agent action."""
        entry = {
            "step": step_index,
            "action": action,
            "payload": payload,
        }
        self.action_logs.append(entry)
        return entry

    def log_cheat(self, step_index: int, cheat_type: str, details: str) -> dict[str, Any]:
        """Record a cheating-related event."""
        entry = {
            "step": step_index,
            "type": cheat_type,
            "details": details,
        }
        self.cheat_logs.append(entry)
        return entry

    def log_reward(self, step_index: int, reward: float, details: dict[str, Any]) -> dict[str, Any]:
        """Record final reward details."""
        entry = {
            "step": step_index,
            "reward": reward,
            "details": details,
        }
        self.reward_logs.append(entry)
        return entry

    def log_trajectory_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Record one trajectory summary snapshot."""
        self.trajectory_summaries.append(summary)
        return summary

    def export(self) -> dict[str, list[dict[str, Any]]]:
        """Export all logs as a serializable structure."""
        return {
            "actions": list(self.action_logs),
            "cheating": list(self.cheat_logs),
            "rewards": list(self.reward_logs),
            "trajectory_summaries": list(self.trajectory_summaries),
        }
