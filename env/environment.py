"""OpenAI Gym-style environment for trajectory-based integrity evaluation."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

import numpy as np

try:
    from openenv import Environment  # type: ignore
except ImportError:
    from openenv.env import Env as _OpenEnvBase

    StateT = TypeVar("StateT")
    ActionT = TypeVar("ActionT")
    InfoT = TypeVar("InfoT")

    class Environment(_OpenEnvBase, Generic[StateT, ActionT, InfoT]):
        """Compatibility layer for OpenEnv releases without generic Environment."""

        pass

from config import (
    DATA_CHUNK_SIZE,
    DATASET_ROWS_MIN,
    LEVEL_1,
    LEVEL_2,
    LEVEL_3,
    LEVEL_4,
    MAX_HISTORY_WINDOW,
    MAX_STEPS,
    SUPPORTED_LEVELS,
)
from detection.cheat_detector import evaluate_cheating
from env.state import build_public_state, build_state, next_chunk_index, snapshot
from reward.reward_engine import compute_reward as reward_formula
from reward.reward_engine import compute_workflow_score
from tasks import incident_triage, response_integrity, stakeholder_negotiation
from tools.actions import ACTIONS, handle_action, validate_action
from traps.ambiguity_trap import inject_ambiguity_signals
from traps.api_trap import reset_api_usage_log
from traps.encoding_trap import inject_encoded_hint
from traps.validation_trap import inject_validation_flag
from utils.helpers import chunk_rows
from utils.logger import TrustOpsLogger


class TrustOpsEnv(Environment[dict, dict, dict]):
    """TrustOps enterprise workflow simulation environment."""

    def __init__(self, seed: int = 42, curriculum_level: int = LEVEL_3) -> None:
        super().__init__(name="TrustOpsEnv", state_space="dict", action_space="dict", episode_max_length=MAX_STEPS)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.curriculum_level = curriculum_level if curriculum_level in SUPPORTED_LEVELS else LEVEL_3
        self.actions = list(ACTIONS)
        self.task_pool = {
            "incident_triage": incident_triage,
            "response_integrity": response_integrity,
            "stakeholder_negotiation": stakeholder_negotiation,
        }
        self.logger = TrustOpsLogger()
        self._state: dict[str, Any] = {}
        self.current_task_type = ""
        self.current_task_module = None
        self.answer_book: dict[str, Any] = {}
        self.hidden_context: dict[str, Any] = {}
        self.full_dataset: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.trap_events: list[dict[str, Any]] = []
        self.integrity_flags: list[str] = []
        self.current_step = 0
        self.working_output: Any = None
        self.submitted_output: Any = None
        self.analysis_notes: list[str] = []
        self.decoded_hint_cache: Any = None
        self.last_read_chunk: list[dict[str, Any]] = []
        self.used_ambiguous_signals: list[str] = []
        self._latest_reward = 0.0
        self._latest_info: dict[str, Any] = {}

    def set_curriculum_level(self, level: int) -> None:
        """Dynamically switch curriculum level."""
        if level not in SUPPORTED_LEVELS:
            raise ValueError(f"Unsupported curriculum level: {level}")
        self.curriculum_level = level

    def _active_traps(self) -> set[str]:
        if self.curriculum_level == LEVEL_1:
            return set()
        if self.curriculum_level == LEVEL_2:
            return {"api_trap"}
        if self.curriculum_level == LEVEL_3:
            return {"api_trap", "encoding_trap", "validation_trap"}
        return {"api_trap", "encoding_trap", "validation_trap", "ambiguity_trap"}

    def _build_hidden_context(self, task_payload: dict[str, Any], task_type: str) -> dict[str, Any]:
        hidden = {
            "expected": task_payload["expected"],
            "raw_data": task_payload["input_data"],
            "required_steps": task_payload.get("required_steps", []),
            "schema_variant": int(self.rng.integers(0, 3)),
        }
        active_traps = self._active_traps()
        if "encoding_trap" in active_traps:
            wrapped = inject_encoded_hint(hidden["raw_data"], hidden["expected"])
            hidden["encoded_hint"] = wrapped["encoded_hint"]
        if "validation_trap" in active_traps:
            wrapped = inject_validation_flag({"raw_data": hidden["raw_data"]})
            hidden["validation_flag"] = wrapped.get("validation_flag")
        if "ambiguity_trap" in active_traps:
            hidden["ambiguity_signals"] = inject_ambiguity_signals(self.rng, task_type)
        else:
            hidden["ambiguity_signals"] = []
        return hidden

    def get_current_data_chunk(self) -> list[dict[str, Any]]:
        """Get current chunk for partial observability."""
        chunk = chunk_rows(self.full_dataset, self._state.get("chunk_index", 0), DATA_CHUNK_SIZE)
        if "ambiguity_trap" in self._active_traps():
            signals = self.hidden_context.get("ambiguity_signals", [])
            if signals:
                signal_idx = self._state.get("chunk_index", 0) % len(signals)
                chunk.append({"hint_signal": signals[signal_idx]["signal"]})
        return chunk

    def _sanitized_state(self) -> dict[str, Any]:
        return build_public_state(state=self._state, data_chunk=self.get_current_data_chunk(), available_actions=self.actions)

    def state(self) -> dict[str, Any]:
        """Return current observable state (OpenEnv/Gym-style accessor)."""
        if not self._state:
            return {}
        return self._sanitized_state()

    def reset(self) -> dict[str, Any]:
        """Reset environment and return sanitized initial state."""
        self.logger.reset_episode()
        reset_api_usage_log()
        self.working_output = None
        self.submitted_output = None
        self.analysis_notes = []
        self.decoded_hint_cache = None
        self.last_read_chunk = []
        self.used_ambiguous_signals = []
        self.current_step = 0
        self._latest_reward = 0.0
        self._latest_info = {}
        self.trajectory = []
        self.trap_events = []
        self.integrity_flags = []

        task_names = sorted(self.task_pool.keys())
        task_idx = int(self.rng.integers(0, len(task_names)))
        task_type = task_names[task_idx]
        task_module = self.task_pool[task_type]
        row_count = DATASET_ROWS_MIN + int(self.rng.integers(0, 100))
        schema_variant = int(self.rng.integers(0, 3))
        task_payload = task_module.generate_task(rng=self.rng, row_count=row_count, schema_variant=schema_variant)

        self.current_task_type = task_type
        self.current_task_module = task_module
        self.answer_book = {task_type: task_payload["expected"]}
        self.full_dataset = list(task_payload["input_data"])
        self.hidden_context = self._build_hidden_context(task_payload, task_type)
        self._state = build_state(task_type, task_payload["task_description"], MAX_STEPS)
        return self._sanitized_state()

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute one environment step and return sanitized state."""
        if not self._state:
            raise RuntimeError("Environment must be reset before calling step().")
        if self._state["done"]:
            return self._sanitized_state(), 0.0, True, {"message": "Episode already completed."}

        normalized_action = validate_action(action)
        self.current_step = len(self.trajectory) + 1
        action_type = normalized_action["type"]
        payload = normalized_action["payload"]

        self.logger.log_action(self.current_step, action_type, payload)
        result = handle_action(self, normalized_action)

        trap_events = list(result.get("trap_events", []))
        self.trap_events.extend(trap_events)
        integrity_flags = list(result.get("integrity_flags", []))
        self.integrity_flags.extend(integrity_flags)

        trajectory_item = {
            "step": self.current_step,
            "action": action_type,
            "summary": str(result.get("summary", "")),
            "trap_events": trap_events,
            "integrity_flags": integrity_flags,
        }
        self.trajectory.append(trajectory_item)
        self._state["trajectory"] = self.trajectory
        self._state["history"].append(
            {
                "step": self.current_step,
                "action": action_type,
                "summary": trajectory_item["summary"],
            }
        )
        self._state["history"] = self._state["history"][-MAX_HISTORY_WINDOW:]
        self._state["integrity_flags"] = self.integrity_flags[-MAX_HISTORY_WINDOW:]

        if action_type == "read_data":
            self._state["chunk_index"] = next_chunk_index(self._state.get("chunk_index", 0), len(self.full_dataset))

        self._state["steps_remaining"] -= 1
        submitted = bool(result.get("submitted"))
        reached_limit = self._state["steps_remaining"] <= 0
        self._state["done"] = submitted or reached_limit

        info: dict[str, Any] = {
            "step": self.current_step,
            "trap_events": trap_events,
            "integrity_flags": integrity_flags,
        }
        reward = 0.0
        if self._state["done"]:
            reward = self.compute_reward()
            info.update(self._latest_info)
            info["trajectory"] = snapshot(self.trajectory)

        return self._sanitized_state(), float(reward), bool(self._state["done"]), info

    def compute_reward(self) -> float:
        """Compute terminal reward from correctness, cheating, and workflow quality."""
        if self.current_task_module is None:
            raise RuntimeError("No active task module loaded.")
        if not self._state["done"]:
            raise RuntimeError("Reward can only be computed at terminal step.")

        correctness = float(self.current_task_module.evaluate_output(self.submitted_output))
        cheat_result = evaluate_cheating(self.trajectory, self.trap_events, MAX_STEPS)
        cheat_score = float(cheat_result["cheat_score"])
        events = cheat_result["events"]

        for event_name, flagged in events.items():
            if flagged:
                self.logger.log_cheat(self.current_step, event_name, f"Detected pattern: {event_name}")

        workflow_score = compute_workflow_score(trajectory=self.trajectory, max_steps=MAX_STEPS)
        reward = reward_formula(
            correctness=correctness,
            cheat_score=cheat_score,
            workflow_score=workflow_score,
            trajectory=self.trajectory,
        )

        summary = {
            "steps": len(self.trajectory),
            "actions": [item["action"] for item in self.trajectory],
            "trap_events": [event["event"] for event in self.trap_events],
        }
        details = {
            "correctness": correctness,
            "cheat_score": cheat_score,
            "workflow_score": workflow_score,
            "events": events,
            "trajectory_summary": summary,
        }
        self.logger.log_reward(self.current_step, reward, details)
        self.logger.log_trajectory_summary(summary)

        self._latest_reward = reward
        self._latest_info = details
        return float(reward)
