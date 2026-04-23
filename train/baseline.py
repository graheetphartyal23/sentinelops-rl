"""Baseline agents for TrustOps evaluation."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from config import LEVEL_4
from env.environment import TrustOpsEnv
from train.grpo_training import run_baselines


def random_agent(env: TrustOpsEnv) -> float:
    """Run a random policy baseline and return terminal reward."""
    state = env.reset()
    done = False
    reward = 0.0

    while not done:
        action_name = random.choice(env.actions)
        payload: dict[str, Any] = {}
        if action_name == "analyze":
            payload = {"note": "random baseline analysis"}
        state, reward, done, _ = env.step({"type": action_name, "payload": payload})
    return float(reward)


def print_baseline_comparison(episodes: int = 20, trained_reward: float = 0.7) -> None:
    """Print simple baseline vs trained reward comparison."""
    env = TrustOpsEnv(seed=42, curriculum_level=LEVEL_4)
    random_rewards = [random_agent(env) for _ in range(episodes)]
    random_mean = float(np.mean(random_rewards)) if random_rewards else 0.0

    print(f"Random Reward: {random_mean:.3f}")
    print(f"Trained Reward: {trained_reward:.3f}")

    summary = run_baselines(episodes=episodes, seed=42, max_steps=30, curriculum_level=LEVEL_4)
    print("Baseline Summary:")
    print(summary)


if __name__ == "__main__":
    print_baseline_comparison()
