"""HuggingFace Space demo for TrustOps environment."""

from __future__ import annotations

import os
import sys

import gradio as gr

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.environment import TrustOpsEnv


env = TrustOpsEnv(seed=42)


def run_demo() -> str:
    """Run a simple deterministic policy for demo purposes."""
    state = env.reset()
    done = False
    actions_log: list[str] = []
    reward = 0.0

    while not done:
        if state.get("steps_remaining", 0) <= 1:
            action = {"type": "submit", "payload": {}}
        elif len(actions_log) == 0:
            action = {"type": "read_data", "payload": {}}
        elif len(actions_log) == 1:
            action = {"type": "analyze", "payload": {"note": "demo analysis"}}
        elif len(actions_log) == 2:
            action = {"type": "write_data", "payload": {"output": {"brief": "collaborative time_bound response"}}}
        else:
            action = {"type": "submit", "payload": {}}
        state, reward, done, _ = env.step(action)
        actions_log.append(action["type"])

    return f"Trajectory: {actions_log}\nReward: {reward:.4f}"


demo = gr.Interface(fn=run_demo, inputs=[], outputs="text", title="TrustOps Demo")

if __name__ == "__main__":
    demo.launch()
