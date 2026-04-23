"""Phase 5 metrics and graph generation for final evaluation package."""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from config import LEVEL_4
from train.grpo_training import run_baseline_curves, run_baselines


def smooth(data: list[float], window: int = 10) -> list[float]:
    """Compute moving average smoothing."""
    smoothed: list[float] = []
    for idx in range(len(data)):
        start = max(0, idx - window)
        segment = data[start : idx + 1]
        smoothed.append(float(sum(segment) / max(len(segment), 1)))
    return smoothed


def _plot_single_metric(
    episodes: list[int],
    grpo_values: list[float],
    baseline_values: list[float],
    baseline_label: str,
    ylabel: str,
    title: str,
    output_path: str,
    smoothing_window: int,
) -> None:
    """Save one chart with raw+smoothed series for GRPO and baseline."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, grpo_values, alpha=0.25, color="tab:blue", label="GRPO (raw)")
    plt.plot(episodes, smooth(grpo_values, window=smoothing_window), color="tab:blue", linewidth=2.2, label="GRPO (smooth)")
    plt.plot(episodes, baseline_values, alpha=0.25, color="tab:orange", label=f"{baseline_label} (raw)")
    plt.plot(
        episodes,
        smooth(baseline_values, window=smoothing_window),
        color="tab:orange",
        linewidth=2.2,
        linestyle="--",
        label=f"{baseline_label} (smooth)",
    )
    plt.xlabel("Episodes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_cheat_breakdown(
    episodes: list[int], api: list[float], hint: list[float], bypass: list[float], output_path: str, smoothing_window: int
) -> None:
    """Plot cheating mechanism breakdown over training."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, api, alpha=0.2, color="firebrick", label="API usage raw")
    plt.plot(episodes, smooth(api, window=smoothing_window), color="firebrick", linewidth=2.2, label="API usage smooth")
    plt.plot(episodes, hint, alpha=0.2, color="darkorange", label="Hint decoding raw")
    plt.plot(
        episodes, smooth(hint, window=smoothing_window), color="darkorange", linewidth=2.2, label="Hint decoding smooth"
    )
    plt.plot(episodes, bypass, alpha=0.2, color="slateblue", label="Bypass raw")
    plt.plot(
        episodes, smooth(bypass, window=smoothing_window), color="slateblue", linewidth=2.2, label="Bypass smooth"
    )
    plt.xlabel("Episodes")
    plt.ylabel("Cheat Breakdown Rate")
    plt.title("Cheating Breakdown Over Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _write_sample_trajectories(metrics: dict[str, Any], output_path: str) -> None:
    """Write before/after trajectory comparison text artifact."""
    best_path = str(metrics.get("best_path", [""])[0]) if len(metrics.get("best_path", [])) > 0 else ""
    worst_path = str(metrics.get("worst_path", [""])[0]) if len(metrics.get("worst_path", [])) > 0 else ""
    best_reward = float(metrics.get("best_reward_seen", [0.0])[0]) if len(metrics.get("best_reward_seen", [])) > 0 else 0.0
    worst_reward = (
        float(metrics.get("worst_reward_seen", [0.0])[0]) if len(metrics.get("worst_reward_seen", [])) > 0 else 0.0
    )

    lines = [
        "Trajectory Comparison (Demo Artifact)",
        "",
        f"BEFORE Training-like behavior (low reward={worst_reward:.4f})",
        f"{worst_path} ❌",
        "",
        f"AFTER Training-like behavior (high reward={best_reward:.4f})",
        f"{best_path} ✅",
        "",
        "Narrative:",
        "Initially, trajectories often exploit shortcuts.",
        "After optimization, trajectories favor structured multi-step workflows.",
    ]
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_baseline_table(
    baseline_summary: dict[str, dict[str, float]], metrics: dict[str, Any], output_path: str
) -> None:
    """Write baseline vs GRPO table artifact."""
    grpo_reward = float(np.mean(metrics["rewards"]))
    grpo_cheat = float(np.mean(metrics["cheat_rates"]))
    grpo_steps = float(np.mean(metrics["avg_steps"]))

    rows = [
        "Model,Reward,Cheating,AvgSteps",
        f"Random,{baseline_summary['random']['reward']:.4f},{baseline_summary['random']['cheat_rate']:.4f},{baseline_summary['random']['avg_steps']:.2f}",
        f"Heuristic,{baseline_summary['heuristic']['reward']:.4f},{baseline_summary['heuristic']['cheat_rate']:.4f},{baseline_summary['heuristic']['avg_steps']:.2f}",
        f"GRPO,{grpo_reward:.4f},{grpo_cheat:.4f},{grpo_steps:.2f}",
    ]
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(rows))


def parse_args() -> argparse.Namespace:
    """Parse CLI args for Phase 5 report generation."""
    parser = argparse.ArgumentParser(description="Generate Phase 5 metrics package.")
    parser.add_argument("--metrics", default="train/grpo_metrics.npz")
    parser.add_argument("--output-dir", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum-level", type=int, default=LEVEL_4)
    parser.add_argument("--smoothing-window", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = np.load(args.metrics, allow_pickle=True)

    rewards = metrics["rewards"].astype(float).tolist()
    cheat_rates = metrics["cheat_rates"].astype(float).tolist()
    workflow_scores = metrics["workflow_scores"].astype(float).tolist()
    avg_steps = metrics["avg_steps"].astype(float).tolist()
    api_cheat_breakdown = metrics["api_cheat_breakdown"].astype(float).tolist()
    hint_cheat_breakdown = metrics["hint_cheat_breakdown"].astype(float).tolist()
    bypass_cheat_breakdown = metrics["bypass_cheat_breakdown"].astype(float).tolist()
    episodes = list(range(len(rewards)))

    baseline_curves = run_baseline_curves(
        episodes=len(rewards),
        seed=args.seed,
        max_steps=30,
        curriculum_level=args.curriculum_level,
    )
    baseline_summary = run_baselines(
        episodes=max(30, min(len(rewards), 60)),
        seed=args.seed,
        max_steps=30,
        curriculum_level=args.curriculum_level,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    reward_curve_path = os.path.join(args.output_dir, "reward_curve.png")
    cheating_curve_path = os.path.join(args.output_dir, "cheating_curve.png")
    workflow_curve_path = os.path.join(args.output_dir, "workflow_curve.png")
    steps_curve_path = os.path.join(args.output_dir, "steps_curve.png")
    cheat_breakdown_path = os.path.join(args.output_dir, "cheat_breakdown.png")
    trajectories_path = os.path.join(args.output_dir, "sample_trajectories.txt")
    baseline_table_path = os.path.join(args.output_dir, "baseline_vs_trained.csv")

    _plot_single_metric(
        episodes=episodes,
        grpo_values=rewards,
        baseline_values=baseline_curves["random_rewards"],
        baseline_label="Random",
        ylabel="Average Reward",
        title="Reward Improvement Over Training",
        output_path=reward_curve_path,
        smoothing_window=args.smoothing_window,
    )
    _plot_single_metric(
        episodes=episodes,
        grpo_values=cheat_rates,
        baseline_values=baseline_curves["random_cheat_rates"],
        baseline_label="Random",
        ylabel="Cheating Rate",
        title="Cheating Rate Over Training",
        output_path=cheating_curve_path,
        smoothing_window=args.smoothing_window,
    )
    _plot_single_metric(
        episodes=episodes,
        grpo_values=workflow_scores,
        baseline_values=baseline_curves["random_workflow_scores"],
        baseline_label="Random",
        ylabel="Workflow Score",
        title="Workflow Quality Over Training",
        output_path=workflow_curve_path,
        smoothing_window=args.smoothing_window,
    )
    _plot_single_metric(
        episodes=episodes,
        grpo_values=avg_steps,
        baseline_values=baseline_curves["random_steps"],
        baseline_label="Random",
        ylabel="Average Steps",
        title="Average Steps Per Episode",
        output_path=steps_curve_path,
        smoothing_window=args.smoothing_window,
    )
    _plot_cheat_breakdown(
        episodes=episodes,
        api=api_cheat_breakdown,
        hint=hint_cheat_breakdown,
        bypass=bypass_cheat_breakdown,
        output_path=cheat_breakdown_path,
        smoothing_window=args.smoothing_window,
    )
    _write_sample_trajectories(metrics=dict(metrics), output_path=trajectories_path)
    _write_baseline_table(baseline_summary=baseline_summary, metrics=dict(metrics), output_path=baseline_table_path)

    print("Phase 5 package generated:")
    print(f"- {reward_curve_path}")
    print(f"- {cheating_curve_path}")
    print(f"- {workflow_curve_path}")
    print(f"- {steps_curve_path}")
    print(f"- {cheat_breakdown_path}")
    print(f"- {trajectories_path}")
    print(f"- {baseline_table_path}")


if __name__ == "__main__":
    main()
