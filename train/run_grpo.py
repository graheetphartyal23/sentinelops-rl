"""CLI entrypoint for TrustOps GRPO training."""

from __future__ import annotations

import argparse
import os

import numpy as np

from config import LEVEL_4
from train.grpo_training import GRPOConfig, evaluate_untrained_model_baseline, run_baselines, train_grpo


def parse_args() -> argparse.Namespace:
    """Parse command-line options for GRPO training."""
    parser = argparse.ArgumentParser(description="Run GRPO training on TrustOps.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--curriculum-level", type=int, default=LEVEL_4)
    parser.add_argument("--no-curriculum-schedule", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="train/grpo_metrics.npz")
    parser.add_argument("--baseline-episodes", type=int, default=30)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--untrained-episodes", type=int, default=0)
    return parser.parse_args()


def plot_metrics(output_npz: str) -> str:
    """Render reward/cheat/workflow/behavior curves to a PNG file."""
    import matplotlib.pyplot as plt

    metrics = np.load(output_npz)
    rewards = metrics["rewards"]
    cheat_rates = metrics["cheat_rates"]
    workflow_scores = metrics["workflow_scores"]
    avg_steps = metrics["avg_steps"]
    api_usage_frequencies = metrics["api_usage_frequencies"]
    hint_decode_frequencies = metrics["hint_decode_frequencies"]
    api_cheat_breakdown = metrics["api_cheat_breakdown"]
    hint_cheat_breakdown = metrics["hint_cheat_breakdown"]
    bypass_cheat_breakdown = metrics["bypass_cheat_breakdown"]
    curriculum_levels = metrics["curriculum_levels"]

    fig, axes = plt.subplots(5, 1, figsize=(10, 16), sharex=True)
    axes[0].plot(rewards, label="Mean Reward")
    axes[0].set_ylabel("Reward")
    axes[0].legend(loc="best")

    axes[1].plot(cheat_rates, label="Cheat Rate", color="tomato")
    axes[1].set_ylabel("Cheat Rate")
    axes[1].legend(loc="best")

    axes[2].plot(workflow_scores, label="Workflow Score", color="seagreen")
    axes[2].plot(avg_steps, label="Avg Steps", color="royalblue")
    axes[2].set_ylabel("Workflow / Steps")
    axes[2].legend(loc="best")

    axes[3].plot(api_usage_frequencies, label="API Usage Frequency", color="purple")
    axes[3].plot(hint_decode_frequencies, label="Hint Decode Frequency", color="goldenrod")
    axes[3].plot(curriculum_levels, label="Curriculum Level", color="gray", linestyle="--")
    axes[3].set_ylabel("Behavior Metrics")
    axes[3].legend(loc="best")

    axes[4].plot(api_cheat_breakdown, label="API Cheat Breakdown", color="firebrick")
    axes[4].plot(hint_cheat_breakdown, label="Hint Cheat Breakdown", color="darkorange")
    axes[4].plot(bypass_cheat_breakdown, label="Bypass Cheat Breakdown", color="slateblue")
    axes[4].set_ylabel("Cheat Breakdown")
    axes[4].set_xlabel("Episode")
    axes[4].legend(loc="best")

    fig.tight_layout()
    plot_path = os.path.splitext(output_npz)[0] + "_curve.png"
    fig.savefig(plot_path, dpi=150)
    return plot_path


def print_baseline_comparison(
    baselines: dict[str, dict[str, float]],
    grpo_metrics: dict[str, list[float]],
    untrained_baseline: dict[str, float] | None = None,
) -> None:
    """Print baseline vs GRPO comparison table."""
    grpo_row = {
        "reward": float(np.mean(grpo_metrics["rewards"])) if grpo_metrics["rewards"] else 0.0,
        "cheat_rate": float(np.mean(grpo_metrics["cheat_rates"])) if grpo_metrics["cheat_rates"] else 0.0,
        "avg_steps": float(np.mean(grpo_metrics["avg_steps"])) if grpo_metrics["avg_steps"] else 0.0,
        "api_usage_frequency": float(np.mean(grpo_metrics["api_usage_frequencies"]))
        if grpo_metrics["api_usage_frequencies"]
        else 0.0,
        "hint_decode_frequency": float(np.mean(grpo_metrics["hint_decode_frequencies"]))
        if grpo_metrics["hint_decode_frequencies"]
        else 0.0,
        "api_cheat_breakdown": float(np.mean(grpo_metrics["api_cheat_breakdown"]))
        if grpo_metrics["api_cheat_breakdown"]
        else 0.0,
        "hint_cheat_breakdown": float(np.mean(grpo_metrics["hint_cheat_breakdown"]))
        if grpo_metrics["hint_cheat_breakdown"]
        else 0.0,
        "bypass_cheat_breakdown": float(np.mean(grpo_metrics["bypass_cheat_breakdown"]))
        if grpo_metrics["bypass_cheat_breakdown"]
        else 0.0,
    }
    print("-" * 72)
    print("Model           Reward   Cheating   AvgSteps   API_Freq   Hint_Freq")
    print(
        f"Random          {baselines['random']['reward']:.4f}   {baselines['random']['cheat_rate']:.4f}   "
        f"{baselines['random']['avg_steps']:.2f}      {baselines['random']['api_usage_frequency']:.4f}    "
        f"{baselines['random']['hint_decode_frequency']:.4f}"
    )
    print(
        f"Heuristic       {baselines['heuristic']['reward']:.4f}   {baselines['heuristic']['cheat_rate']:.4f}   "
        f"{baselines['heuristic']['avg_steps']:.2f}      {baselines['heuristic']['api_usage_frequency']:.4f}    "
        f"{baselines['heuristic']['hint_decode_frequency']:.4f}"
    )
    if untrained_baseline:
        print(
            f"Untrained LLM   {untrained_baseline['reward']:.4f}   {untrained_baseline['cheat_rate']:.4f}   "
            f"{untrained_baseline['avg_steps']:.2f}      {untrained_baseline['api_usage_frequency']:.4f}    "
            f"{untrained_baseline['hint_decode_frequency']:.4f}"
        )
    print(
        f"GRPO            {grpo_row['reward']:.4f}   {grpo_row['cheat_rate']:.4f}   "
        f"{grpo_row['avg_steps']:.2f}      {grpo_row['api_usage_frequency']:.4f}    "
        f"{grpo_row['hint_decode_frequency']:.4f}"
    )
    print(
        "Cheat Breakdown (GRPO): "
        f"API usage: {grpo_row['api_cheat_breakdown'] * 100:.1f}% | "
        f"Hint decoding: {grpo_row['hint_cheat_breakdown'] * 100:.1f}% | "
        f"Bypass: {grpo_row['bypass_cheat_breakdown'] * 100:.1f}%"
    )


def print_trajectory_visualization(grpo_metrics: dict[str, list[float]]) -> None:
    """Print before/after trajectory paths for demos."""
    worst_path = grpo_metrics.get("worst_path", [""])[0] if grpo_metrics.get("worst_path") else ""
    best_path = grpo_metrics.get("best_path", [""])[0] if grpo_metrics.get("best_path") else ""
    worst_reward = grpo_metrics.get("worst_reward_seen", [0.0])[0] if grpo_metrics.get("worst_reward_seen") else 0.0
    best_reward = grpo_metrics.get("best_reward_seen", [0.0])[0] if grpo_metrics.get("best_reward_seen") else 0.0

    print("-" * 72)
    print("Trajectory Visualization")
    print(f"Before (low reward={worst_reward:.4f}): {worst_path} [BAD]")
    print(f"After  (high reward={best_reward:.4f}): {best_path} [GOOD]")


def main() -> None:
    args = parse_args()
    cfg = GRPOConfig(
        model_name=args.model_name,
        num_episodes=args.episodes,
        group_size=args.group_size,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        curriculum_level=args.curriculum_level,
        use_curriculum_schedule=not args.no_curriculum_schedule,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        output_log_path=args.output,
    )
    baselines = {}
    untrained_baseline = None
    if not args.skip_baselines:
        baselines = run_baselines(
            episodes=args.baseline_episodes,
            seed=args.seed,
            max_steps=args.max_steps,
            curriculum_level=args.curriculum_level,
        )
    if args.untrained_episodes > 0:
        untrained_baseline = evaluate_untrained_model_baseline(
            model_name=args.model_name,
            episodes=args.untrained_episodes,
            seed=args.seed,
            max_steps=args.max_steps,
            curriculum_level=args.curriculum_level,
        )
    metrics = train_grpo(cfg)
    print("-" * 60)
    print(f"Final mean reward: {float(np.mean(metrics['rewards'])):.4f}")
    print(f"Final mean cheat rate: {float(np.mean(metrics['cheat_rates'])):.4f}")
    print(f"Final mean workflow score: {float(np.mean(metrics['workflow_scores'])):.4f}")
    print(f"Final mean steps: {float(np.mean(metrics['avg_steps'])):.2f}")
    print(f"Final API usage frequency: {float(np.mean(metrics['api_usage_frequencies'])):.4f}")
    print(f"Final hint decode frequency: {float(np.mean(metrics['hint_decode_frequencies'])):.4f}")
    print(
        "Final cheating breakdown: "
        f"API={float(np.mean(metrics['api_cheat_breakdown'])) * 100:.1f}% "
        f"Hint={float(np.mean(metrics['hint_cheat_breakdown'])) * 100:.1f}% "
        f"Bypass={float(np.mean(metrics['bypass_cheat_breakdown'])) * 100:.1f}%"
    )
    print_trajectory_visualization(metrics)
    if baselines:
        print_baseline_comparison(baselines, metrics, untrained_baseline=untrained_baseline)
    curve_path = plot_metrics(args.output)
    print(f"Saved metrics file: {args.output}")
    print(f"Saved curve plot: {curve_path}")


if __name__ == "__main__":
    main()
