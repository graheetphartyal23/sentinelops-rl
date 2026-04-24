---
title: Sentinelops Rl
emoji: 📉
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

# SentinelOps-RL (OpenEnv Hackathon Submission)

SentinelOps-RL is an integrity-aware reinforcement learning environment for enterprise cyber incident response.  
The project trains LLM policies to perform multi-step security workflows while resisting reward-hacking shortcuts.

## Submission Links

- Hugging Face Space URL: `<PASTE_HF_SPACE_URL_HERE>`
- Colab notebook URL: `<PASTE_COLAB_URL_HERE>`
- Mini-blog / YouTube video URL (<2 min): `<PASTE_VIDEO_OR_BLOG_URL_HERE>`
- Optional slides URL: `<PASTE_SLIDES_URL_HERE>`

## Problem Statement

Frontier LLM agents often optimize proxy rewards and exploit shortcuts instead of following safe incident-response procedures.  
SentinelOps-RL targets this gap by combining:
- verifiable task outcomes,
- trajectory-level integrity constraints,
- adversarial trap design,
- long-horizon workflow behavior.

The goal is not only to solve incident tasks, but to solve them **without cheating**.

## Environment Design

### Task tracks
- `incident_triage`: identify attack root cause, prioritize assets, propose mitigation
- `response_integrity`: build policy-compliant action plan while avoiding forbidden shortcuts
- `stakeholder_negotiation`: produce cross-team response brief under conflicting priorities

### Observation/action loop
- `reset()` returns the first chunked observation
- `step(action)` applies one action and returns `(state, reward, done, info)`
- `state()` returns current observable view

### Action space
- `read_data`
- `analyze`
- `write_data`
- `call_api` (trap vector at higher curriculum levels)
- `decode_hint` (trap vector at higher curriculum levels)
- `submit`

### Trap and anti-hacking system
- hidden API answer trap
- encoded hint trap
- validation bypass trap
- ambiguity signals for adversarial curriculum

### Reward
Terminal-only reward:

`reward = correctness * (1 - cheat_score) * workflow_score`

Where:
- `correctness` is task-verifier score
- `cheat_score` is deterministic from trajectory events
- `workflow_score` measures multi-step process quality

### Rubric-style scoring components (OpenEnv-aligned language)

SentinelOps-RL uses composable scoring components that act like rubric checks:
- **Outcome Rubric (`correctness`)**: did the final submission solve the task objective?
- **Integrity Rubric (`cheat_score`)**: did the trajectory exploit traps (API/hint/bypass)?
- **Process Rubric (`workflow_score`)**: did the agent follow a reasonable multi-step workflow?

These are computed independently and then composed into the final reward, which reduces reward hacking by requiring both correctness and clean behavior.

## Data Strategy (Large, Complex Corpora)

The project includes a corpus catalog for large real-world sources:
- BGL logs
- HDFS logs
- NVD CVE feed
- CISA KEV
- MITRE ATT&CK
- Enron email corpus

See:
- `data_pipeline/corpus_catalog.py`
- `data_pipeline/prepare_corpora.py`

Generate the corpus manifest:

```bash
python -m data_pipeline.prepare_corpora
```

## OpenEnv Compliance

- OpenEnv package version: **0.1.13** (latest available on PyPI at submission time)
- OpenEnv manifest: `openenv.yaml`
- OpenEnv-compatible environment class: `env/environment.py`
- FastAPI environment server: `server/main.py`
- Client/server separation: `client/trustops_client.py`

Run server locally:

```bash
python server/main.py
```

Client usage example:

```python
from client.trustops_client import TrustOpsClient

client = TrustOpsClient("http://127.0.0.1:8000")
state = client.reset()
state, reward, done, info = client.step({"type": "read_data", "payload": {}})
```

## Training Pipeline (TRL + GRPO)

Install:

```bash
pip install -r requirements_grpo.txt
```

Run GRPO training:

```bash
python -m train.run_grpo --model-name "Qwen/Qwen2.5-0.5B-Instruct" --episodes 300 --group-size 4 --max-steps 30 --lr 1e-5 --temperature 0.7 --top-p 0.9
```

Optional (explicit untrained-LLM baseline):

```bash
python -m train.run_grpo --model-name "Qwen/Qwen2.5-0.5B-Instruct" --episodes 300 --group-size 4 --max-steps 30 --lr 1e-5 --untrained-episodes 8
```

Training includes:
- grouped rollouts
- relative advantage normalization
- curriculum schedule (L1 -> L2 -> L3)
- baseline comparisons (random + heuristic + untrained LLM policy)
- cheat breakdown (API / hint / bypass)
- trajectory before/after visualization

## Metrics and Results Package

Generate evaluation artifacts:

```bash
python -m train.phase5_evaluation --metrics train/grpo_metrics_60ep_v3.npz --output-dir train/results_60ep_v3
```

Outputs:
- `train/results_60ep_v3/reward_curve.png`
- `train/results_60ep_v3/cheating_curve.png`
- `train/results_60ep_v3/workflow_curve.png`
- `train/results_60ep_v3/steps_curve.png`
- `train/results_60ep_v3/cheat_breakdown.png`
- `train/results_60ep_v3/sample_trajectories.txt`
- `train/results_60ep_v3/baseline_vs_trained.csv`

### Latest Training Run (60 Episodes, tiny-gpt2)

This run is intentionally longer than quick smoke tests to show visible metric movement over curriculum levels.

#### Before vs After (Random / Untrained / GRPO)

| Policy | Reward (mean) | Cheating (mean) | Avg Steps |
|---|---:|---:|---:|
| Random | 0.0000 | 0.4225 | 2.75 |
| Untrained LLM | 0.0000 | 0.6250 | 5.00 |
| GRPO (trained) | 0.0017 | 0.6433 | 4.45 |

#### Judge-facing result highlights

- This run demonstrates end-to-end training/evaluation wiring, but learning quality is still limited with `tiny-gpt2`.
- Behavior change appears in trajectories (more read/analyze/write steps in higher-reward episodes), but cheating remains high.
- Final submission run should use a stronger model and longer horizon to show clearer reward-up and cheating-down trends.

Recommended final evidence run (for stronger learning signal):

```bash
python -m train.run_grpo --model-name "Qwen/Qwen2.5-0.5B-Instruct" --episodes 200 --group-size 4 --max-steps 30 --lr 1e-5 --temperature 0.7 --top-p 0.9 --baseline-episodes 40 --untrained-episodes 12 --output train/grpo_metrics_qwen_200ep.npz
python -m train.phase5_evaluation --metrics train/grpo_metrics_qwen_200ep.npz --output-dir train/results_qwen_200ep --smoothing-window 12
```

### Embedded Plots (with Captions)

![Reward curve: GRPO vs random baseline across 60 training episodes.](train/results_60ep_v3/reward_curve.png)
Reward trend over training episodes (raw + smoothed), with random baseline overlay for quick judge comparison.

![Cheating curve: cheating rate trend across training episodes.](train/results_60ep_v3/cheating_curve.png)
Cheating-rate trend over episodes (lower is better), with baseline overlay to show policy behavior gap.

![Workflow score curve: process quality trend across training episodes.](train/results_60ep_v3/workflow_curve.png)
Workflow quality trend over episodes; higher indicates better multi-step process quality.

![Average steps curve: episode length trend across training episodes.](train/results_60ep_v3/steps_curve.png)
Average number of steps taken before submission over episodes.

![Cheat breakdown curve: API, hint, and bypass cheating components over episodes.](train/results_60ep_v3/cheat_breakdown.png)
Cheating composition by source (API/hint/bypass) across training.

## Baseline and Inference

- Baseline module: `train/baseline.py`
- Inference entrypoint: `inference.py`

Run inference:

```bash
python inference.py
```

## Demo Deployment

- HF Space app: `hf_space/app.py`
- HF Space requirements: `hf_space/requirements.txt`
- Docker reproducibility: `Dockerfile`

## Quick References

- `SUBMISSION_CHECKLIST.md` for requirement tracking
- `train_colab.ipynb` for runnable Colab training flow
