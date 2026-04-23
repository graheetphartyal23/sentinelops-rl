# SentinelOps-RL (OpenEnv Hackathon Submission)

SentinelOps-RL is an integrity-aware reinforcement learning environment for enterprise cyber incident response.  
The project trains LLM policies to perform multi-step security workflows while resisting reward-hacking shortcuts.

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
python -m train.phase5_evaluation --metrics train/grpo_metrics.npz --output-dir train
```

Outputs:
- `train/reward_curve.png`
- `train/cheating_curve.png`
- `train/workflow_curve.png`
- `train/steps_curve.png`
- `train/cheat_breakdown.png`
- `train/sample_trajectories.txt`
- `train/baseline_vs_trained.csv`

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

## Submission Links (Fill Before Final Submission)

- Hugging Face Space URL: `<ADD_URL>`
- Colab notebook URL: `<ADD_URL>`
- Mini-blog / YouTube video URL (<2 min): `<ADD_URL>`
- Optional slides URL: `<ADD_URL>`

## Quick References

- `SUBMISSION_CHECKLIST.md` for requirement tracking
- `train_colab.ipynb` for runnable Colab training flow
