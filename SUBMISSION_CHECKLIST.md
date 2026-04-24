# OpenEnv Hackathon Submission Checklist

## Minimum Requirements

- [x] OpenEnv used (latest available release on PyPI at submission time: `0.1.13`)
- [x] OpenEnv manifest present: `openenv.yaml`
- [x] OpenEnv entrypoint server present: `server/main.py`
- [x] Training script present with HF TRL stack: `train/run_grpo.py`, `train/grpo_training.py`
- [x] Colab notebook present: `train_colab.ipynb`
- [x] Environment hosted-ready demo app present: `hf_space/app.py`
- [x] Hugging Face Space requirements present: `hf_space/requirements.txt`
- [x] Metrics + graph generation script present: `train/phase5_evaluation.py`
- [x] Baseline comparison present: `train/baseline.py` and `train/baseline_vs_trained.csv`
- [x] Untrained LLM policy baseline support present (`--untrained-episodes` in `train/run_grpo.py`)
- [x] README includes problem, environment, training, results, and artifact links

## Judging Criteria Mapping

- **Environment Innovation (40%)**
  - Enterprise integrity-aware workflow with hidden shortcuts, schema drift, ambiguity signals, and trajectory-aware scoring.
- **Storytelling (30%)**
  - `README.md` + `train/sample_trajectories.txt` + graph artifacts.
- **Improvement in Rewards (20%)**
  - `train/reward_curve.png`, `train/cheating_curve.png`, `train/workflow_curve.png`, `train/steps_curve.png`.
- **Reward & Training Pipeline (10%)**
  - Multi-component cheat detection and workflow scoring in environment; GRPO training script in `train/grpo_training.py`.

## Extra Artifacts

- Inference entrypoint: `inference.py`
- Docker reproducibility: `Dockerfile`
- OpenEnv server/client separation:
  - Server: `server/main.py`
  - Client: `client/trustops_client.py`
