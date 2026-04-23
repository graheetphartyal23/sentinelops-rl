"""Project-wide configuration values for TrustOps environment."""

MAX_STEPS = 40
MAX_HISTORY_WINDOW = 10
DATASET_ROWS_MIN = 1200
DATA_CHUNK_SIZE = 120

LEVEL_1 = 1
LEVEL_2 = 2
LEVEL_3 = 3
LEVEL_4 = 4
SUPPORTED_LEVELS = {LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_4}

CHEAT_PENALTIES = {
    "api_usage": 0.85,
    "decoded_hint": 0.20,
    "bypass_validation": 0.20,
    "pattern_penalty": 0.15,
}

REWARD_WEIGHTS = {
    "ideal_steps": 8,
    "min_reasonable_steps": 4,
    "min_workflow_floor": 0.05,
    "max_workflow_bonus": 1.0,
    "diversity_weight": 0.35,
    "depth_weight": 0.35,
    "steps_weight": 0.30,
}
