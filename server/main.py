"""OpenEnv-compatible FastAPI server for TrustOps."""

from __future__ import annotations

import os
import sys
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.environment import TrustOpsEnv

app = FastAPI(title="TrustOps OpenEnv Server", version="0.1.0")
ENV = TrustOpsEnv(seed=42)


class StepRequest(BaseModel):
    """Step payload wrapper."""

    action: dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    """Environment step response."""

    state: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset() -> dict[str, Any]:
    return ENV.reset()


@app.post("/step", response_model=StepResponse)
def step(payload: StepRequest) -> StepResponse:
    state, reward, done, info = ENV.step(payload.action)
    return StepResponse(state=state, reward=reward, done=done, info=info)


@app.get("/state")
def state() -> dict[str, Any]:
    return ENV.state()


def create_environment(seed: int = 42) -> TrustOpsEnv:
    """Factory helper used by OpenEnv tooling."""
    return TrustOpsEnv(seed=seed)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
