"""Client wrapper for remote/local TrustOps OpenEnv server."""

from __future__ import annotations

from typing import Any

import requests


class TrustOpsClient:
    """HTTP client for interacting with TrustOps server endpoints."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/reset", timeout=30)
        response.raise_for_status()
        return response.json()

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        response = requests.post(f"{self.base_url}/step", json={"action": action}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return payload["state"], float(payload["reward"]), bool(payload["done"]), payload["info"]

    def state(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return response.json()
