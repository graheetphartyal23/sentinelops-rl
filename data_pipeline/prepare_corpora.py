"""Dataset preparation helpers for large SentinelOps corpora."""

from __future__ import annotations

import json
from pathlib import Path

from data_pipeline.corpus_catalog import CORPUS_SOURCES


def write_manifest(target_dir: str = "data") -> str:
    """Write corpus manifest for reproducible data prep."""
    destination = Path(target_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "corpus_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(CORPUS_SOURCES, handle, indent=2, sort_keys=True)
    return str(manifest_path)


def main() -> None:
    manifest = write_manifest()
    print(f"Wrote corpus manifest: {manifest}")
    print("Download links and target scales are now tracked for training reproducibility.")


if __name__ == "__main__":
    main()
