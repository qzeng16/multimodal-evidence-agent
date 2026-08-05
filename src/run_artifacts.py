"""Utilities for creating isolated, reproducible run directories."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_RUNS_ROOT = Path("outputs/runs")


@dataclass(frozen=True)
class RunArtifacts:
    """Paths associated with one verification or evaluation run."""

    run_id: str
    run_directory: Path
    predictions_path: Path
    metrics_path: Path
    manifest_path: Path


def slugify(value: str) -> str:
    """Convert a human-readable name into a safe path component."""

    normalized = value.strip().lower()

    normalized = re.sub(
        r"[^a-z0-9._-]+",
        "-",
        normalized,
    )

    normalized = re.sub(
        r"-{2,}",
        "-",
        normalized,
    )

    normalized = normalized.strip(
        "-._"
    )

    return normalized or "run"


def create_run_artifacts(
    run_name: Optional[str] = None,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    timestamp: Optional[datetime] = None,
    unique_suffix: Optional[str] = None,
) -> RunArtifacts:
    """Create and return an isolated directory for one run."""

    resolved_timestamp = timestamp or datetime.now(
        timezone.utc
    )

    if resolved_timestamp.tzinfo is None:
        resolved_timestamp = resolved_timestamp.replace(
            tzinfo=timezone.utc
        )

    timestamp_text = resolved_timestamp.astimezone(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    safe_name = slugify(
        run_name or "evaluation"
    )

    suffix = (
        unique_suffix
        or uuid.uuid4().hex[:8]
    )

    safe_suffix = slugify(
        suffix
    )

    run_id = (
        f"{timestamp_text}_"
        f"{safe_name}_"
        f"{safe_suffix}"
    )

    run_directory = (
        Path(runs_root)
        / run_id
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return RunArtifacts(
        run_id=run_id,
        run_directory=run_directory,
        predictions_path=(
            run_directory
            / "predictions.jsonl"
        ),
        metrics_path=(
            run_directory
            / "metrics.json"
        ),
        manifest_path=(
            run_directory
            / "run_manifest.json"
        ),
    )


def write_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    """Write a dictionary as formatted UTF-8 JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    records: Iterable[Dict[str, Any]],
) -> None:
    """Write dictionaries to a UTF-8 JSONL file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for record in records:
            serialized = json.dumps(
                record,
                ensure_ascii=False,
            )

            output_file.write(
                serialized + "\n"
            )


def write_run_manifest(
    artifacts: RunArtifacts,
    metadata: Dict[str, Any],
) -> None:
    """Write metadata describing how one run was produced."""

    manifest = {
        "run_id": artifacts.run_id,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "artifacts": {
            "predictions": (
                artifacts.predictions_path.name
            ),
            "metrics": (
                artifacts.metrics_path.name
            ),
            "manifest": (
                artifacts.manifest_path.name
            ),
        },
        "metadata": metadata,
    }

    write_json(
        artifacts.manifest_path,
        manifest,
    )
