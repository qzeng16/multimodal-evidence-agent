import json
from datetime import datetime, timezone
from pathlib import Path

from src.run_artifacts import (
    create_run_artifacts,
    slugify,
    write_json,
    write_jsonl,
    write_run_manifest,
)


def test_slugify_produces_safe_name() -> None:
    """Human-readable names should become safe path names."""

    assert (
        slugify(
            "Sample 004 / Cached Run"
        )
        == "sample-004-cached-run"
    )


def test_slugify_uses_fallback_for_empty_value() -> None:
    """An empty name should still produce a valid slug."""

    assert slugify("   ") == "run"


def test_create_run_artifacts_creates_directory(
    tmp_path: Path,
) -> None:
    """A run should receive a unique isolated directory."""

    timestamp = datetime(
        2026,
        8,
        4,
        22,
        30,
        0,
        tzinfo=timezone.utc,
    )

    artifacts = create_run_artifacts(
        run_name="sample_004",
        runs_root=tmp_path,
        timestamp=timestamp,
        unique_suffix="abcd1234",
    )

    assert (
        artifacts.run_id
        == (
            "20260804T223000Z_"
            "sample_004_"
            "abcd1234"
        )
    )

    assert artifacts.run_directory.exists()

    assert (
        artifacts.predictions_path
        == (
            artifacts.run_directory
            / "predictions.jsonl"
        )
    )

    assert (
        artifacts.metrics_path
        == (
            artifacts.run_directory
            / "metrics.json"
        )
    )

    assert (
        artifacts.manifest_path
        == (
            artifacts.run_directory
            / "run_manifest.json"
        )
    )


def test_two_runs_receive_different_directories(
    tmp_path: Path,
) -> None:
    """Separate executions should not overwrite each other."""

    timestamp = datetime(
        2026,
        8,
        4,
        22,
        30,
        0,
        tzinfo=timezone.utc,
    )

    first = create_run_artifacts(
        run_name="evaluation",
        runs_root=tmp_path,
        timestamp=timestamp,
        unique_suffix="first",
    )

    second = create_run_artifacts(
        run_name="evaluation",
        runs_root=tmp_path,
        timestamp=timestamp,
        unique_suffix="second",
    )

    assert (
        first.run_directory
        != second.run_directory
    )


def test_write_json(
    tmp_path: Path,
) -> None:
    """JSON output should be readable after writing."""

    output_path = (
        tmp_path
        / "metrics.json"
    )

    write_json(
        output_path,
        {
            "accuracy": 1.0,
            "evaluated_examples": 21,
        },
    )

    loaded = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert loaded["accuracy"] == 1.0
    assert loaded["evaluated_examples"] == 21


def test_write_jsonl(
    tmp_path: Path,
) -> None:
    """JSONL output should preserve one record per line."""

    output_path = (
        tmp_path
        / "predictions.jsonl"
    )

    write_jsonl(
        output_path,
        [
            {
                "example_id": "sample_001",
                "label": "supported",
            },
            {
                "example_id": "sample_002",
                "label": "refuted",
            },
        ],
    )

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2

    assert (
        json.loads(lines[0])["example_id"]
        == "sample_001"
    )

    assert (
        json.loads(lines[1])["example_id"]
        == "sample_002"
    )


def test_write_run_manifest(
    tmp_path: Path,
) -> None:
    """The manifest should describe the run and its files."""

    timestamp = datetime(
        2026,
        8,
        4,
        22,
        30,
        0,
        tzinfo=timezone.utc,
    )

    artifacts = create_run_artifacts(
        run_name="sample_004",
        runs_root=tmp_path,
        timestamp=timestamp,
        unique_suffix="manifest",
    )

    write_run_manifest(
        artifacts,
        {
            "example_count": 1,
            "cache_enabled": True,
        },
    )

    manifest = json.loads(
        artifacts.manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["run_id"]
        == artifacts.run_id
    )

    assert (
        manifest["metadata"]["example_count"]
        == 1
    )

    assert (
        manifest["metadata"]["cache_enabled"]
        is True
    )

    assert (
        manifest["artifacts"]["predictions"]
        == "predictions.jsonl"
    )
