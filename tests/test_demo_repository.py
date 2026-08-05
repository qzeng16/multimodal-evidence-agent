import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.demo_repository import (
    DEMO_RESULTS_PATH,
    build_demo_summary,
    find_demo_result,
    list_demo_summaries,
    load_demo_results,
)


def make_record(
    example_id: str = "sample_test",
) -> Dict[str, Any]:
    """Create one valid demo record for isolated tests."""

    return {
        "example_id": example_id,
        "claim": "A test claim.",
        "image_path": (
            "data/images/sample_001.png"
        ),
        "category": "visual_state",
        "predicted_label": "supported",
        "confidence": 0.95,
        "predicted_use_ocr": False,
        "rationale": "Visible evidence supports the claim.",
        "evidence": [],
        "tool_trace": [],
    }


def write_payload(
    path: Path,
    payload: Any,
) -> None:
    """Write a JSON payload to a temporary file."""

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_curated_demo_file_loads() -> None:
    """The committed demo dataset should contain six examples."""

    records = load_demo_results(
        DEMO_RESULTS_PATH
    )

    assert len(records) == 6

    assert {
        record["example_id"]
        for record in records
    } == {
        "sample_001",
        "sample_002",
        "sample_003",
        "sample_004",
        "sample_005",
        "sample_012",
    }


def test_demo_labels_cover_all_outcomes() -> None:
    """The curated dataset should demonstrate all three labels."""

    records = load_demo_results()

    labels = {
        record["predicted_label"]
        for record in records
    }

    assert labels == {
        "supported",
        "refuted",
        "insufficient",
    }


def test_demo_examples_cover_ocr_and_visual_routes() -> None:
    """The demo should include OCR and non-OCR executions."""

    records = load_demo_results()

    ocr_values = {
        record["predicted_use_ocr"]
        for record in records
    }

    assert ocr_values == {
        True,
        False,
    }


def test_list_demo_summaries_is_lightweight() -> None:
    """Summary records should omit large evidence details."""

    summaries = list_demo_summaries()

    assert len(summaries) == 6

    first_summary = summaries[0]

    assert "example_id" in first_summary
    assert "predicted_label" in first_summary

    assert "evidence" not in first_summary
    assert "tool_trace" not in first_summary
    assert "rationale" not in first_summary


def test_build_demo_summary() -> None:
    """A full record should convert to the public summary shape."""

    record = make_record()

    summary = build_demo_summary(
        record
    )

    assert summary == {
        "example_id": "sample_test",
        "claim": "A test claim.",
        "image_path": (
            "data/images/sample_001.png"
        ),
        "category": "visual_state",
        "predicted_label": "supported",
        "confidence": 0.95,
        "predicted_use_ocr": False,
    }


def test_find_demo_result_returns_matching_record() -> None:
    """An existing ID should return its full static result."""

    result = find_demo_result(
        "sample_004"
    )

    assert result is not None
    assert result["example_id"] == "sample_004"
    assert result["predicted_use_ocr"] is True
    assert result["evidence"]
    assert result["tool_trace"]


def test_find_demo_result_returns_none_for_unknown_id() -> None:
    """An unknown ID should not produce a fabricated result."""

    result = find_demo_result(
        "not-a-real-example"
    )

    assert result is None


def test_missing_demo_file_is_rejected(
    tmp_path: Path,
) -> None:
    """A missing data file should raise a clear error."""

    missing_path = (
        tmp_path
        / "missing.json"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Demo results file not found",
    ):
        load_demo_results(
            missing_path
        )


def test_invalid_json_is_rejected(
    tmp_path: Path,
) -> None:
    """Malformed JSON should not be silently accepted."""

    path = (
        tmp_path
        / "invalid.json"
    )

    path.write_text(
        "{not valid json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        load_demo_results(
            path
        )


def test_non_array_payload_is_rejected(
    tmp_path: Path,
) -> None:
    """The top-level demo payload must be a JSON array."""

    path = (
        tmp_path
        / "object.json"
    )

    write_payload(
        path,
        make_record(),
    )

    with pytest.raises(
        ValueError,
        match="JSON array",
    ):
        load_demo_results(
            path
        )


def test_duplicate_example_ids_are_rejected(
    tmp_path: Path,
) -> None:
    """Demo IDs must be unique."""

    path = (
        tmp_path
        / "duplicates.json"
    )

    write_payload(
        path,
        [
            make_record(
                "duplicate"
            ),
            make_record(
                "duplicate"
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Duplicate demo example_id"
        ),
    ):
        load_demo_results(
            path
        )


def test_missing_required_field_is_rejected(
    tmp_path: Path,
) -> None:
    """Incomplete records should fail validation."""

    path = (
        tmp_path
        / "missing-field.json"
    )

    record = make_record()

    del record["rationale"]

    write_payload(
        path,
        [record],
    )

    with pytest.raises(
        ValueError,
        match="missing fields: rationale",
    ):
        load_demo_results(
            path
        )
