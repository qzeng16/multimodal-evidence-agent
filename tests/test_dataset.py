from pathlib import Path

import pytest

from src.dataset import (
    find_example_by_id,
    load_examples,
    select_examples,
)


DATA_PATH = Path("data/samples.jsonl")


def test_load_examples_returns_full_dataset() -> None:
    """The project dataset should load all annotated examples."""

    examples = load_examples(DATA_PATH)

    assert len(examples) == 21
    assert examples[0].example_id == "sample_001"


def test_find_example_by_id_returns_match() -> None:
    """A known example ID should return the correct example."""

    examples = load_examples(DATA_PATH)

    example = find_example_by_id(
        examples=examples,
        example_id="sample_004",
    )

    assert example is not None
    assert example.example_id == "sample_004"
    assert example.gold_label == "supported"
    assert example.expected_use_ocr is True


def test_find_example_by_id_returns_none_for_unknown_id() -> None:
    """An unknown example ID should not produce a false match."""

    examples = load_examples(DATA_PATH)

    example = find_example_by_id(
        examples=examples,
        example_id="missing_example",
    )

    assert example is None


def test_select_examples_returns_one_example() -> None:
    """Selecting a valid ID should return exactly one example."""

    examples = load_examples(DATA_PATH)

    selected = select_examples(
        examples=examples,
        example_id="sample_004",
    )

    assert len(selected) == 1
    assert selected[0].example_id == "sample_004"


def test_select_examples_returns_all_when_id_is_none() -> None:
    """Omitting an ID should preserve the full dataset."""

    examples = load_examples(DATA_PATH)

    selected = select_examples(
        examples=examples,
        example_id=None,
    )

    assert selected == examples


def test_select_examples_rejects_unknown_id() -> None:
    """Selecting an unknown ID should raise a clear error."""

    examples = load_examples(DATA_PATH)

    with pytest.raises(
        ValueError,
        match="Example ID not found",
    ):
        select_examples(
            examples=examples,
            example_id="missing_example",
        )


def test_load_examples_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A missing JSONL file should raise FileNotFoundError."""

    missing_path = tmp_path / "missing.jsonl"

    with pytest.raises(
        FileNotFoundError,
        match="Dataset not found",
    ):
        load_examples(missing_path)


def test_load_examples_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSONL input should report the line number."""

    invalid_path = tmp_path / "invalid.jsonl"

    invalid_path.write_text(
        '{"example_id": "broken"\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON on dataset line 1",
    ):
        load_examples(invalid_path)
