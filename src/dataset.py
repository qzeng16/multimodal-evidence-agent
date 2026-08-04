import json
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from src.schemas import VerificationInput


def load_examples(
    path: Path,
) -> List[VerificationInput]:
    """Load and validate verification examples from JSONL."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Dataset path is not a file: {path}"
        )

    examples: List[VerificationInput] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                raw_example = json.loads(line)

                example = VerificationInput.model_validate(
                    raw_example
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on dataset line "
                    f"{line_number}: {error}"
                ) from error

            except ValidationError as error:
                raise ValueError(
                    f"Invalid example on dataset line "
                    f"{line_number}: {error}"
                ) from error

            examples.append(example)

    return examples


def select_examples(
    examples: List[VerificationInput],
    example_id: Optional[str],
) -> List[VerificationInput]:
    """Select one example, or return all examples."""

    if example_id is None:
        return examples

    selected_examples = [
        example
        for example in examples
        if example.example_id == example_id
    ]

    if not selected_examples:
        raise ValueError(
            f"Example ID not found: {example_id}"
        )

    return selected_examples


def find_example_by_id(
    examples: List[VerificationInput],
    example_id: str,
) -> Optional[VerificationInput]:
    """Return one example by ID."""

    return next(
        (
            example
            for example in examples
            if example.example_id == example_id
        ),
        None,
    )