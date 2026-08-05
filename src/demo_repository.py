"""Read and validate curated static demo results."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DEMO_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "demo_results.json"
)

ALLOWED_LABELS = {
    "supported",
    "refuted",
    "insufficient",
}

REQUIRED_FIELDS = {
    "example_id",
    "claim",
    "image_path",
    "category",
    "predicted_label",
    "confidence",
    "predicted_use_ocr",
    "rationale",
    "evidence",
    "tool_trace",
}


def validate_demo_record(
    record: Dict[str, Any],
    index: int,
) -> None:
    """Validate one static demo record."""

    missing_fields = sorted(
        REQUIRED_FIELDS
        - record.keys()
    )

    if missing_fields:
        raise ValueError(
            f"Demo record {index} is missing fields: "
            + ", ".join(missing_fields)
        )

    example_id = record["example_id"]

    if (
        not isinstance(example_id, str)
        or not example_id.strip()
    ):
        raise ValueError(
            f"Demo record {index} has an invalid example_id."
        )

    label = record["predicted_label"]

    if label not in ALLOWED_LABELS:
        raise ValueError(
            f"Demo record {example_id} has "
            f"an invalid predicted_label: {label}"
        )

    confidence = record["confidence"]

    if (
        isinstance(confidence, bool)
        or not isinstance(
            confidence,
            (int, float),
        )
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError(
            f"Demo record {example_id} has "
            "an invalid confidence value."
        )

    if not isinstance(
        record["predicted_use_ocr"],
        bool,
    ):
        raise ValueError(
            f"Demo record {example_id} has "
            "an invalid predicted_use_ocr value."
        )

    if not isinstance(
        record["evidence"],
        list,
    ):
        raise ValueError(
            f"Demo record {example_id} has "
            "an invalid evidence value."
        )

    if not isinstance(
        record["tool_trace"],
        list,
    ):
        raise ValueError(
            f"Demo record {example_id} has "
            "an invalid tool_trace value."
        )


def load_demo_results(
    path: Path = DEMO_RESULTS_PATH,
) -> List[Dict[str, Any]]:
    """Load and validate all static demo records."""

    if not path.exists():
        raise FileNotFoundError(
            f"Demo results file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Demo results path is not a file: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Demo results contain invalid JSON: {path}"
        ) from error

    if not isinstance(payload, list):
        raise ValueError(
            "Demo results must contain a JSON array."
        )

    seen_example_ids = set()
    records: List[Dict[str, Any]] = []

    for index, record in enumerate(
        payload,
        start=1,
    ):
        if not isinstance(record, dict):
            raise ValueError(
                f"Demo record {index} must be a JSON object."
            )

        validate_demo_record(
            record=record,
            index=index,
        )

        example_id = record["example_id"]

        if example_id in seen_example_ids:
            raise ValueError(
                f"Duplicate demo example_id: {example_id}"
            )

        seen_example_ids.add(
            example_id
        )

        records.append(
            record
        )

    return records


def build_demo_summary(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a lightweight representation for the list endpoint."""

    return {
        "example_id": record["example_id"],
        "claim": record["claim"],
        "image_path": record["image_path"],
        "category": record["category"],
        "predicted_label": (
            record["predicted_label"]
        ),
        "confidence": record["confidence"],
        "predicted_use_ocr": (
            record["predicted_use_ocr"]
        ),
    }


def list_demo_summaries(
    path: Path = DEMO_RESULTS_PATH,
) -> List[Dict[str, Any]]:
    """Return lightweight summaries of all demo examples."""

    records = load_demo_results(
        path
    )

    return [
        build_demo_summary(record)
        for record in records
    ]


def find_demo_result(
    example_id: str,
    path: Path = DEMO_RESULTS_PATH,
) -> Optional[Dict[str, Any]]:
    """Return one full demo result by ID."""

    records = load_demo_results(
        path
    )

    for record in records:
        if (
            record["example_id"]
            == example_id
        ):
            return record

    return None
