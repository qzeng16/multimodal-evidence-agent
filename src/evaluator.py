import json
from pathlib import Path
from typing import Dict, List

from src.schemas import VerificationInput, VerificationResult


LABELS = [
    "supported",
    "refuted",
    "insufficient",
]


def save_predictions(
    examples: List[VerificationInput],
    results: List[VerificationResult],
    output_path: Path,
) -> None:
    """Save predictions, evidence, and tool traces as JSONL."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_by_id = {
        result.example_id: result
        for result in results
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for example in examples:
            result = result_by_id.get(
                example.example_id
            )

            if result is None:
                continue

            if example.gold_label is not None:
                correct = (
                    result.label
                    == example.gold_label
                )
            else:
                correct = None

            record = {
                "example_id": example.example_id,
                "claim": example.claim,
                "image_path": example.image_path,
                "context": example.context,
                "gold_label": example.gold_label,
                "predicted_label": result.label,
                "confidence": result.confidence,
                "correct": correct,
                "rationale": result.rationale,
                "evidence": [
                    evidence.model_dump()
                    for evidence in result.evidence
                ],
                "tool_trace": [
                    tool_call.model_dump()
                    for tool_call in result.tool_trace
                ],
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def evaluate_predictions(
    examples: List[VerificationInput],
    results: List[VerificationResult],
) -> Dict:
    """Calculate overall accuracy and per-label metrics."""

    result_by_id = {
        result.example_id: result
        for result in results
    }

    per_label = {
        label: {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
        }
        for label in LABELS
    }

    confusion_matrix = {
        gold_label: {
            predicted_label: 0
            for predicted_label in LABELS
        }
        for gold_label in LABELS
    }

    evaluated_examples = 0
    correct_examples = 0
    failed_examples = 0

    for example in examples:
        if example.gold_label is None:
            continue

        result = result_by_id.get(
            example.example_id
        )

        if result is None:
            failed_examples += 1
            continue

        evaluated_examples += 1

        gold_label = example.gold_label
        predicted_label = result.label

        per_label[gold_label]["total"] += 1

        confusion_matrix[gold_label][
            predicted_label
        ] += 1

        if predicted_label == gold_label:
            correct_examples += 1
            per_label[gold_label]["correct"] += 1

    for label in LABELS:
        label_total = per_label[label]["total"]
        label_correct = per_label[label]["correct"]

        if label_total > 0:
            per_label[label]["accuracy"] = (
                label_correct / label_total
            )

    if evaluated_examples > 0:
        accuracy = (
            correct_examples
            / evaluated_examples
        )
    else:
        accuracy = 0.0

    return {
        "total_examples": len(examples),
        "evaluated_examples": evaluated_examples,
        "correct_examples": correct_examples,
        "failed_examples": failed_examples,
        "accuracy": accuracy,
        "per_label": per_label,
        "confusion_matrix": confusion_matrix,
    }


def save_metrics(
    metrics: Dict,
    output_path: Path,
) -> None:
    """Save evaluation metrics as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_evaluation_report(
    metrics: Dict,
) -> None:
    """Print the complete dataset evaluation report."""

    print("\n" + "=" * 60)
    print("DATASET EVALUATION")
    print("=" * 60)

    print(
        f"Total examples: "
        f"{metrics['total_examples']}"
    )

    print(
        f"Evaluated examples: "
        f"{metrics['evaluated_examples']}"
    )

    print(
        f"Correct examples: "
        f"{metrics['correct_examples']}"
    )

    print(
        f"Failed examples: "
        f"{metrics['failed_examples']}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.3f}"
    )

    print("\nPer-label accuracy:")

    for label in LABELS:
        label_metrics = metrics[
            "per_label"
        ][label]

        print(
            f"- {label}: "
            f"{label_metrics['correct']}/"
            f"{label_metrics['total']} "
            f"({label_metrics['accuracy']:.3f})"
        )

    print("\nConfusion matrix:")

    print(
        "Gold label       "
        "supported  refuted  insufficient"
    )

    for gold_label in LABELS:
        row = metrics[
            "confusion_matrix"
        ][gold_label]

        print(
            f"{gold_label:<16}"
            f"{row['supported']:<11}"
            f"{row['refuted']:<9}"
            f"{row['insufficient']}"
        )