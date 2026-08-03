import json
from pathlib import Path
from typing import Any, Dict, List

from src.schemas import (
    VerificationInput,
    VerificationResult,
)


LABELS = [
    "supported",
    "refuted",
    "insufficient",
]

MODEL_TOOL_NAMES = {
    "image_inspector",
    "ocr_tool",
    "verification_reasoner",
}


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Return zero when the denominator is zero."""

    if denominator == 0:
        return 0.0

    return numerator / denominator


def get_tool_names(
    result: VerificationResult,
) -> List[str]:
    """Return tool names in execution order."""

    return [
        tool_call.tool_name
        for tool_call in result.tool_trace
    ]


def expected_tool_sequence(
    expected_use_ocr: bool,
) -> List[str]:
    """Return the expected minimal tool sequence."""

    sequence = [
        "tool_router",
        "image_inspector",
    ]

    if expected_use_ocr:
        sequence.append("ocr_tool")

    sequence.append(
        "verification_reasoner"
    )

    return sequence


def save_predictions(
    examples: List[VerificationInput],
    results: List[VerificationResult],
    output_path: Path,
) -> None:
    """Save predictions, routes, evidence, and traces as JSONL."""

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

            predicted_use_ocr = (
                result.routing_decision.use_ocr
            )

            route_correct = None

            if example.expected_use_ocr is not None:
                route_correct = (
                    predicted_use_ocr
                    == example.expected_use_ocr
                )

            tool_names = get_tool_names(
                result
            )

            optimal_tool_path = None

            if example.expected_use_ocr is not None:
                expected_sequence = (
                    expected_tool_sequence(
                        example.expected_use_ocr
                    )
                )

                optimal_tool_path = (
                    tool_names
                    == expected_sequence
                )

            if example.gold_label is not None:
                prediction_correct = (
                    result.label
                    == example.gold_label
                )
            else:
                prediction_correct = None

            model_call_count = sum(
                1
                for tool_name in tool_names
                if tool_name in MODEL_TOOL_NAMES
            )

            record = {
                "example_id": example.example_id,
                "claim": example.claim,
                "image_path": example.image_path,
                "context": example.context,
                "category": example.category,
                "gold_label": example.gold_label,
                "expected_use_ocr": (
                    example.expected_use_ocr
                ),
                "predicted_label": result.label,
                "confidence": result.confidence,
                "correct": prediction_correct,
                "predicted_use_ocr": (
                    predicted_use_ocr
                ),
                "route_correct": route_correct,
                "optimal_tool_path": (
                    optimal_tool_path
                ),
                "tool_names": tool_names,
                "tool_call_count": len(
                    tool_names
                ),
                "model_call_count": (
                    model_call_count
                ),
                "routing_decision": (
                    result.routing_decision.model_dump()
                ),
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
) -> Dict[str, Any]:
    """
    Calculate classification, routing, and efficiency metrics.
    """

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

    per_category: Dict[
        str,
        Dict[str, Any],
    ] = {}

    evaluated_examples = 0
    correct_examples = 0
    failed_examples = 0
    successful_examples = 0

    confidence_total = 0.0

    routing_total = 0
    routing_tp = 0
    routing_fp = 0
    routing_fn = 0
    routing_tn = 0

    total_tool_calls = 0
    total_model_calls = 0

    efficiency_annotated_examples = 0
    optimal_tool_paths = 0
    total_extra_tool_calls = 0
    total_missing_tool_calls = 0

    for example in examples:
        result = result_by_id.get(
            example.example_id
        )

        if result is None:
            failed_examples += 1
            continue

        successful_examples += 1
        confidence_total += result.confidence

        tool_names = get_tool_names(
            result
        )

        total_tool_calls += len(
            tool_names
        )

        total_model_calls += sum(
            1
            for tool_name in tool_names
            if tool_name in MODEL_TOOL_NAMES
        )

        if example.expected_use_ocr is not None:
            routing_total += 1

            expected_use_ocr = (
                example.expected_use_ocr
            )

            predicted_use_ocr = (
                result.routing_decision.use_ocr
            )

            if (
                expected_use_ocr
                and predicted_use_ocr
            ):
                routing_tp += 1

            elif (
                not expected_use_ocr
                and predicted_use_ocr
            ):
                routing_fp += 1

            elif (
                expected_use_ocr
                and not predicted_use_ocr
            ):
                routing_fn += 1

            else:
                routing_tn += 1

            expected_sequence = (
                expected_tool_sequence(
                    expected_use_ocr
                )
            )

            efficiency_annotated_examples += 1

            if tool_names == expected_sequence:
                optimal_tool_paths += 1

            total_extra_tool_calls += max(
                0,
                len(tool_names)
                - len(expected_sequence),
            )

            total_missing_tool_calls += max(
                0,
                len(expected_sequence)
                - len(tool_names),
            )

        if example.gold_label is None:
            continue

        evaluated_examples += 1

        gold_label = example.gold_label
        predicted_label = result.label

        per_label[
            gold_label
        ]["total"] += 1

        confusion_matrix[
            gold_label
        ][predicted_label] += 1

        category = (
            example.category
            if example.category
            else "uncategorized"
        )

        if category not in per_category:
            per_category[category] = {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
            }

        per_category[
            category
        ]["total"] += 1

        if predicted_label == gold_label:
            correct_examples += 1

            per_label[
                gold_label
            ]["correct"] += 1

            per_category[
                category
            ]["correct"] += 1

    for label in LABELS:
        label_total = per_label[
            label
        ]["total"]

        label_correct = per_label[
            label
        ]["correct"]

        per_label[
            label
        ]["accuracy"] = safe_divide(
            label_correct,
            label_total,
        )

    for category in per_category:
        category_total = per_category[
            category
        ]["total"]

        category_correct = per_category[
            category
        ]["correct"]

        per_category[
            category
        ]["accuracy"] = safe_divide(
            category_correct,
            category_total,
        )

    accuracy = safe_divide(
        correct_examples,
        evaluated_examples,
    )

    average_confidence = safe_divide(
        confidence_total,
        successful_examples,
    )

    router_accuracy = safe_divide(
        routing_tp + routing_tn,
        routing_total,
    )

    ocr_precision = safe_divide(
        routing_tp,
        routing_tp + routing_fp,
    )

    ocr_recall = safe_divide(
        routing_tp,
        routing_tp + routing_fn,
    )

    ocr_f1 = safe_divide(
        2 * ocr_precision * ocr_recall,
        ocr_precision + ocr_recall,
    )

    ocr_invocation_rate = safe_divide(
        routing_tp + routing_fp,
        routing_total,
    )

    unnecessary_ocr_rate = safe_divide(
        routing_fp,
        routing_fp + routing_tn,
    )

    missed_ocr_rate = safe_divide(
        routing_fn,
        routing_tp + routing_fn,
    )

    average_tool_calls = safe_divide(
        total_tool_calls,
        successful_examples,
    )

    average_model_calls = safe_divide(
        total_model_calls,
        successful_examples,
    )

    optimal_tool_path_rate = safe_divide(
        optimal_tool_paths,
        efficiency_annotated_examples,
    )

    average_extra_tool_calls = safe_divide(
        total_extra_tool_calls,
        efficiency_annotated_examples,
    )

    average_missing_tool_calls = safe_divide(
        total_missing_tool_calls,
        efficiency_annotated_examples,
    )

    return {
        "total_examples": len(examples),
        "evaluated_examples": evaluated_examples,
        "correct_examples": correct_examples,
        "failed_examples": failed_examples,
        "accuracy": accuracy,
        "average_confidence": average_confidence,
        "per_label": per_label,
        "per_category": per_category,
        "confusion_matrix": confusion_matrix,
        "routing": {
            "annotated_examples": routing_total,
            "true_positive": routing_tp,
            "false_positive": routing_fp,
            "false_negative": routing_fn,
            "true_negative": routing_tn,
            "router_accuracy": router_accuracy,
            "ocr_precision": ocr_precision,
            "ocr_recall": ocr_recall,
            "ocr_f1": ocr_f1,
            "ocr_invocation_rate": (
                ocr_invocation_rate
            ),
            "unnecessary_ocr_rate": (
                unnecessary_ocr_rate
            ),
            "missed_ocr_rate": (
                missed_ocr_rate
            ),
        },
        "efficiency": {
            "successful_examples": (
                successful_examples
            ),
            "average_tool_calls": (
                average_tool_calls
            ),
            "average_model_calls": (
                average_model_calls
            ),
            "annotated_examples": (
                efficiency_annotated_examples
            ),
            "optimal_tool_paths": (
                optimal_tool_paths
            ),
            "optimal_tool_path_rate": (
                optimal_tool_path_rate
            ),
            "average_extra_tool_calls": (
                average_extra_tool_calls
            ),
            "average_missing_tool_calls": (
                average_missing_tool_calls
            ),
        },
    }


def save_metrics(
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save all evaluation metrics as JSON."""

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
    metrics: Dict[str, Any],
) -> None:
    """Print classification and agent evaluation metrics."""

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

    print(
        f"Average confidence: "
        f"{metrics['average_confidence']:.3f}"
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

    print("\nPer-category accuracy:")

    for category, category_metrics in (
        metrics["per_category"].items()
    ):
        print(
            f"- {category}: "
            f"{category_metrics['correct']}/"
            f"{category_metrics['total']} "
            f"({category_metrics['accuracy']:.3f})"
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

    routing = metrics[
        "routing"
    ]

    print("\n" + "=" * 60)
    print("TOOL ROUTING EVALUATION")
    print("=" * 60)

    print(
        f"Annotated examples: "
        f"{routing['annotated_examples']}"
    )

    print(
        f"Router accuracy: "
        f"{routing['router_accuracy']:.3f}"
    )

    print(
        f"OCR precision: "
        f"{routing['ocr_precision']:.3f}"
    )

    print(
        f"OCR recall: "
        f"{routing['ocr_recall']:.3f}"
    )

    print(
        f"OCR F1: "
        f"{routing['ocr_f1']:.3f}"
    )

    print(
        f"OCR invocation rate: "
        f"{routing['ocr_invocation_rate']:.3f}"
    )

    print(
        f"Unnecessary OCR rate: "
        f"{routing['unnecessary_ocr_rate']:.3f}"
    )

    print(
        f"Missed OCR rate: "
        f"{routing['missed_ocr_rate']:.3f}"
    )

    efficiency = metrics[
        "efficiency"
    ]

    print("\n" + "=" * 60)
    print("TOOL-USE EFFICIENCY")
    print("=" * 60)

    print(
        f"Average tool calls: "
        f"{efficiency['average_tool_calls']:.3f}"
    )

    print(
        f"Average model calls: "
        f"{efficiency['average_model_calls']:.3f}"
    )

    print(
        f"Optimal tool-path rate: "
        f"{efficiency['optimal_tool_path_rate']:.3f}"
    )

    print(
        f"Average extra tool calls: "
        f"{efficiency['average_extra_tool_calls']:.3f}"
    )

    print(
        f"Average missing tool calls: "
        f"{efficiency['average_missing_tool_calls']:.3f}"
    )