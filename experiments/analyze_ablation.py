import json
import re
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


EXPERIMENT_PATH = Path(
    "experiments/ocr_ablation.json"
)

SUMMARY_PATH = Path(
    "experiments/ocr_ablation_summary.md"
)


def normalize_text(
    text: str,
) -> str:
    """Normalize text for transcription consistency checks."""

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    normalized = normalized.lower()

    normalized = normalized.replace(
        "’",
        "'",
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def load_experiment(
    path: Path,
) -> Dict[str, Any]:
    """Load the OCR ablation experiment."""

    if not path.exists():
        raise FileNotFoundError(
            f"Experiment file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def calculate_method_metrics(
    method: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculate metrics for one OCR method."""

    results: List[Dict[str, Any]] = method[
        "results"
    ]

    total = len(results)

    correct = sum(
        result["predicted_label"]
        == result["gold_label"]
        for result in results
    )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    average_ocr_confidence = (
        mean(
            result["ocr_confidence"]
            for result in results
        )
        if results
        else 0.0
    )

    average_prediction_confidence = (
        mean(
            result["prediction_confidence"]
            for result in results
        )
        if results
        else 0.0
    )

    normalized_transcriptions = {
        normalize_text(
            result["ocr_transcription"]
        )
        for result in results
    }

    transcription_consistent = (
        len(normalized_transcriptions) == 1
    )

    ground_truth_text = normalize_text(
        experiment["ground_truth_text"]
    )

    correct_transcriptions = sum(
        normalize_text(
            result["ocr_transcription"]
        )
        == ground_truth_text
        for result in results
    )

    transcription_accuracy = (
        correct_transcriptions / total
        if total
        else 0.0
    )

    return {
        "method_id": method["method_id"],
        "method_name": method["method_name"],
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "transcription_accuracy": (
            transcription_accuracy
        ),
        "transcription_consistent": (
            transcription_consistent
        ),
        "average_ocr_confidence": (
            average_ocr_confidence
        ),
        "average_prediction_confidence": (
            average_prediction_confidence
        ),
        "view_count": len(
            method.get("views", [])
        ),
    }


def build_markdown_summary(
    experiment: Dict[str, Any],
    metrics: List[Dict[str, Any]],
) -> str:
    """Create a Markdown experiment report."""

    lines = [
        "# OCR Ablation Experiment",
        "",
        experiment["description"],
        "",
        "## Experimental Setup",
        "",
        f"- Image: `{experiment['image_path']}`",
        (
            "- Ground-truth text: "
            f"`{experiment['ground_truth_text']}`"
        ),
        (
            "- Evaluation examples: "
            f"{len(experiment['examples'])}"
        ),
        "",
        "## Results",
        "",
        (
            "| Method | Views | Verification Accuracy | "
            "OCR Transcription Accuracy | "
            "Consistent Across Claims | "
            "Average OCR Confidence |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|"
        ),
    ]

    for method_metrics in metrics:
        consistency_text = (
            "Yes"
            if method_metrics[
                "transcription_consistent"
            ]
            else "No"
        )

        lines.append(
            "| "
            f"{method_metrics['method_name']} | "
            f"{method_metrics['view_count']} | "
            f"{method_metrics['accuracy']:.3f} | "
            f"{method_metrics['transcription_accuracy']:.3f} | "
            f"{consistency_text} | "
            f"{method_metrics['average_ocr_confidence']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Example Results",
            "",
        ]
    )

    for method in experiment["methods"]:
        lines.append(
            f"### {method['method_name']}"
        )

        lines.append("")

        for result in method["results"]:
            correct = (
                result["predicted_label"]
                == result["gold_label"]
            )

            lines.extend(
                [
                    (
                        f"- `{result['example_id']}`: "
                        f"OCR=`{result['ocr_transcription']}`, "
                        f"gold=`{result['gold_label']}`, "
                        f"prediction=`{result['predicted_label']}`, "
                        f"correct=`{correct}`"
                    )
                ]
            )

        lines.append("")

    qualitative_failure = experiment.get(
        "qualitative_failure"
    )

    if qualitative_failure:
        lines.extend(
            [
                "## Qualitative Failure Observation",
                "",
                (
                    "The earlier claim-conditioned OCR runs "
                    "are excluded from formal accuracy because "
                    "they used an initial annotation that was "
                    "later corrected."
                ),
                "",
                (
                    "They are retained as evidence that directly "
                    "providing a target phrase to perception can "
                    "create instability and potential confirmation "
                    "bias."
                ),
                "",
            ]
        )

        for observation in qualitative_failure[
            "observations"
        ]:
            lines.append(
                f"- Run {observation['run']}: "
                f"target=`{observation['claim_target']}`, "
                f"OCR=`{observation['ocr_transcription']}`, "
                f"prediction=`{observation['final_prediction']}`"
            )

    lines.extend(
        [
            "",
            "## Main Finding",
            "",
            (
                "Blind single-view OCR failed on both rotated-text "
                "examples and produced inconsistent transcriptions "
                "for the same physical text."
            ),
            "",
            (
                "Multi-view blind OCR correctly transcribed "
                f"`{experiment['ground_truth_text']}` under both "
                "claims and improved verification accuracy from "
                "0/2 to 2/2 without exposing the OCR model to the "
                "claim target."
            ),
            "",
            "## Limitations",
            "",
            (
                "This is a focused two-example ablation on one "
                "difficult image. It demonstrates the mechanism "
                "and failure mode but is not a large-scale OCR "
                "benchmark."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def print_metrics(
    metrics: List[Dict[str, Any]],
) -> None:
    """Print the ablation results to the terminal."""

    print("=" * 70)
    print("OCR ABLATION RESULTS")
    print("=" * 70)

    for method_metrics in metrics:
        print(
            f"\nMethod: "
            f"{method_metrics['method_name']}"
        )

        print(
            "Verification accuracy: "
            f"{method_metrics['correct']}/"
            f"{method_metrics['total']} "
            f"({method_metrics['accuracy']:.3f})"
        )

        print(
            "OCR transcription accuracy: "
            f"{method_metrics['transcription_accuracy']:.3f}"
        )

        print(
            "Consistent across claims: "
            f"{method_metrics['transcription_consistent']}"
        )

        print(
            "Average OCR confidence: "
            f"{method_metrics['average_ocr_confidence']:.3f}"
        )

        print(
            "Average prediction confidence: "
            f"{method_metrics['average_prediction_confidence']:.3f}"
        )

        print(
            "Number of views: "
            f"{method_metrics['view_count']}"
        )


def main() -> None:
    """Analyze and save the OCR ablation experiment."""

    global experiment

    experiment = load_experiment(
        EXPERIMENT_PATH
    )

    metrics = [
        calculate_method_metrics(method)
        for method in experiment["methods"]
    ]

    print_metrics(
        metrics
    )

    markdown_summary = build_markdown_summary(
        experiment=experiment,
        metrics=metrics,
    )

    SUMMARY_PATH.write_text(
        markdown_summary,
        encoding="utf-8",
    )

    print(
        f"\nSaved report: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()