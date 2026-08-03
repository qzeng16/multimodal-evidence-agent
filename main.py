import json
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from src.evaluator import (
    evaluate_predictions,
    print_evaluation_report,
    save_metrics,
    save_predictions,
)
from src.image_inspector import inspect_image
from src.image_loader import load_image_metadata
from src.ocr_tool import extract_text_from_image
from src.schemas import (
    OCRExtraction,
    ToolRoutingDecision,
    VerificationInput,
    VerificationResult,
    VisualInspection,
)
from src.tool_router import route_tools
from src.verifier import verify_claim


DATA_PATH = Path(
    "data/samples.jsonl"
)

PREDICTIONS_PATH = Path(
    "outputs/predictions.jsonl"
)

METRICS_PATH = Path(
    "outputs/metrics.json"
)


def load_examples(
    path: Path,
) -> List[VerificationInput]:
    """Load and validate examples from a JSONL file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    examples: List[
        VerificationInput
    ] = []

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
                raw_example = json.loads(
                    line
                )

                example = (
                    VerificationInput.model_validate(
                        raw_example
                    )
                )

                examples.append(
                    example
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {error}"
                ) from error

            except ValidationError as error:
                raise ValueError(
                    f"Invalid example on line "
                    f"{line_number}: {error}"
                ) from error

    return examples


def print_routing_decision(
    decision: ToolRoutingDecision,
) -> None:
    """Print the tools selected by the router."""

    print("\nTool routing decision:")

    print(
        "Use Image Inspector: "
        f"{decision.use_image_inspector}"
    )

    print(
        "Use OCR: "
        f"{decision.use_ocr}"
    )

    print(
        "Reasoning: "
        f"{decision.reasoning}"
    )

    print(
        "Matched keywords: "
        f"{decision.matched_keywords}"
    )

    print(
        "Text targets: "
        f"{decision.text_targets}"
    )


def print_visual_inspection(
    inspection: VisualInspection,
) -> None:
    """Print Image Inspector evidence."""

    print("\nVisual inspection:")

    print(
        f"Scene: "
        f"{inspection.scene_description}"
    )

    print("\nSupporting observations:")

    if inspection.supporting_observations:
        for observation in (
            inspection.supporting_observations
        ):
            print(
                f"- {observation}"
            )
    else:
        print("- None")

    print("\nContradicting observations:")

    if inspection.contradicting_observations:
        for observation in (
            inspection.contradicting_observations
        ):
            print(
                f"- {observation}"
            )
    else:
        print("- None")

    print("\nVisible text:")

    if inspection.visible_text:
        for visible_text in (
            inspection.visible_text
        ):
            print(
                f"- {visible_text}"
            )
    else:
        print("- None")

    print("\nUncertainty notes:")

    if inspection.uncertainty_notes:
        for note in (
            inspection.uncertainty_notes
        ):
            print(
                f"- {note}"
            )
    else:
        print("- None")


def print_ocr_extraction(
    extraction: OCRExtraction,
) -> None:
    """Print structured OCR evidence."""

    print("\nOCR extraction:")

    if extraction.detected_text:
        for index, text_span in enumerate(
            extraction.detected_text,
            start=1,
        ):
            print(
                f"{index}. Text: "
                f"{text_span.text}"
            )

            print(
                f"   Location: "
                f"{text_span.location}"
            )

            print(
                f"   Confidence: "
                f"{text_span.confidence:.2f}"
            )

            print(
                f"   Relevance: "
                f"{text_span.relevance_to_claim:.2f}"
            )
    else:
        print("- No readable text detected")

    print("\nTarget matches:")

    if extraction.target_matches:
        for match in (
            extraction.target_matches
        ):
            print(
                f"- {match}"
            )
    else:
        print("- None")

    print("\nTarget mismatches:")

    if extraction.target_mismatches:
        for mismatch in (
            extraction.target_mismatches
        ):
            print(
                f"- {mismatch}"
            )
    else:
        print("- None")

    print("\nOCR uncertainty notes:")

    if extraction.uncertainty_notes:
        for note in (
            extraction.uncertainty_notes
        ):
            print(
                f"- {note}"
            )
    else:
        print("- None")


def print_verification_result(
    result: VerificationResult,
) -> None:
    """Print the final verification result."""

    print("\nFinal verification result:")

    print(
        f"Label: {result.label}"
    )

    print(
        f"Confidence: "
        f"{result.confidence:.2f}"
    )

    print(
        f"Rationale: "
        f"{result.rationale}"
    )

    print("\nSelected evidence:")

    if result.evidence:
        for evidence in (
            result.evidence
        ):
            print(
                f"- [{evidence.modality}] "
                f"{evidence.content}"
            )
    else:
        print("- None")

    print("\nTool trace:")

    if result.tool_trace:
        for index, tool_call in enumerate(
            result.tool_trace,
            start=1,
        ):
            print(
                f"{index}. "
                f"{tool_call.tool_name}"
            )

            print(
                f"   Output: "
                f"{tool_call.tool_output_summary}"
            )
    else:
        print("- None")


def process_example(
    example: VerificationInput,
) -> Optional[VerificationResult]:
    """Run the dynamic tool-using pipeline for one example."""

    result: Optional[
        VerificationResult
    ] = None

    print(
        f"Example ID: "
        f"{example.example_id}"
    )

    print(
        f"Claim: "
        f"{example.claim}"
    )

    print(
        f"Image: "
        f"{example.image_path}"
    )

    print(
        f"Context: "
        f"{example.context}"
    )

    print(
        f"Gold label: "
        f"{example.gold_label}"
    )

    try:
        metadata = load_image_metadata(
            example.image_path
        )

        print("\nImage metadata:")

        print(
            f"File name: "
            f"{metadata['file_name']}"
        )

        print(
            f"Format: "
            f"{metadata['format']}"
        )

        print(
            f"Width: "
            f"{metadata['width']} pixels"
        )

        print(
            f"Height: "
            f"{metadata['height']} pixels"
        )

        print(
            f"Color mode: "
            f"{metadata['mode']}"
        )

        routing_decision = route_tools(
            example.claim
        )

        print_routing_decision(
            routing_decision
        )

        if not routing_decision.use_image_inspector:
            raise RuntimeError(
                "The current pipeline requires "
                "the Image Inspector."
            )

        print(
            "\nCalling Image Inspector..."
        )

        inspection = inspect_image(
            image_path=example.image_path,
            claim=example.claim,
            context=example.context,
        )

        print_visual_inspection(
            inspection
        )

        ocr_extraction: Optional[
            OCRExtraction
        ] = None

        if routing_decision.use_ocr:
            print(
                "\nCalling OCR Tool..."
            )

            ocr_extraction = (
                extract_text_from_image(
                    image_path=example.image_path,
                    claim=example.claim,
                    text_targets=(
                        routing_decision.text_targets
                    ),
                )
            )

            print_ocr_extraction(
                ocr_extraction
            )
        else:
            print(
                "\nOCR Tool skipped by router."
            )

        print(
            "\nCalling Verification Reasoner..."
        )

        result = verify_claim(
            example=example,
            inspection=inspection,
            routing_decision=routing_decision,
            ocr_extraction=ocr_extraction,
        )

        print_verification_result(
            result
        )

        if example.gold_label is not None:
            is_correct = (
                result.label
                == example.gold_label
            )

            print("\nEvaluation:")

            print(
                "Prediction matches gold label: "
                f"{is_correct}"
            )

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as error:
        print(
            f"\nProcessing error: "
            f"{error}"
        )

    except Exception as error:
        print(
            "\nUnexpected API error: "
            f"{type(error).__name__}: "
            f"{error}"
        )

    print(
        "\n" + "-" * 60
    )

    return result


def main() -> None:
    """Run the dynamic multimodal verification pipeline."""

    examples = load_examples(
        DATA_PATH
    )

    print(
        f"Loaded {len(examples)} "
        "verification example(s).\n"
    )

    results: List[
        VerificationResult
    ] = []

    for example in examples:
        result = process_example(
            example
        )

        if result is not None:
            results.append(
                result
            )

    save_predictions(
        examples=examples,
        results=results,
        output_path=PREDICTIONS_PATH,
    )

    metrics = evaluate_predictions(
        examples=examples,
        results=results,
    )

    save_metrics(
        metrics=metrics,
        output_path=METRICS_PATH,
    )

    print_evaluation_report(
        metrics
    )

    print("\nSaved files:")

    print(
        f"- {PREDICTIONS_PATH}"
    )

    print(
        f"- {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()