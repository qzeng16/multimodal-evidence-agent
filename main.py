import argparse
from pathlib import Path
from typing import List, Optional

from src.dataset import (
    load_examples,
    select_examples,
)
from src.evaluator import (
    evaluate_predictions,
    print_evaluation_report,
    save_metrics,
    save_predictions,
)
from src.image_loader import load_image_metadata
from src.pipeline import (
    PipelineExecution,
    run_verification,
)
from src.schemas import (
    OCRExtraction,
    ToolRoutingDecision,
    VerificationInput,
    VerificationResult,
    VisualInspection,
)


DATA_PATH = Path("data/samples.jsonl")
PREDICTIONS_PATH = Path("outputs/predictions.jsonl")
METRICS_PATH = Path("outputs/metrics.json")

MODEL_TOOL_NAMES = {
    "image_inspector",
    "ocr_tool",
    "verification_reasoner",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the multimodal evidence verification agent."
        )
    )

    parser.add_argument(
        "--example-id",
        type=str,
        default=None,
        help=(
            "Run only one example, such as sample_004. "
            "When omitted, all examples are processed."
        ),
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Disable the Image Inspector and OCR disk cache. "
            "Useful for uncached latency measurements."
        ),
    )

    return parser.parse_args()


def count_logical_model_calls(
    execution: PipelineExecution,
) -> int:
    """
    Count model stages in the logical tool path.

    This count does not change when a stage is served from cache.
    """

    return sum(
        tool_call.tool_name in MODEL_TOOL_NAMES
        for tool_call in execution.result.tool_trace
    )


def calculate_cache_hit_rate(
    execution: PipelineExecution,
) -> float:
    """Calculate the cache hit rate for one execution."""

    cache_lookups = (
        execution.cache_hits
        + execution.cache_misses
    )

    if cache_lookups == 0:
        return 0.0

    return (
        execution.cache_hits
        / cache_lookups
    )


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
    """Print evidence returned by the Image Inspector."""

    print("\nVisual inspection:")

    print(
        f"Scene: {inspection.scene_description}"
    )

    print("\nSupporting observations:")

    if inspection.supporting_observations:
        for observation in (
            inspection.supporting_observations
        ):
            print(f"- {observation}")
    else:
        print("- None")

    print("\nContradicting observations:")

    if inspection.contradicting_observations:
        for observation in (
            inspection.contradicting_observations
        ):
            print(f"- {observation}")
    else:
        print("- None")

    print("\nVisible text:")

    if inspection.visible_text:
        for visible_text in inspection.visible_text:
            print(f"- {visible_text}")
    else:
        print("- None")

    print("\nUncertainty notes:")

    if inspection.uncertainty_notes:
        for note in inspection.uncertainty_notes:
            print(f"- {note}")
    else:
        print("- None")


def print_ocr_extraction(
    extraction: OCRExtraction,
) -> None:
    """Print evidence returned by the OCR tool."""

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
        for match in extraction.target_matches:
            print(f"- {match}")
    else:
        print("- None")

    print("\nTarget mismatches:")

    if extraction.target_mismatches:
        for mismatch in extraction.target_mismatches:
            print(f"- {mismatch}")
    else:
        print("- None")

    print("\nOCR uncertainty notes:")

    if extraction.uncertainty_notes:
        for note in extraction.uncertainty_notes:
            print(f"- {note}")
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
        for evidence in result.evidence:
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


def print_pipeline_metrics(
    execution: PipelineExecution,
) -> None:
    """Print latency, cache, and model-call information."""

    latency = execution.latency

    cache_hit_rate = calculate_cache_hit_rate(
        execution
    )

    logical_model_calls = (
        count_logical_model_calls(
            execution
        )
    )

    print("\nPipeline performance:")

    print(
        "Routing latency: "
        f"{latency.routing_seconds:.3f} seconds"
    )

    print(
        "Image Inspector latency: "
        f"{latency.image_inspector_seconds:.3f} seconds"
    )

    print(
        "OCR latency: "
        f"{latency.ocr_seconds:.3f} seconds"
    )

    print(
        "Verification Reasoner latency: "
        f"{latency.verification_reasoner_seconds:.3f} seconds"
    )

    print(
        "Total latency: "
        f"{latency.total_seconds:.3f} seconds"
    )

    print(
        "Cache enabled: "
        f"{execution.cache_enabled}"
    )

    print(
        "Cache hits: "
        f"{execution.cache_hits}"
    )

    print(
        "Cache misses: "
        f"{execution.cache_misses}"
    )

    print(
        "Cache hit rate: "
        f"{cache_hit_rate:.3f}"
    )

    print(
        "Logical model path calls: "
        f"{logical_model_calls}"
    )

    print(
        "Actual model API calls: "
        f"{execution.model_call_count}"
    )


def print_runtime_cache_report(
    executions: List[PipelineExecution],
) -> None:
    """Print aggregate runtime and cache statistics."""

    print("\n" + "=" * 60)
    print("RUNTIME CACHE METRICS")
    print("=" * 60)

    if not executions:
        print("Executions: 0")
        return

    total_cache_hits = sum(
        execution.cache_hits
        for execution in executions
    )

    total_cache_misses = sum(
        execution.cache_misses
        for execution in executions
    )

    total_cache_lookups = (
        total_cache_hits
        + total_cache_misses
    )

    if total_cache_lookups == 0:
        cache_hit_rate = 0.0
    else:
        cache_hit_rate = (
            total_cache_hits
            / total_cache_lookups
        )

    total_actual_model_calls = sum(
        execution.model_call_count
        for execution in executions
    )

    total_logical_model_calls = sum(
        count_logical_model_calls(
            execution
        )
        for execution in executions
    )

    average_actual_model_calls = (
        total_actual_model_calls
        / len(executions)
    )

    average_logical_model_calls = (
        total_logical_model_calls
        / len(executions)
    )

    average_total_latency = (
        sum(
            execution.latency.total_seconds
            for execution in executions
        )
        / len(executions)
    )

    avoided_model_calls = (
        total_logical_model_calls
        - total_actual_model_calls
    )

    print(
        f"Executions: {len(executions)}"
    )

    print(
        f"Cache lookups: "
        f"{total_cache_lookups}"
    )

    print(
        f"Cache hits: "
        f"{total_cache_hits}"
    )

    print(
        f"Cache misses: "
        f"{total_cache_misses}"
    )

    print(
        f"Cache hit rate: "
        f"{cache_hit_rate:.3f}"
    )

    print(
        f"Total logical model path calls: "
        f"{total_logical_model_calls}"
    )

    print(
        f"Total actual model API calls: "
        f"{total_actual_model_calls}"
    )

    print(
        f"Model calls avoided by cache: "
        f"{avoided_model_calls}"
    )

    print(
        f"Average logical model path calls: "
        f"{average_logical_model_calls:.3f}"
    )

    print(
        f"Average actual model API calls: "
        f"{average_actual_model_calls:.3f}"
    )

    print(
        f"Average total latency: "
        f"{average_total_latency:.3f} seconds"
    )


def process_example(
    example: VerificationInput,
    use_cache: bool,
) -> Optional[PipelineExecution]:
    """Run the reusable pipeline for one example."""

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

    print(
        f"Category: "
        f"{example.category}"
    )

    print(
        f"Expected use OCR: "
        f"{example.expected_use_ocr}"
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

        print(
            "\nRunning verification pipeline..."
        )

        execution = run_verification(
            example,
            use_cache=use_cache,
        )

        print_routing_decision(
            execution.routing_decision
        )

        print_visual_inspection(
            execution.inspection
        )

        if execution.ocr_extraction is not None:
            print_ocr_extraction(
                execution.ocr_extraction
            )
        else:
            print(
                "\nOCR Tool skipped by router."
            )

        print_verification_result(
            execution.result
        )

        print_pipeline_metrics(
            execution
        )

        if example.gold_label is not None:
            is_correct = (
                execution.result.label
                == example.gold_label
            )

            print("\nEvaluation:")

            print(
                "Prediction matches gold label: "
                f"{is_correct}"
            )

        if example.expected_use_ocr is not None:
            route_correct = (
                execution.routing_decision.use_ocr
                == example.expected_use_ocr
            )

            print(
                "OCR routing matches annotation: "
                f"{route_correct}"
            )

        return execution

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

    return None


def main() -> None:
    """Run the multimodal verification CLI."""

    arguments = parse_arguments()

    use_cache = not arguments.no_cache

    all_examples = load_examples(
        DATA_PATH
    )

    examples = select_examples(
        examples=all_examples,
        example_id=arguments.example_id,
    )

    print(
        f"Loaded {len(examples)} "
        "verification example(s).\n"
    )

    print(
        f"Disk cache enabled: {use_cache}\n"
    )

    results: List[VerificationResult] = []

    executions: List[
        PipelineExecution
    ] = []

    for index, example in enumerate(
        examples,
        start=1,
    ):
        if index > 1:
            print(
                "\n" + "-" * 60
            )

        execution = process_example(
            example=example,
            use_cache=use_cache,
        )

        if execution is not None:
            executions.append(
                execution
            )

            results.append(
                execution.result
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

    print_runtime_cache_report(
        executions
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