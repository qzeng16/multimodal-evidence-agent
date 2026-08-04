from time import perf_counter
from typing import Optional

from pydantic import BaseModel, Field

from src.image_inspector import inspect_image
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


class PipelineLatency(BaseModel):
    """Execution time for each pipeline stage."""

    routing_seconds: float = Field(
        ge=0.0
    )

    image_inspector_seconds: float = Field(
        ge=0.0
    )

    ocr_seconds: float = Field(
        ge=0.0
    )

    verification_reasoner_seconds: float = Field(
        ge=0.0
    )

    total_seconds: float = Field(
        ge=0.0
    )


class PipelineExecution(BaseModel):
    """Complete output from one pipeline execution."""

    routing_decision: ToolRoutingDecision
    inspection: VisualInspection

    ocr_extraction: Optional[
        OCRExtraction
    ] = None

    result: VerificationResult
    latency: PipelineLatency

    model_call_count: int = Field(
        ge=0
    )


def run_verification(
    example: VerificationInput,
) -> PipelineExecution:
    """
    Execute the complete multimodal verification pipeline.

    This function does not print to the terminal, evaluate a
    dataset, or write output files. It can therefore be reused
    by the CLI, API, tests, and future interfaces.
    """

    total_start = perf_counter()

    routing_start = perf_counter()

    routing_decision = route_tools(
        example.claim
    )

    routing_seconds = (
        perf_counter()
        - routing_start
    )

    if not routing_decision.use_image_inspector:
        raise RuntimeError(
            "The current verification pipeline "
            "requires the Image Inspector."
        )

    inspection_start = perf_counter()

    inspection = inspect_image(
        image_path=example.image_path,
        claim=example.claim,
        context=example.context,
    )

    image_inspector_seconds = (
        perf_counter()
        - inspection_start
    )

    ocr_extraction: Optional[
        OCRExtraction
    ] = None

    ocr_seconds = 0.0

    if routing_decision.use_ocr:
        ocr_start = perf_counter()

        ocr_extraction = (
            extract_text_from_image(
                image_path=example.image_path,
                claim=example.claim,
                text_targets=(
                    routing_decision.text_targets
                ),
            )
        )

        ocr_seconds = (
            perf_counter()
            - ocr_start
        )

    verifier_start = perf_counter()

    result = verify_claim(
        example=example,
        inspection=inspection,
        routing_decision=routing_decision,
        ocr_extraction=ocr_extraction,
    )

    verification_reasoner_seconds = (
        perf_counter()
        - verifier_start
    )

    total_seconds = (
        perf_counter()
        - total_start
    )

    model_call_count = (
        2
        + int(
            routing_decision.use_ocr
        )
    )

    latency = PipelineLatency(
        routing_seconds=routing_seconds,
        image_inspector_seconds=(
            image_inspector_seconds
        ),
        ocr_seconds=ocr_seconds,
        verification_reasoner_seconds=(
            verification_reasoner_seconds
        ),
        total_seconds=total_seconds,
    )

    return PipelineExecution(
        routing_decision=routing_decision,
        inspection=inspection,
        ocr_extraction=ocr_extraction,
        result=result,
        latency=latency,
        model_call_count=model_call_count,
    )