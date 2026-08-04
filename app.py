import json
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from src.pipeline import (
    PipelineExecution,
    PipelineLatency,
    run_verification,
)
from src.schemas import (
    EvidenceItem,
    ToolCallRecord,
    ToolRoutingDecision,
    VerificationInput,
    VerificationLabel,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "samples.jsonl"
)

ALLOWED_IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "images"
).resolve()


app = FastAPI(
    title=(
        "Multimodal Evidence "
        "Verification Agent"
    ),
    description=(
        "Verify textual claims against image evidence "
        "using dynamic tool routing, visual inspection, "
        "blind OCR, and structured reasoning."
    ),
    version="1.0.0",
)


class VerificationRequest(BaseModel):
    """Request body for a new verification task."""

    claim: str = Field(
        min_length=1
    )

    image_path: str = Field(
        min_length=1
    )

    context: Optional[str] = None
    example_id: Optional[str] = None


class VerificationResponse(BaseModel):
    """Public API response for one verification task."""

    example_id: str
    label: VerificationLabel

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    rationale: str
    used_ocr: bool

    routing_decision: ToolRoutingDecision

    evidence: List[
        EvidenceItem
    ] = Field(
        default_factory=list
    )

    tool_trace: List[
        ToolCallRecord
    ] = Field(
        default_factory=list
    )

    latency: PipelineLatency

    model_call_count: int = Field(
        ge=0
    )


def load_examples(
    path: Path,
) -> List[VerificationInput]:
    """Load verification examples from JSONL."""

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

            examples.append(
                example
            )

    return examples


def validate_image_path(
    image_path: str,
) -> str:
    """
    Validate that an API image path is inside data/images.

    This prevents callers from asking the server to read
    arbitrary local files.
    """

    candidate_path = Path(
        image_path
    ).expanduser()

    if not candidate_path.is_absolute():
        candidate_path = (
            PROJECT_ROOT
            / candidate_path
        )

    resolved_path = (
        candidate_path.resolve()
    )

    try:
        relative_to_image_root = (
            resolved_path.relative_to(
                ALLOWED_IMAGE_ROOT
            )
        )

    except ValueError as error:
        raise ValueError(
            "image_path must point to a file "
            "inside data/images."
        ) from error

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            f"Image path is not a file: "
            f"{image_path}"
        )

    project_relative_path = (
        Path("data")
        / "images"
        / relative_to_image_root
    )

    return (
        project_relative_path.as_posix()
    )


def build_response(
    execution: PipelineExecution,
) -> VerificationResponse:
    """Convert an internal pipeline result into an API response."""

    result = execution.result

    return VerificationResponse(
        example_id=result.example_id,
        label=result.label,
        confidence=result.confidence,
        rationale=result.rationale,
        used_ocr=(
            execution.routing_decision.use_ocr
        ),
        routing_decision=(
            execution.routing_decision
        ),
        evidence=result.evidence,
        tool_trace=result.tool_trace,
        latency=execution.latency,
        model_call_count=(
            execution.model_call_count
        ),
    )


def execute_example(
    example: VerificationInput,
) -> VerificationResponse:
    """Run one example and translate errors into HTTP errors."""

    try:
        execution = run_verification(
            example
        )

        return build_response(
            execution
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        ) from error


@app.get(
    "/health",
    tags=["system"],
)
def health() -> dict:
    """Return API health status without calling a model."""

    return {
        "status": "ok",
        "service": (
            "multimodal-evidence-agent"
        ),
        "version": "1.0.0",
    }


@app.post(
    "/verify",
    response_model=VerificationResponse,
    tags=["verification"],
)
def verify(
    request: VerificationRequest,
) -> VerificationResponse:
    """Verify a custom claim against a local image."""

    try:
        validated_image_path = (
            validate_image_path(
                request.image_path
            )
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    example_id = (
        request.example_id
        or f"api_{uuid4().hex[:12]}"
    )

    example = VerificationInput(
        example_id=example_id,
        claim=request.claim,
        image_path=validated_image_path,
        context=request.context,
    )

    return execute_example(
        example
    )


@app.post(
    "/verify-example/{example_id}",
    response_model=VerificationResponse,
    tags=["verification"],
)
def verify_example(
    example_id: str,
) -> VerificationResponse:
    """Run one example from data/samples.jsonl."""

    try:
        examples = load_examples(
            DATA_PATH
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    selected_example = next(
        (
            example
            for example in examples
            if example.example_id
            == example_id
        ),
        None,
    )

    if selected_example is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Example ID not found: "
                f"{example_id}"
            ),
        )

    try:
        validated_image_path = (
            validate_image_path(
                selected_example.image_path
            )
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    selected_example = (
        selected_example.model_copy(
            update={
                "image_path": (
                    validated_image_path
                )
            }
        )
    )

    return execute_example(
        selected_example
    )