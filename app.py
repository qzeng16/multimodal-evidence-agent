import os
import secrets
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    status,
)
from pydantic import BaseModel, Field

from src.dataset import (
    find_example_by_id,
    load_examples,
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


load_dotenv()


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

SERVICE_API_KEY_ENV = (
    "SERVICE_API_KEY"
)

MODEL_TOOL_NAMES = {
    "image_inspector",
    "ocr_tool",
    "verification_reasoner",
}


app = FastAPI(
    title=(
        "Multimodal Evidence "
        "Verification Agent"
    ),
    description=(
        "Verify textual claims against image evidence "
        "using dynamic tool routing, visual inspection, "
        "blind OCR, structured reasoning, and disk caching."
    ),
    version="1.1.0",
)


class VerificationRequest(BaseModel):
    """Request body for a verification task."""

    claim: str = Field(
        min_length=1
    )

    image_path: str = Field(
        min_length=1
    )

    context: Optional[str] = None
    example_id: Optional[str] = None

    use_cache: bool = True


class VerificationResponse(BaseModel):
    """Public response for one verification task."""

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

    cache_enabled: bool
    cache_hits: int = Field(ge=0)
    cache_misses: int = Field(ge=0)

    cache_hit_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    logical_model_call_count: int = Field(
        ge=0
    )

    actual_model_call_count: int = Field(
        ge=0
    )

    model_call_count: int = Field(
        ge=0
    )


def get_service_api_key() -> Optional[str]:
    """Return the configured API key, if authentication is enabled."""

    configured_key = os.getenv(
        SERVICE_API_KEY_ENV
    )

    if configured_key is None:
        return None

    configured_key = configured_key.strip()

    return configured_key or None


def require_service_api_key(
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key",
        description=(
            "Service access key required when "
            "SERVICE_API_KEY is configured."
        ),
    ),
) -> None:
    """Require a valid service API key when one is configured."""

    expected_key = get_service_api_key()

    if expected_key is None:
        return

    if (
        x_api_key is None
        or not secrets.compare_digest(
            x_api_key,
            expected_key,
        )
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail=(
                "Invalid or missing API key."
            ),
            headers={
                "WWW-Authenticate": "ApiKey"
            },
        )


def calculate_cache_hit_rate(
    execution: PipelineExecution,
) -> float:
    """Calculate cache hit rate for one request."""

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


def count_logical_model_calls(
    execution: PipelineExecution,
) -> int:
    """Count model stages in the logical tool path."""

    return sum(
        tool_call.tool_name in MODEL_TOOL_NAMES
        for tool_call in execution.result.tool_trace
    )


def validate_image_path(
    image_path: str,
) -> str:
    """
    Validate that an image is inside data/images.

    This prevents callers from reading arbitrary local files.
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
    """Convert an internal execution into an API response."""

    result = execution.result

    cache_hit_rate = calculate_cache_hit_rate(
        execution
    )

    logical_model_call_count = (
        count_logical_model_calls(
            execution
        )
    )

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
        cache_enabled=(
            execution.cache_enabled
        ),
        cache_hits=execution.cache_hits,
        cache_misses=execution.cache_misses,
        cache_hit_rate=cache_hit_rate,
        logical_model_call_count=(
            logical_model_call_count
        ),
        actual_model_call_count=(
            execution.model_call_count
        ),
        model_call_count=(
            execution.model_call_count
        ),
    )


def execute_example(
    example: VerificationInput,
    use_cache: bool,
) -> VerificationResponse:
    """Run one example and translate errors into HTTP errors."""

    try:
        execution = run_verification(
            example,
            use_cache=use_cache,
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
    """Return API status without calling a model."""

    return {
        "status": "ok",
        "service": (
            "multimodal-evidence-agent"
        ),
        "version": "1.1.0",
        "disk_cache_supported": True,
        "api_key_authentication_enabled": (
            get_service_api_key()
            is not None
        ),
    }


@app.post(
    "/verify",
    response_model=VerificationResponse,
    tags=["verification"],
    dependencies=[
        Depends(
            require_service_api_key
        )
    ],
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
        example=example,
        use_cache=request.use_cache,
    )


@app.post(
    "/verify-example/{example_id}",
    response_model=VerificationResponse,
    tags=["verification"],
    dependencies=[
        Depends(
            require_service_api_key
        )
    ],
)
def verify_example(
    example_id: str,
    use_cache: bool = True,
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

    selected_example = find_example_by_id(
        examples=examples,
        example_id=example_id,
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
        example=selected_example,
        use_cache=use_cache,
    )
