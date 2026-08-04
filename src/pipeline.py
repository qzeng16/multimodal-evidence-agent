from pathlib import Path
from time import perf_counter
from typing import Optional

from pydantic import BaseModel, Field

from src.cache import (
    build_model_cache_key,
    file_sha256,
    load_cached_model,
    optional_file_sha256,
    save_cached_model,
)
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


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

OCR_REGIONS_CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "ocr_regions.json"
)

IMAGE_INSPECTOR_TOOL_NAME = (
    "image_inspector"
)

IMAGE_INSPECTOR_CACHE_VERSION = (
    "claim-conditioned-v1"
)

OCR_TOOL_NAME = "ocr_tool"

OCR_CACHE_VERSION = (
    "blind-multiview-v1"
)


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

    cache_hits: int = Field(
        default=0,
        ge=0,
    )

    cache_misses: int = Field(
        default=0,
        ge=0,
    )

    cache_enabled: bool = True


def run_verification(
    example: VerificationInput,
    use_cache: bool = True,
    cache_root: Optional[Path] = None,
) -> PipelineExecution:
    """
    Execute the multimodal verification pipeline.

    Image Inspector and OCR results may be loaded from
    the disk cache. The Verification Reasoner always runs.
    """

    total_start = perf_counter()

    model_call_count = 0
    cache_hits = 0
    cache_misses = 0

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

    image_hash: Optional[str] = None

    if use_cache:
        image_hash = file_sha256(
            example.image_path
        )

    inspection_start = perf_counter()

    inspection: Optional[
        VisualInspection
    ] = None

    inspection_cache_key: Optional[
        str
    ] = None

    if use_cache:
        if image_hash is None:
            raise RuntimeError(
                "Image hash was not created "
                "while caching was enabled."
            )

        inspection_cache_key = (
            build_model_cache_key(
                tool_name=(
                    IMAGE_INSPECTOR_TOOL_NAME
                ),
                tool_version=(
                    IMAGE_INSPECTOR_CACHE_VERSION
                ),
                image_sha256=image_hash,
                inputs={
                    "claim": example.claim,
                    "context": example.context,
                },
            )
        )

        inspection = load_cached_model(
            tool_name=(
                IMAGE_INSPECTOR_TOOL_NAME
            ),
            cache_key=(
                inspection_cache_key
            ),
            model_type=VisualInspection,
            cache_root=cache_root,
        )

        if inspection is None:
            cache_misses += 1
        else:
            cache_hits += 1

    if inspection is None:
        inspection = inspect_image(
            image_path=example.image_path,
            claim=example.claim,
            context=example.context,
        )

        model_call_count += 1

        if (
            use_cache
            and inspection_cache_key
            is not None
        ):
            save_cached_model(
                tool_name=(
                    IMAGE_INSPECTOR_TOOL_NAME
                ),
                tool_version=(
                    IMAGE_INSPECTOR_CACHE_VERSION
                ),
                cache_key=(
                    inspection_cache_key
                ),
                model=inspection,
                cache_root=cache_root,
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

        ocr_cache_key: Optional[
            str
        ] = None

        if use_cache:
            if image_hash is None:
                raise RuntimeError(
                    "Image hash was not created "
                    "while caching was enabled."
                )

            ocr_config_hash = (
                optional_file_sha256(
                    str(
                        OCR_REGIONS_CONFIG_PATH
                    )
                )
            )

            ocr_cache_key = (
                build_model_cache_key(
                    tool_name=OCR_TOOL_NAME,
                    tool_version=(
                        OCR_CACHE_VERSION
                    ),
                    image_sha256=image_hash,
                    inputs={
                        "text_targets": list(
                            routing_decision
                            .text_targets
                        ),
                        "ocr_regions_config_sha256": (
                            ocr_config_hash
                        ),
                    },
                )
            )

            ocr_extraction = (
                load_cached_model(
                    tool_name=OCR_TOOL_NAME,
                    cache_key=(
                        ocr_cache_key
                    ),
                    model_type=(
                        OCRExtraction
                    ),
                    cache_root=cache_root,
                )
            )

            if ocr_extraction is None:
                cache_misses += 1
            else:
                cache_hits += 1

        if ocr_extraction is None:
            ocr_extraction = (
                extract_text_from_image(
                    image_path=(
                        example.image_path
                    ),
                    claim=example.claim,
                    text_targets=(
                        routing_decision
                        .text_targets
                    ),
                )
            )

            model_call_count += 1

            if (
                use_cache
                and ocr_cache_key
                is not None
            ):
                save_cached_model(
                    tool_name=OCR_TOOL_NAME,
                    tool_version=(
                        OCR_CACHE_VERSION
                    ),
                    cache_key=(
                        ocr_cache_key
                    ),
                    model=ocr_extraction,
                    cache_root=cache_root,
                )

        ocr_seconds = (
            perf_counter()
            - ocr_start
        )

    verifier_start = perf_counter()

    result = verify_claim(
        example=example,
        inspection=inspection,
        routing_decision=(
            routing_decision
        ),
        ocr_extraction=(
            ocr_extraction
        ),
    )

    model_call_count += 1

    verification_reasoner_seconds = (
        perf_counter()
        - verifier_start
    )

    total_seconds = (
        perf_counter()
        - total_start
    )

    latency = PipelineLatency(
        routing_seconds=(
            routing_seconds
        ),
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
        routing_decision=(
            routing_decision
        ),
        inspection=inspection,
        ocr_extraction=(
            ocr_extraction
        ),
        result=result,
        latency=latency,
        model_call_count=(
            model_call_count
        ),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_enabled=use_cache,
    )