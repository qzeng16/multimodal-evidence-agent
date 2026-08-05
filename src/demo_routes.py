"""Public read-only routes for curated static demo results."""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.demo_repository import (
    find_demo_result,
    list_demo_summaries,
)
from src.schemas import VerificationLabel


router = APIRouter(
    prefix="/demo",
    tags=["demo"],
)


class DemoExampleSummary(BaseModel):
    """Lightweight summary for one curated demo example."""

    example_id: str
    claim: str
    image_path: str
    category: str
    predicted_label: VerificationLabel

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    predicted_use_ocr: bool


class DemoExampleListResponse(BaseModel):
    """Response returned by the public demo list endpoint."""

    count: int = Field(
        ge=0
    )

    examples: List[
        DemoExampleSummary
    ]


def raise_demo_data_error(
    error: Exception,
) -> None:
    """Translate static data errors into an HTTP response."""

    raise HTTPException(
        status_code=500,
        detail=(
            "Static demo data is unavailable: "
            f"{error}"
        ),
    ) from error


@router.get(
    "/examples",
    response_model=DemoExampleListResponse,
)
def get_demo_examples() -> DemoExampleListResponse:
    """Return summaries of all curated demo examples."""

    try:
        summaries = list_demo_summaries()

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        raise_demo_data_error(
            error
        )

    return DemoExampleListResponse(
        count=len(summaries),
        examples=summaries,
    )


@router.get(
    "/examples/{example_id}",
    response_model=Dict[str, Any],
)
def get_demo_example(
    example_id: str,
) -> Dict[str, Any]:
    """Return one complete saved verification result."""

    try:
        result = find_demo_result(
            example_id
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        raise_demo_data_error(
            error
        )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Demo example not found: "
                f"{example_id}"
            ),
        )

    return result
