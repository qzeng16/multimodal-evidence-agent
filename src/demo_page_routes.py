"""Routes for the public browser-based static demo."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from src.demo_web import DEMO_HTML


router = APIRouter(
    tags=["demo"],
)


@router.get(
    "/",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def home_page() -> HTMLResponse:
    """Serve the static portfolio demo at the site root."""

    return HTMLResponse(
        content=DEMO_HTML
    )


@router.get(
    "/demo",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def demo_page() -> HTMLResponse:
    """Serve an explicit alias for the portfolio demo."""

    return HTMLResponse(
        content=DEMO_HTML
    )
