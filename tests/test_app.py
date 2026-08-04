import os
from pathlib import Path

import pytest


os.environ.setdefault(
    "OPENAI_API_KEY",
    "sk-test-key-for-unit-tests",
)

from app import (  # noqa: E402
    app,
    health,
    validate_image_path,
)


def test_health_response() -> None:
    """The health function should return service metadata."""

    response = health()

    assert response["status"] == "ok"
    assert (
        response["service"]
        == "multimodal-evidence-agent"
    )
    assert response["version"] == "1.1.0"
    assert response["disk_cache_supported"] is True


def test_health_route_is_registered() -> None:
    """FastAPI should expose GET /health."""

    health_routes = [
        route
        for route in app.routes
        if route.path == "/health"
    ]

    assert len(health_routes) == 1
    assert "GET" in health_routes[0].methods


def test_verify_route_is_registered() -> None:
    """FastAPI should expose POST /verify."""

    verify_routes = [
        route
        for route in app.routes
        if route.path == "/verify"
    ]

    assert len(verify_routes) == 1
    assert "POST" in verify_routes[0].methods


def test_verify_example_route_is_registered() -> None:
    """FastAPI should expose the example endpoint."""

    example_routes = [
        route
        for route in app.routes
        if (
            route.path
            == "/verify-example/{example_id}"
        )
    ]

    assert len(example_routes) == 1
    assert "POST" in example_routes[0].methods


def test_valid_image_path_is_accepted() -> None:
    """A project image should pass the path safety check."""

    validated_path = validate_image_path(
        "data/images/sample_001.png"
    )

    assert (
        validated_path
        == "data/images/sample_001.png"
    )


def test_absolute_project_image_path_is_accepted() -> None:
    """An absolute path inside data/images should be accepted."""

    absolute_path = Path(
        "data/images/sample_001.png"
    ).resolve()

    validated_path = validate_image_path(
        str(absolute_path)
    )

    assert (
        validated_path
        == "data/images/sample_001.png"
    )


def test_path_outside_image_directory_is_rejected() -> None:
    """The API must reject arbitrary local file access."""

    with pytest.raises(
        ValueError,
        match=(
            "image_path must point to a file "
            "inside data/images"
        ),
    ):
        validate_image_path(
            "/etc/passwd"
        )


def test_missing_image_is_rejected() -> None:
    """A nonexistent project image should be rejected."""

    with pytest.raises(
        FileNotFoundError,
        match="Image not found",
    ):
        validate_image_path(
            "data/images/missing-image.png"
        )
