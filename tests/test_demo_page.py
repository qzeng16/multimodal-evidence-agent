"""Tests for the browser-based static demo page."""

from fastapi.testclient import TestClient

import app as app_module
from app import app


client = TestClient(
    app
)


def get_registered_route(
    path: str,
):
    """Return exactly one route registered at a path."""

    routes = [
        route
        for route in app.routes
        if route.path == path
    ]

    assert len(routes) == 1

    return routes[0]


def test_root_serves_demo_page() -> None:
    """The application root should serve the browser demo."""

    response = client.get(
        "/"
    )

    assert response.status_code == 200

    assert (
        "text/html"
        in response.headers["content-type"]
    )

    assert (
        "Multimodal Evidence Verification Agent"
        in response.text
    )


def test_demo_alias_serves_page() -> None:
    """The explicit /demo path should serve the same page."""

    response = client.get(
        "/demo"
    )

    assert response.status_code == 200

    assert (
        "Static portfolio demo"
        in response.text
    )


def test_page_uses_public_demo_api() -> None:
    """The page should load data from the static demo API."""

    response = client.get(
        "/"
    )

    assert response.status_code == 200

    assert (
        '"/demo/examples"'
        in response.text
    )

    assert (
        "/demo/examples/${encodeURIComponent"
        in response.text
    )

    assert (
        "/demo-images/"
        in response.text
    )


def test_static_demo_image_is_public() -> None:
    """A curated image should be publicly served."""

    response = client.get(
        "/demo-images/sample_001.png"
    )

    assert response.status_code == 200

    assert (
        response.headers["content-type"]
        == "image/png"
    )

    assert response.content


def test_demo_page_is_public(
    monkeypatch,
) -> None:
    """The browser page should not require X-API-Key."""

    monkeypatch.setenv(
        "SERVICE_API_KEY",
        "protected-service-key",
    )

    root_response = client.get(
        "/"
    )

    alias_response = client.get(
        "/demo"
    )

    assert root_response.status_code == 200
    assert alias_response.status_code == 200

    for path in (
        "/",
        "/demo",
    ):
        route = get_registered_route(
            path
        )

        assert (
            route.dependant.dependencies
            == []
        )


def test_demo_page_does_not_embed_secrets(
    monkeypatch,
) -> None:
    """The public HTML must not contain configured secrets."""

    service_key = (
        "do-not-render-this-service-key"
    )

    openai_key = (
        "sk-do-not-render-this-openai-key"
    )

    monkeypatch.setenv(
        "SERVICE_API_KEY",
        service_key,
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        openai_key,
    )

    response = client.get(
        "/"
    )

    assert service_key not in response.text
    assert openai_key not in response.text
    assert "X-API-Key" not in response.text


def test_demo_page_does_not_call_pipeline(
    monkeypatch,
) -> None:
    """Loading the static demo must not call the model pipeline."""

    def fail_if_called(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "The static demo must not call "
            "run_verification."
        )

    monkeypatch.setattr(
        app_module,
        "run_verification",
        fail_if_called,
    )

    page_response = client.get(
        "/"
    )

    image_response = client.get(
        "/demo-images/sample_001.png"
    )

    list_response = client.get(
        "/demo/examples"
    )

    assert page_response.status_code == 200
    assert image_response.status_code == 200
    assert list_response.status_code == 200
