"""Tests for the public read-only demo API."""

from fastapi.testclient import TestClient

import app as app_module
from app import app


client = TestClient(
    app
)


def get_registered_route(
    path: str,
):
    """Return exactly one FastAPI route by path."""

    routes = [
        route
        for route in app.routes
        if route.path == path
    ]

    assert len(routes) == 1

    return routes[0]


def test_demo_list_returns_six_examples() -> None:
    """The list endpoint should expose all curated examples."""

    response = client.get(
        "/demo/examples"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["count"] == 6
    assert len(payload["examples"]) == 6

    example_ids = {
        example["example_id"]
        for example
        in payload["examples"]
    }

    assert example_ids == {
        "sample_001",
        "sample_002",
        "sample_003",
        "sample_004",
        "sample_005",
        "sample_012",
    }


def test_demo_list_returns_lightweight_summaries() -> None:
    """List records should omit full evidence and trace details."""

    response = client.get(
        "/demo/examples"
    )

    assert response.status_code == 200

    first_example = (
        response.json()["examples"][0]
    )

    assert "claim" in first_example
    assert "predicted_label" in first_example
    assert "confidence" in first_example

    assert "evidence" not in first_example
    assert "tool_trace" not in first_example
    assert "rationale" not in first_example


def test_demo_detail_returns_full_saved_result() -> None:
    """The detail endpoint should return evidence and tool trace."""

    response = client.get(
        "/demo/examples/sample_004"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["example_id"] == "sample_004"

    assert (
        payload["predicted_label"]
        == "supported"
    )

    assert (
        payload["predicted_use_ocr"]
        is True
    )

    assert payload["evidence"]
    assert payload["tool_trace"]
    assert payload["rationale"]


def test_unknown_demo_example_returns_404() -> None:
    """The API should not fabricate an unknown static result."""

    response = client.get(
        "/demo/examples/not-a-real-example"
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": (
            "Demo example not found: "
            "not-a-real-example"
        )
    }


def test_demo_routes_are_public(
    monkeypatch,
) -> None:
    """Demo endpoints should work without X-API-Key."""

    monkeypatch.setenv(
        "SERVICE_API_KEY",
        "protected-service-key",
    )

    list_response = client.get(
        "/demo/examples"
    )

    detail_response = client.get(
        "/demo/examples/sample_004"
    )

    assert list_response.status_code == 200
    assert detail_response.status_code == 200

    for path in (
        "/demo/examples",
        "/demo/examples/{example_id}",
    ):
        route = get_registered_route(
            path
        )

        assert (
            route.dependant.dependencies
            == []
        )


def test_demo_endpoint_does_not_call_model(
    monkeypatch,
) -> None:
    """Static demo requests must never invoke the model pipeline."""

    def fail_if_called(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "run_verification must not be called "
            "by a static demo endpoint."
        )

    monkeypatch.setattr(
        app_module,
        "run_verification",
        fail_if_called,
    )

    response = client.get(
        "/demo/examples/sample_004"
    )

    assert response.status_code == 200


def test_demo_routes_appear_in_openapi() -> None:
    """The two public demo routes should appear in API docs."""

    schema = app.openapi()

    assert (
        "/demo/examples"
        in schema["paths"]
    )

    assert (
        "/demo/examples/{example_id}"
        in schema["paths"]
    )

    assert (
        "get"
        in schema["paths"][
            "/demo/examples"
        ]
    )
