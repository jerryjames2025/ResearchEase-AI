from __future__ import annotations

from fastapi.testclient import (
    TestClient,
)

from backend.main import app


client = TestClient(
    app
)


def test_health_endpoint() -> None:
    response = client.get(
        "/api/v1/health"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "status"
    ] == "healthy"

    assert payload[
        "version"
    ] == "7.0.0"


def test_root_endpoint() -> None:
    response = client.get(
        "/"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "documentation"
    ] == "/docs"


def test_reject_non_pdf_upload() -> None:
    response = client.post(
        "/api/v1/papers/upload",
        files={
            "file": (
                "notes.txt",
                b"not a pdf",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    payload = response.json()

    assert payload[
        "error"
    ] == "invalid_file_type"


def test_chat_request_validation() -> None:
    response = client.post(
        "/api/v1/chat/fake-session",
        json={
            "question": "",
            "answer_mode": "paper_only",
        },
    )

    assert response.status_code == 422

    payload = response.json()

    assert payload[
        "error"
    ] == "validation_error"


def test_invalid_answer_mode() -> None:
    response = client.post(
        "/api/v1/chat/fake-session",
        json={
            "question": (
                "Explain the methodology"
            ),
            "answer_mode": (
                "unsupported_mode"
            ),
        },
    )

    assert response.status_code == 422