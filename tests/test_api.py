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
    ] == "8.0.0"


def test_root_endpoint() -> None:
    response = client.get(
        "/"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "documentation"
    ] == "/docs"

    assert payload[
        "storage_health"
    ] == (
        "/api/v1/storage/health"
    )


def test_storage_health_endpoint() -> None:
    response = client.get(
        "/api/v1/storage/health"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "status"
    ] in {
        "healthy",
        "degraded",
    }

    assert "postgresql" in payload[
        "services"
    ]

    assert "mongodb" in payload[
        "services"
    ]

    assert "redis" in payload[
        "services"
    ]

    assert "faiss_disk" in payload[
        "services"
    ]


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
            "answer_mode": (
                "paper_only"
            ),
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
    
def test_llm_provider_catalog() -> None:
    response = client.get(
        "/api/v1/llm/providers"
    )

    assert response.status_code == 200

    payload = response.json()

    providers = {
        item["provider"]
        for item
        in payload["providers"]
    }

    assert providers == {
        "ollama",
        "openai",
        "anthropic",
        "google",
        "huggingface",
    }


def test_invalid_llm_provider_header() -> None:
    response = client.post(
        "/api/v1/chat/fake-session",
        headers={
            (
                "X-ResearchEase-"
                "LLM-Provider"
            ): "unsupported"
        },
        json={
            "question": (
                "Explain the methodology"
            ),
            "answer_mode": (
                "paper_only"
            ),
        },
    )

    assert response.status_code == 400

    payload = response.json()

    assert payload[
        "error"
    ] == "invalid_llm_provider"