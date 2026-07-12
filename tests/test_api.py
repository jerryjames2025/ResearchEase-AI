from __future__ import annotations
from backend.core.settings import get_settings

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
    settings = get_settings()

    assert payload["status"] == "healthy"
    assert payload["version"] == settings.api_version


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

def test_vector_provider_catalog() -> None:
    response = client.get(
        "/api/v1/vector/providers"
    )

    assert response.status_code == 200

    payload = response.json()

    providers = {
        item["provider"]
        for item
        in payload["providers"]
    }

    assert providers == {
        "faiss",
        "pinecone",
    }


def test_vector_health_endpoint() -> None:
    response = client.get(
        "/api/v1/vector/health"
    )

    assert response.status_code == 200

    payload = response.json()

    assert "faiss" in payload[
        "providers"
    ]

    assert "pinecone" in payload[
        "providers"
    ]


def test_invalid_vector_backend() -> None:
    response = client.post(
        "/api/v1/papers/fake-session/index",
        json={
            "embedding_model": (
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            "device": "cpu",
            "vector_backend": (
                "unsupported"
            ),
        },
    )

    assert response.status_code == 422
    
def test_evaluation_health_endpoint() -> None:
    response = client.get(
        "/api/v1/evaluation/health"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] in {
        "healthy",
        "degraded",
    }

    assert (
        "tracking_uri"
        in payload
    )

    assert (
        "experiment_name"
        in payload
    )


def test_evaluation_template() -> None:
    response = client.get(
        "/api/v1/evaluation/template"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "dataset_name"
    ]

    assert len(
        payload["cases"]
    ) >= 1

    assert (
        "question"
        in payload["cases"][0]
    )


def test_empty_evaluation_dataset_rejected() -> None:
    response = client.post(
        "/api/v1/evaluation/run",
        json={
            "session_id": (
                "fake-session"
            ),
            "dataset_name": (
                "empty-dataset"
            ),
            "cases": [],
            "top_k": 5,
            "minimum_score": 0.0,
            "run_generation": False,
            "use_llm_judge": False,
        },
    )

    assert response.status_code == 422

def test_fine_tuning_health_endpoint() -> None:
    response = client.get(
        "/api/v1/fine-tuning/health"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "healthy"
    assert "cuda_available" in payload
    assert "base_model" in payload


def test_fine_tuning_template_endpoint() -> None:
    response = client.get(
        "/api/v1/fine-tuning/template"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["adapter_name"]
    assert payload["base_model"]
    assert len(payload["examples"]) >= 4


def test_fine_tuning_dataset_validation() -> None:
    response = client.post(
        "/api/v1/fine-tuning/validate",
        json={
            "examples": [
                {
                    "instruction": (
                        "Explain RAG."
                    ),
                    "input": (
                        "RAG retrieves context."
                    ),
                    "output": (
                        "RAG retrieves relevant "
                        "context before generation."
                    ),
                }
            ]
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "example_count"
    ] == 1


def test_fine_tuning_adapter_catalog() -> None:
    response = client.get(
        "/api/v1/fine-tuning/adapters"
    )

    assert response.status_code == 200

    payload = response.json()

    assert "adapters" in payload


def test_fine_tuning_rejects_too_few_examples() -> None:
    response = client.post(
        "/api/v1/fine-tuning/jobs",
        json={
            "adapter_name": (
                "invalid-small-dataset"
            ),
            "examples": [
                {
                    "instruction": (
                        "Explain RAG."
                    ),
                    "output": (
                        "RAG retrieves context."
                    ),
                }
            ],
        },
    )

    assert response.status_code == 422