from __future__ import annotations

from typing import Any

import requests


class APIClientError(RuntimeError):
    """
    Raised when the FastAPI backend returns an error.
    """


class ResearchEaseAPI:
    """
    HTTP client for the ResearchEase FastAPI backend.
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: int = 600,
        llm_provider: str = "ollama",
        llm_model: str = "",
        enable_fallback: bool = False,
    ) -> None:
        self.base_url = (
            base_url.rstrip("/")
        )

        self.timeout_seconds = (
            timeout_seconds
        )

        self.llm_provider = (
            llm_provider
        )

        self.llm_model = llm_model

        self.enable_fallback = (
            enable_fallback
        )

    def _llm_headers(
        self,
    ) -> dict[str, str]:
        return {
            (
                "X-ResearchEase-"
                "LLM-Provider"
            ): self.llm_provider,

            (
                "X-ResearchEase-"
                "LLM-Model"
            ): self.llm_model,

            (
                "X-ResearchEase-"
                "LLM-Fallback"
            ): str(
                self.enable_fallback
            ).lower(),
        }

    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> Any:
        url = (
            f"{self.base_url}{path}"
        )

        supplied_headers = dict(
            kwargs.pop(
                "headers",
                {},
            )
        )

        supplied_headers.update(
            self._llm_headers()
        )

        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=(
                    self.timeout_seconds
                ),
                headers=(
                    supplied_headers
                ),
                **kwargs,
            )

        except requests.RequestException as exc:
            raise APIClientError(
                "Unable to communicate with "
                f"the FastAPI backend: {exc}"
            ) from exc

        if not response.ok:
            try:
                payload = response.json()

                detail = payload.get(
                    "detail",
                    payload,
                )

            except ValueError:
                detail = response.text

            raise APIClientError(
                (
                    "API request failed "
                    f"({response.status_code}): "
                    f"{detail}"
                )
            )

        if not response.content:
            return None

        return response.json()

    def health(self) -> dict:
        return self._request(
            "GET",
            "/api/v1/health",
        )

    def readiness(self) -> dict:
        return self._request(
            "GET",
            "/api/v1/ready",
        )

    def storage_health(
        self,
    ) -> dict:
        return self._request(
            "GET",
            "/api/v1/storage/health",
        )

    def llm_providers(
        self,
    ) -> dict:
        return self._request(
            "GET",
            "/api/v1/llm/providers",
        )

    def test_llm_provider(
        self,
        provider: str,
        model: str,
        enable_fallback: bool,
    ) -> dict:
        return self._request(
            "POST",
            "/api/v1/llm/test",
            json={
                "provider": provider,
                "model": model,
                "enable_fallback": (
                    enable_fallback
                ),
            },
        )

    def upload_paper(
        self,
        filename: str,
        data: bytes,
    ) -> dict:
        return self._request(
            "POST",
            "/api/v1/papers/upload",
            files={
                "file": (
                    filename,
                    data,
                    "application/pdf",
                )
            },
        )

    def list_sessions(
        self,
        limit: int = 100,
    ) -> dict:
        return self._request(
            "GET",
            (
                "/api/v1/papers"
                f"?limit={limit}"
            ),
        )

    def get_session(
        self,
        session_id: str,
    ) -> dict:
        return self._request(
            "GET",
            (
                "/api/v1/papers/"
                f"{session_id}"
            ),
        )

    def delete_session(
        self,
        session_id: str,
    ) -> dict:
        return self._request(
            "DELETE",
            (
                "/api/v1/papers/"
                f"{session_id}"
            ),
        )

    def analyze_paper(
        self,
        session_id: str,
        payload: dict,
    ) -> dict:
        return self._request(
            "POST",
            (
                "/api/v1/papers/"
                f"{session_id}/analyze"
            ),
            json=payload,
        )

    def get_analysis(
        self,
        session_id: str,
    ) -> dict:
        return self._request(
            "GET",
            (
                "/api/v1/papers/"
                f"{session_id}/analysis"
            ),
        )

    def build_index(
        self,
        session_id: str,
        payload: dict,
    ) -> dict:
        return self._request(
            "POST",
            (
                "/api/v1/papers/"
                f"{session_id}/index"
            ),
            json=payload,
        )

    def chat(
        self,
        session_id: str,
        payload: dict,
    ) -> dict:
        return self._request(
            "POST",
            (
                "/api/v1/chat/"
                f"{session_id}"
            ),
            json=payload,
        )

    def chat_history(
        self,
        session_id: str,
    ) -> dict:
        return self._request(
            "GET",
            (
                "/api/v1/chat/"
                f"{session_id}/history"
            ),
        )

    def research_search(
        self,
        payload: dict,
    ) -> dict:
        return self._request(
            "POST",
            "/api/v1/research/search",
            json=payload,
        )

    def explain_equation(
        self,
        payload: dict,
    ) -> dict:
        return self._request(
            "POST",
            "/api/v1/math/equation",
            json=payload,
        )

    def explain_topic(
        self,
        payload: dict,
    ) -> dict:
        return self._request(
            "POST",
            "/api/v1/math/topic",
            json=payload,
        )

    def literature_review(
        self,
        files: list[
            tuple[str, bytes]
        ],
        form_data: dict,
    ) -> dict:
        multipart_files = [
            (
                "files",
                (
                    filename,
                    data,
                    "application/pdf",
                ),
            )
            for filename, data
            in files
        ]

        return self._request(
            "POST",
            "/api/v1/literature/review",
            files=multipart_files,
            data=form_data,
        )