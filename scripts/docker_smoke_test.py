from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceCheck:
    name: str
    url: str
    expect_json: bool = False


CHECKS = [
    ServiceCheck(
        name="FastAPI health",
        url=(
            "http://127.0.0.1:8000"
            "/api/v1/health"
        ),
        expect_json=True,
    ),
    ServiceCheck(
        name="Storage health",
        url=(
            "http://127.0.0.1:8000"
            "/api/v1/storage/health"
        ),
        expect_json=True,
    ),
    ServiceCheck(
        name="Vector health",
        url=(
            "http://127.0.0.1:8000"
            "/api/v1/vector/health"
        ),
        expect_json=True,
    ),
    ServiceCheck(
        name="Evaluation health",
        url=(
            "http://127.0.0.1:8000"
            "/api/v1/evaluation/health"
        ),
        expect_json=True,
    ),
    ServiceCheck(
        name="Fine-tuning health",
        url=(
            "http://127.0.0.1:8000"
            "/api/v1/fine-tuning/health"
        ),
        expect_json=True,
    ),
    ServiceCheck(
        name="Streamlit",
        url=(
            "http://127.0.0.1:8501"
            "/_stcore/health"
        ),
    ),
    ServiceCheck(
        name="MLflow",
        url="http://127.0.0.1:5000/health",
    ),
]


def check_service(
    check: ServiceCheck,
    timeout_seconds: int = 10,
) -> bool:
    try:
        with urllib.request.urlopen(
            check.url,
            timeout=timeout_seconds,
        ) as response:
            body = response.read().decode(
                "utf-8",
                errors="replace",
            )

            if response.status != 200:
                print(
                    f"[FAIL] {check.name}: "
                    f"HTTP {response.status}"
                )
                return False

            print(
                f"[PASS] {check.name}: "
                f"HTTP {response.status}"
            )

            if (
                check.expect_json
                and body.strip()
            ):
                try:
                    payload = json.loads(
                        body
                    )

                    status = payload.get(
                        "status",
                        "unknown",
                    )

                    print(
                        f"       status={status}"
                    )

                except json.JSONDecodeError:
                    print(
                        "       warning: response "
                        "was not JSON"
                    )

            return True

    except (
        urllib.error.URLError,
        TimeoutError,
    ) as exc:
        print(
            f"[FAIL] {check.name}: {exc}"
        )
        return False


def main() -> int:
    print(
        "ResearchEase AI Version 13 "
        "Docker smoke test"
    )

    print("-" * 60)

    # Allow a recently started stack a brief moment
    # before the checks.
    time.sleep(1)

    results = [
        check_service(check)
        for check in CHECKS
    ]

    print("-" * 60)

    passed = sum(results)
    total = len(results)

    print(
        f"Passed {passed}/{total} checks."
    )

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())