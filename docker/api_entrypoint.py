from __future__ import annotations

import os
import subprocess
import sys


def env_flag(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(
        name,
        str(default),
    )

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def run_migrations() -> None:
    print(
        "Applying ResearchEase database migrations...",
        flush=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "upgrade",
            "head",
        ],
        check=True,
    )

    print(
        "Database migrations completed.",
        flush=True,
    )


def start_api() -> None:
    host = os.getenv(
        "RESEARCHEASE_API_HOST",
        "0.0.0.0",
    )

    port = os.getenv(
        "RESEARCHEASE_API_PORT",
        "8000",
    )

    workers = os.getenv(
        "RESEARCHEASE_API_WORKERS",
        "1",
    )

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        host,
        "--port",
        port,
        "--workers",
        workers,
        "--proxy-headers",
        "--forwarded-allow-ips",
        "*",
    ]

    print(
        "Starting FastAPI:",
        " ".join(command),
        flush=True,
    )

    os.execv(
        sys.executable,
        command,
    )


def main() -> None:
    if env_flag(
        "RESEARCHEASE_RUN_MIGRATIONS",
        True,
    ):
        run_migrations()

    start_api()


if __name__ == "__main__":
    main()