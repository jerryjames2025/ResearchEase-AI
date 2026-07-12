from __future__ import annotations

from pathlib import Path

from backend.core.settings import (
    get_settings,
)

from backend.storage.mongo import (
    ping_mongo,
)

from backend.storage.postgres import (
    ping_postgres,
)

from backend.storage.redis_cache import (
    ping_redis,
)


def storage_health() -> dict:
    settings = get_settings()

    (
        postgres_ok,
        postgres_detail,
    ) = ping_postgres()

    (
        mongo_ok,
        mongo_detail,
    ) = ping_mongo()

    (
        redis_ok,
        redis_detail,
    ) = ping_redis()

    faiss_ok = True

    faiss_detail = (
        "FAISS persistence directory "
        "is writable."
    )

    try:
        path = Path(
            settings.faiss_directory
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        probe = (
            path
            / ".write-test"
        )

        probe.write_text(
            "ok",
            encoding="utf-8",
        )

        probe.unlink(
            missing_ok=True
        )

    except Exception as exc:
        faiss_ok = False

        faiss_detail = (
            "FAISS directory is not "
            f"writable: {exc}"
        )

    services = {
        "postgresql": {
            "healthy": postgres_ok,
            "detail": postgres_detail,
        },
        "mongodb": {
            "healthy": mongo_ok,
            "detail": mongo_detail,
        },
        "redis": {
            "healthy": redis_ok,
            "detail": redis_detail,
        },
        "faiss_disk": {
            "healthy": faiss_ok,
            "detail": faiss_detail,
        },
    }

    all_healthy = all(
        item["healthy"]
        for item in services.values()
    )

    return {
        "status": (
            "healthy"
            if all_healthy
            else "degraded"
        ),
        "services": services,
    }