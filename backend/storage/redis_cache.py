from __future__ import annotations

import hashlib
import json
import logging

from functools import lru_cache
from typing import Any

import redis

from redis.exceptions import (
    RedisError,
)

from backend.core.settings import (
    get_settings,
)


logger = logging.getLogger(
    "researchease.redis"
)


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    settings = get_settings()

    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=3,
        health_check_interval=30,
    )


def ping_redis() -> tuple[
    bool,
    str,
]:
    try:
        get_redis_client().ping()

        return (
            True,
            "Redis is connected.",
        )

    except Exception as exc:
        return (
            False,
            (
                "Redis connection "
                f"failed: {exc}"
            ),
        )


class RedisCache:
    @staticmethod
    def make_key(
        namespace: str,
        payload: Any,
    ) -> str:

        serialized = json.dumps(
            payload,
            sort_keys=True,
            default=str,
            separators=(
                ",",
                ":",
            ),
        )

        digest = hashlib.sha256(
            serialized.encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            f"researchease:"
            f"{namespace}:"
            f"{digest}"
        )

    def get_json(
        self,
        key: str,
    ) -> Any | None:

        try:
            value = (
                get_redis_client()
                .get(key)
            )

            return (
                json.loads(value)
                if value
                else None
            )

        except (
            RedisError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning(
                (
                    "Redis GET failed "
                    "for %s: %s"
                ),
                key,
                exc,
            )

            return None

    def set_json(
        self,
        key: str,
        value: Any,
        ttl_seconds: int,
    ) -> bool:

        try:
            get_redis_client().setex(
                key,
                max(
                    1,
                    int(ttl_seconds),
                ),
                json.dumps(
                    value,
                    default=str,
                ),
            )

            return True

        except RedisError as exc:
            logger.warning(
                (
                    "Redis SET failed "
                    "for %s: %s"
                ),
                key,
                exc,
            )

            return False

    def delete(
        self,
        key: str,
    ) -> None:

        try:
            get_redis_client().delete(
                key
            )

        except RedisError as exc:
            logger.warning(
                (
                    "Redis DELETE failed "
                    "for %s: %s"
                ),
                key,
                exc,
            )

    def delete_prefix(
        self,
        prefix: str,
    ) -> None:

        try:
            client = get_redis_client()

            keys = list(
                client.scan_iter(
                    match=f"{prefix}*",
                    count=100,
                )
            )

            if keys:
                client.delete(
                    *keys
                )

        except RedisError as exc:
            logger.warning(
                (
                    "Redis prefix deletion "
                    "failed for %s: %s"
                ),
                prefix,
                exc,
            )


redis_cache = RedisCache()