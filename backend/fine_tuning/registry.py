from __future__ import annotations

import json
import shutil
from pathlib import Path
from threading import RLock
from typing import Any

from backend.core.settings import (
    get_settings,
)


class AdapterRegistryError(
    RuntimeError
):
    pass


class AdapterRegistry:
    def __init__(self) -> None:
        self.settings = get_settings()

        self.root = (
            self.settings
            .fine_tuning_output_dir
            .resolve()
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = RLock()

    @staticmethod
    def validate_name(
        adapter_name: str,
    ) -> str:
        cleaned = adapter_name.strip()

        if not cleaned:
            raise AdapterRegistryError(
                "Adapter name cannot be empty."
            )

        allowed = (
            cleaned.replace("_", "")
            .replace("-", "")
            .isalnum()
        )

        if not allowed:
            raise AdapterRegistryError(
                "Adapter names may contain only "
                "letters, numbers, hyphens, "
                "and underscores."
            )

        return cleaned

    def adapter_path(
        self,
        adapter_name: str,
    ) -> Path:
        cleaned = self.validate_name(
            adapter_name
        )

        candidate = (
            self.root / cleaned
        ).resolve()

        if candidate.parent != self.root:
            raise AdapterRegistryError(
                "Invalid adapter path."
            )

        return candidate

    def exists(
        self,
        adapter_name: str,
    ) -> bool:
        path = self.adapter_path(
            adapter_name
        )

        return (
            path.is_dir()
            and (
                path
                / "adapter_config.json"
            ).exists()
        )

    def ensure_available(
        self,
        adapter_name: str,
    ) -> Path:
        path = self.adapter_path(
            adapter_name
        )

        if path.exists():
            raise AdapterRegistryError(
                f"Adapter '{adapter_name}' "
                "already exists."
            )

        return path

    def metadata(
        self,
        adapter_name: str,
    ) -> dict[str, Any]:
        path = self.adapter_path(
            adapter_name
        )

        if not path.exists():
            raise AdapterRegistryError(
                f"Adapter '{adapter_name}' "
                "was not found."
            )

        metadata_path = (
            path / "metadata.json"
        )

        metadata: dict[
            str,
            Any,
        ] = {}

        if metadata_path.exists():
            try:
                metadata = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )
            except (
                OSError,
                json.JSONDecodeError,
            ):
                metadata = {}

        size_bytes = sum(
            file.stat().st_size
            for file in path.rglob("*")
            if file.is_file()
        )

        return {
            "adapter_name": (
                adapter_name
            ),
            "path": str(path),
            "base_model": (
                metadata.get(
                    "base_model",
                    "",
                )
            ),
            "created_at": (
                metadata.get(
                    "created_at",
                    "",
                )
            ),
            "trainable_parameters": int(
                metadata.get(
                    "trainable_parameters",
                    0,
                )
            ),
            "total_parameters": int(
                metadata.get(
                    "total_parameters",
                    0,
                )
            ),
            "trainable_percentage": float(
                metadata.get(
                    "trainable_percentage",
                    0.0,
                )
            ),
            "training_examples": int(
                metadata.get(
                    "training_examples",
                    0,
                )
            ),
            "evaluation_examples": int(
                metadata.get(
                    "evaluation_examples",
                    0,
                )
            ),
            "size_bytes": size_bytes,
            "metadata": metadata,
        }

    def list_adapters(
        self,
    ) -> list[dict[str, Any]]:
        adapters: list[
            dict[str, Any]
        ] = []

        for path in sorted(
            self.root.iterdir()
        ):
            if not path.is_dir():
                continue

            if not (
                path
                / "adapter_config.json"
            ).exists():
                continue

            adapters.append(
                self.metadata(path.name)
            )

        return adapters

    def write_metadata(
        self,
        adapter_name: str,
        metadata: dict[str, Any],
    ) -> None:
        path = self.adapter_path(
            adapter_name
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            path / "metadata.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    def delete(
        self,
        adapter_name: str,
    ) -> None:
        path = self.adapter_path(
            adapter_name
        )

        if not path.exists():
            raise AdapterRegistryError(
                f"Adapter '{adapter_name}' "
                "was not found."
            )

        with self._lock:
            shutil.rmtree(path)


adapter_registry = AdapterRegistry()