from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from backend.core.settings import get_settings
from backend.fine_tuning.registry import adapter_registry
from backend.fine_tuning.schemas import FineTuneJobStatus, FineTuneRequest
from backend.fine_tuning.trainer import lora_trainer_service


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FineTuneJobManager:
    """Persist and execute local LoRA fine-tuning jobs."""

    def __init__(self) -> None:
        self.settings = get_settings()

        self.jobs_directory = (
            self.settings.fine_tuning_jobs_dir.resolve()
        )
        self.jobs_directory.mkdir(parents=True, exist_ok=True)

        self._lock = RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(
                1,
                self.settings.fine_tuning_max_concurrent_jobs,
            ),
            thread_name_prefix="researchease-lora",
        )

        self._mark_interrupted_jobs()

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_directory / f"{job_id}.json"

    def _write_job(self, job: dict[str, Any]) -> None:
        """Write a job atomically enough for the single-process dev server."""
        with self._lock:
            self._job_path(job["job_id"]).write_text(
                json.dumps(
                    job,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )

    def _read_job(self, job_id: str) -> dict[str, Any]:
        path = self._job_path(job_id)

        if not path.exists():
            raise KeyError(
                f"Fine-tuning job '{job_id}' was not found."
            )

        return json.loads(path.read_text(encoding="utf-8"))

    def _mark_interrupted_jobs(self) -> None:
        """Mark stale queued/running jobs after an API restart."""
        for path in self.jobs_directory.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))

                if job.get("status") in {
                    FineTuneJobStatus.QUEUED.value,
                    FineTuneJobStatus.RUNNING.value,
                }:
                    job["status"] = FineTuneJobStatus.INTERRUPTED.value
                    job["stage"] = "interrupted"
                    job["error"] = (
                        "The FastAPI process stopped before training "
                        "completed."
                    )
                    job["completed_at"] = _now()
                    self._write_job(job)

            except (OSError, json.JSONDecodeError, KeyError):
                continue

    def create_job(self, request: FineTuneRequest) -> dict[str, Any]:
        if adapter_registry.exists(request.adapter_name):
            raise ValueError(
                f"Adapter '{request.adapter_name}' already exists."
            )

        for existing in self.list_jobs():
            if (
                existing.get("adapter_name") == request.adapter_name
                and existing.get("status")
                in {
                    FineTuneJobStatus.QUEUED.value,
                    FineTuneJobStatus.RUNNING.value,
                }
            ):
                raise ValueError(
                    "A training job for this adapter is already active."
                )

        job_id = str(uuid4())

        job: dict[str, Any] = {
            "job_id": job_id,
            "adapter_name": request.adapter_name,
            "status": FineTuneJobStatus.QUEUED.value,
            "stage": "queued",
            "progress": 0.0,
            "created_at": _now(),
            "started_at": "",
            "completed_at": "",
            "metrics": {},
            "result": {},
            "error": "",
            "request": request.model_dump(mode="json"),
        }

        self._write_job(job)
        self._executor.submit(self._run_job, job_id)

        return job

    def _update_progress(
        self,
        job_id: str,
        stage: str,
        progress: float,
        metrics: dict[str, Any],
    ) -> None:
        job = self._read_job(job_id)

        job["stage"] = stage
        job["progress"] = max(0.0, min(1.0, float(progress)))
        job["metrics"] = {
            **job.get("metrics", {}),
            **metrics,
        }

        self._write_job(job)

    def _run_job(self, job_id: str) -> None:
        job = self._read_job(job_id)

        job["status"] = FineTuneJobStatus.RUNNING.value
        job["stage"] = "starting"
        job["started_at"] = _now()
        job["progress"] = 0.01
        self._write_job(job)

        try:
            request = FineTuneRequest.model_validate(job["request"])

            result = lora_trainer_service.train(
                job_id=job_id,
                request=request,
                progress_callback=lambda stage, progress, metrics: (
                    self._update_progress(
                        job_id,
                        stage,
                        progress,
                        metrics,
                    )
                ),
            )

            job = self._read_job(job_id)
            job["status"] = FineTuneJobStatus.COMPLETED.value
            job["stage"] = "completed"
            job["progress"] = 1.0
            job["completed_at"] = _now()
            job["result"] = result
            job["error"] = ""
            self._write_job(job)

        except Exception as exc:
            job = self._read_job(job_id)
            job["status"] = FineTuneJobStatus.FAILED.value
            job["stage"] = "failed"
            job["completed_at"] = _now()
            job["error"] = (
                f"{exc}\n\n{traceback.format_exc(limit=8)}"
            )
            self._write_job(job)

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self._read_job(job_id)
        job.pop("request", None)
        return job

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []

        for path in self.jobs_directory.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                job.pop("request", None)
                jobs.append(job)
            except (OSError, json.JSONDecodeError):
                continue

        jobs.sort(
            key=lambda item: item.get("created_at", ""),
            reverse=True,
        )

        return jobs[: max(1, int(limit))]


fine_tune_job_manager = FineTuneJobManager()