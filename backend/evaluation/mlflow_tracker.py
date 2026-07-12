from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from threading import RLock
from typing import Any

import mlflow
from mlflow import MlflowClient

from backend.core.settings import (
    get_settings,
)


class MLflowTrackerError(
    RuntimeError
):
    """
    Raised when an MLflow tracking operation fails.
    """


def _timestamp_to_iso(
    timestamp_ms: int | None,
) -> str:
    if not timestamp_ms:
        return ""

    return (
        datetime.fromtimestamp(
            timestamp_ms / 1000,
            tz=timezone.utc,
        )
        .isoformat()
    )


def _safe_parameter(
    value: Any,
) -> str:
    if isinstance(
        value,
        (dict, list, tuple, set),
    ):
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    else:
        rendered = str(value)

    return rendered[:500]


class MLflowEvaluationTracker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._lock = RLock()

        os.environ.setdefault(
            "MLFLOW_HTTP_REQUEST_TIMEOUT",
            str(
                self.settings
                .mlflow_request_timeout_seconds
            ),
        )

    def configure(self) -> None:
        mlflow.set_tracking_uri(
            self.settings
            .mlflow_tracking_uri
        )

    def client(self) -> MlflowClient:
        self.configure()

        return MlflowClient(
            tracking_uri=(
                self.settings
                .mlflow_tracking_uri
            )
        )

    def health(
        self,
    ) -> tuple[bool, str]:
        try:
            client = self.client()

            client.search_experiments(
                max_results=1
            )

            return (
                True,
                "MLflow tracking server is connected.",
            )

        except Exception as exc:
            return (
                False,
                (
                    "MLflow tracking server "
                    f"is unavailable: {exc}"
                ),
            )

    def ensure_experiment(
        self,
    ) -> str:
        client = self.client()

        experiment = (
            client.get_experiment_by_name(
                self.settings
                .mlflow_experiment_name
            )
        )

        if experiment is not None:
            return experiment.experiment_id

        return client.create_experiment(
            self.settings
            .mlflow_experiment_name
        )

    def log_evaluation(
        self,
        *,
        run_name: str,
        parameters: dict[str, Any],
        aggregate_metrics: dict[
            str,
            float,
        ],
        dataset_payload: dict,
        result_payload: dict,
        tags: dict[str, str],
    ) -> str:
        """
        Log one complete RAG evaluation to MLflow.
        """

        with self._lock:
            self.configure()

            experiment_id = (
                self.ensure_experiment()
            )

            with mlflow.start_run(
                experiment_id=(
                    experiment_id
                ),
                run_name=run_name,
                tags=tags,
            ) as active_run:
                safe_parameters = {
                    key: _safe_parameter(
                        value
                    )
                    for key, value
                    in parameters.items()
                }

                mlflow.log_params(
                    safe_parameters
                )

                safe_metrics = {
                    key: float(value)
                    for key, value
                    in aggregate_metrics.items()
                    if math.isfinite(
                        float(value)
                    )
                }

                if safe_metrics:
                    mlflow.log_metrics(
                        safe_metrics
                    )

                mlflow.log_dict(
                    dataset_payload,
                    (
                        "evaluation/"
                        "dataset.json"
                    ),
                )

                mlflow.log_dict(
                    result_payload,
                    (
                        "evaluation/"
                        "results.json"
                    ),
                )

                with tempfile.TemporaryDirectory() as temp_dir:
                    csv_path = (
                        Path(temp_dir)
                        / "results.csv"
                    )

                    self._write_results_csv(
                        csv_path,
                        result_payload.get(
                            "cases",
                            [],
                        ),
                    )

                    mlflow.log_artifact(
                        str(csv_path),
                        artifact_path=(
                            "evaluation"
                        ),
                    )

                return (
                    active_run.info.run_id
                )

    @staticmethod
    def _write_results_csv(
        path: Path,
        cases: list[dict],
    ) -> None:
        fieldnames = [
            "case_id",
            "question",
            "expected_pages",
            "retrieved_pages",
            "answer",
            "error",
            "metrics",
            "judge",
        ]

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for case in cases:
                writer.writerow(
                    {
                        "case_id": (
                            case.get(
                                "case_id",
                                "",
                            )
                        ),
                        "question": (
                            case.get(
                                "question",
                                "",
                            )
                        ),
                        "expected_pages": (
                            json.dumps(
                                case.get(
                                    "expected_pages",
                                    [],
                                )
                            )
                        ),
                        "retrieved_pages": (
                            json.dumps(
                                case.get(
                                    "retrieved_pages",
                                    [],
                                )
                            )
                        ),
                        "answer": (
                            case.get(
                                "answer",
                                "",
                            )
                        ),
                        "error": (
                            case.get(
                                "error",
                                "",
                            )
                        ),
                        "metrics": (
                            json.dumps(
                                case.get(
                                    "metrics",
                                    {},
                                )
                            )
                        ),
                        "judge": (
                            json.dumps(
                                case.get(
                                    "judge",
                                    {},
                                )
                            )
                        ),
                    }
                )

    def list_runs(
        self,
        limit: int = 50,
    ) -> list[dict]:
        client = self.client()

        experiment = (
            client.get_experiment_by_name(
                self.settings
                .mlflow_experiment_name
            )
        )

        if experiment is None:
            return []

        runs = client.search_runs(
            experiment_ids=[
                experiment.experiment_id
            ],
            order_by=[
                (
                    "attributes.start_time "
                    "DESC"
                )
            ],
            max_results=max(
                1,
                min(
                    int(limit),
                    200,
                ),
            ),
        )

        return [
            self._serialize_run(
                run
            )
            for run in runs
        ]

    def get_run(
        self,
        run_id: str,
    ) -> dict:
        run = self.client().get_run(
            run_id
        )

        return self._serialize_run(
            run
        )

    @staticmethod
    def _serialize_run(
        run,
    ) -> dict:
        tags = dict(
            run.data.tags
            or {}
        )

        return {
            "run_id": (
                run.info.run_id
            ),
            "run_name": tags.get(
                "mlflow.runName",
                "",
            ),
            "status": (
                run.info.status
            ),
            "start_time": (
                _timestamp_to_iso(
                    run.info.start_time
                )
            ),
            "end_time": (
                _timestamp_to_iso(
                    run.info.end_time
                )
            ),
            "artifact_uri": (
                run.info.artifact_uri
                or ""
            ),
            "parameters": dict(
                run.data.params
                or {}
            ),
            "metrics": {
                key: float(value)
                for key, value
                in (
                    run.data.metrics
                    or {}
                ).items()
            },
            "tags": tags,
        }


mlflow_tracker = (
    MLflowEvaluationTracker()
)