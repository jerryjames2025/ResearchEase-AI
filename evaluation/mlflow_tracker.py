from __future__ import annotations

import csv
import json
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from backend.core.settings import (
    get_settings,
)


class MLflowTrackingError(
    RuntimeError
):
    """
    Raised when an MLflow tracking
    operation fails.
    """


def _iso_time(
    milliseconds: int | None,
) -> str:
    if not milliseconds:
        return ""

    return datetime.fromtimestamp(
        milliseconds / 1000,
        tz=timezone.utc,
    ).isoformat()


def _stringify(
    value: Any,
    max_length: int = 500,
) -> str:
    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
        ),
    ):
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    else:
        text = str(value)

    return text[:max_length]


class MLflowTracker:
    def __init__(self) -> None:
        self.settings = (
            get_settings()
        )

    def configure(self) -> None:
        Path(
            self.settings
            .mlflow_artifact_directory
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        Path(
            self.settings
            .evaluation_directory
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        tracking_uri = (
            self.settings
            .mlflow_tracking_uri
            .strip()
        )

        if tracking_uri.startswith(
            "sqlite:///"
        ):
            database_path = (
                tracking_uri
                .removeprefix(
                    "sqlite:///"
                )
            )

            Path(
                database_path
            ).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        mlflow.set_tracking_uri(
            tracking_uri
        )

    def client(
        self,
    ) -> MlflowClient:
        self.configure()

        return MlflowClient(
            tracking_uri=(
                self.settings
                .mlflow_tracking_uri
            )
        )

    def ensure_experiment(
        self,
    ) -> str:
        client = self.client()

        experiment = (
            client
            .get_experiment_by_name(
                self.settings
                .mlflow_experiment_name
            )
        )

        if experiment is not None:
            return (
                experiment
                .experiment_id
            )

        artifact_location = Path(
            self.settings
            .mlflow_artifact_directory
        ).resolve().as_uri()

        return client.create_experiment(
            name=(
                self.settings
                .mlflow_experiment_name
            ),
            artifact_location=(
                artifact_location
            ),
            tags={
                "application": (
                    "ResearchEase AI"
                ),
                "evaluation_type": (
                    "RAG"
                ),
            },
        )

    def health(
        self,
    ) -> tuple[
        bool,
        str,
    ]:
        try:
            experiment_id = (
                self.ensure_experiment()
            )

            return (
                True,
                (
                    "MLflow is configured. "
                    "Experiment ID: "
                    f"{experiment_id}."
                ),
            )

        except Exception as exc:
            return (
                False,
                (
                    "MLflow is unavailable: "
                    f"{exc}"
                ),
            )

    def log_evaluation(
        self,
        *,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        dataset: dict[str, Any],
        report: dict[str, Any],
        tags: dict[str, str],
    ) -> tuple[
        str,
        str,
    ]:
        self.configure()

        experiment_id = (
            self.ensure_experiment()
        )

        safe_tags = {
            str(key)[:250]: (
                _stringify(
                    value,
                    5000,
                )
            )
            for key, value
            in tags.items()
        }

        safe_tags.setdefault(
            "researchease.version",
            self.settings.api_version,
        )

        safe_tags.setdefault(
            "researchease.workflow",
            "rag_evaluation",
        )

        try:
            with mlflow.start_run(
                experiment_id=(
                    experiment_id
                ),
                run_name=(
                    run_name or None
                ),
                tags=safe_tags,
            ) as active_run:

                for key, value in (
                    params.items()
                ):
                    mlflow.log_param(
                        str(key)[:250],
                        _stringify(
                            value
                        ),
                    )

                for key, value in (
                    metrics.items()
                ):
                    mlflow.log_metric(
                        str(key)[:250],
                        float(value),
                    )

                mlflow.log_dict(
                    dataset,
                    (
                        "evaluation/"
                        "dataset.json"
                    ),
                )

                mlflow.log_dict(
                    report,
                    (
                        "evaluation/"
                        "report.json"
                    ),
                )

                output_dir = Path(
                    self.settings
                    .evaluation_directory
                )

                csv_path = (
                    output_dir
                    / (
                        f"{active_run.info.run_id}"
                        "_cases.csv"
                    )
                )

                rows = []

                for item in report.get(
                    "results",
                    [],
                ):
                    flattened = {
                        "case_id": (
                            item.get(
                                "case_id",
                                "",
                            )
                        ),
                        "question": (
                            item.get(
                                "question",
                                "",
                            )
                        ),
                        "answer": (
                            item.get(
                                "answer",
                                "",
                            )
                        ),
                        "latency_seconds": (
                            item.get(
                                "latency_seconds",
                                0.0,
                            )
                        ),
                        "error": (
                            item.get(
                                "error",
                                "",
                            )
                        ),
                    }

                    for (
                        metric_name,
                        metric_value,
                    ) in item.get(
                        "metrics",
                        {},
                    ).items():
                        flattened[
                            (
                                "metric_"
                                f"{metric_name}"
                            )
                        ] = metric_value

                    for (
                        judge_name,
                        judge_value,
                    ) in item.get(
                        "judge",
                        {},
                    ).items():
                        if isinstance(
                            judge_value,
                            (
                                str,
                                int,
                                float,
                                bool,
                            ),
                        ):
                            flattened[
                                (
                                    "judge_"
                                    f"{judge_name}"
                                )
                            ] = judge_value

                    rows.append(
                        flattened
                    )

                fieldnames = sorted(
                    {
                        key
                        for row in rows
                        for key in row
                    }
                )

                with csv_path.open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=(
                            fieldnames
                        ),
                    )

                    writer.writeheader()
                    writer.writerows(
                        rows
                    )

                mlflow.log_artifact(
                    str(csv_path),
                    artifact_path=(
                        "evaluation"
                    ),
                )

                return (
                    active_run
                    .info
                    .run_id,
                    active_run
                    .info
                    .artifact_uri,
                )

        except Exception as exc:
            raise MLflowTrackingError(
                (
                    "Unable to log the "
                    "evaluation run: "
                    f"{exc}"
                )
            ) from exc

    def list_runs(
        self,
        limit: int = 20,
    ) -> list[
        dict[str, Any]
    ]:
        client = self.client()

        experiment_id = (
            self.ensure_experiment()
        )

        runs = client.search_runs(
            experiment_ids=[
                experiment_id
            ],
            order_by=[
                (
                    "attributes."
                    "start_time DESC"
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
            {
                "run_id": (
                    run.info.run_id
                ),

                "run_name": (
                    run.data.tags.get(
                        "mlflow.runName",
                        "",
                    )
                ),

                "status": (
                    run.info.status
                ),

                "start_time": (
                    _iso_time(
                        run.info
                        .start_time
                    )
                ),

                "end_time": (
                    _iso_time(
                        run.info
                        .end_time
                    )
                ),

                "metrics": {
                    key: float(value)
                    for key, value
                    in run.data
                    .metrics.items()
                },

                "params": dict(
                    run.data.params
                ),

                "tags": {
                    key: value
                    for key, value
                    in run.data
                    .tags.items()
                    if not key.startswith(
                        "mlflow."
                    )
                },

                "artifact_uri": (
                    run.info
                    .artifact_uri
                ),
            }
            for run in runs
        ]

    def get_run(
        self,
        run_id: str,
    ) -> dict[str, Any]:
        run = (
            self.client()
            .get_run(
                run_id
            )
        )

        return {
            "run_id": (
                run.info.run_id
            ),

            "run_name": (
                run.data.tags.get(
                    "mlflow.runName",
                    "",
                )
            ),

            "status": (
                run.info.status
            ),

            "start_time": (
                _iso_time(
                    run.info.start_time
                )
            ),

            "end_time": (
                _iso_time(
                    run.info.end_time
                )
            ),

            "metrics": {
                key: float(value)
                for key, value
                in run.data
                .metrics.items()
            },

            "params": dict(
                run.data.params
            ),

            "tags": dict(
                run.data.tags
            ),

            "artifact_uri": (
                run.info.artifact_uri
            ),
        }


mlflow_tracker = MLflowTracker()