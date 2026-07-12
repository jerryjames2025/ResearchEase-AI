from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from backend.core.errors import APIError
from backend.core.retrieval import (
    retrieve_session_context,
)
from backend.core.session_store import (
    session_store,
)
from backend.core.settings import (
    get_settings,
)
from backend.evaluation.metrics import (
    answer_context_overlap,
    answer_keyword_coverage,
    retrieval_page_metrics,
    safe_mean,
    unique_in_order,
)
from backend.evaluation.mlflow_tracker import (
    mlflow_tracker,
)
from backend.evaluation.schemas import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationJudgeResult,
    RAGEvaluationResponse,
    RetrievedEvaluationSource,
)
from backend.llm.runtime import (
    get_chat_model_from_spec,
    message_text,
    parse_model_spec,
)


def _source_value(
    source: Any,
    field_name: str,
    default: Any = None,
) -> Any:
    if isinstance(
        source,
        dict,
    ):
        return source.get(
            field_name,
            default,
        )

    return getattr(
        source,
        field_name,
        default,
    )


def _normalize_source(
    source: Any,
    default_rank: int,
) -> RetrievedEvaluationSource:
    chunk = _source_value(
        source,
        "chunk",
        None,
    )

    if chunk is not None:
        page_number = _source_value(
            chunk,
            "page_number",
            0,
        )

        text = _source_value(
            chunk,
            "text",
            "",
        )

    else:
        page_number = _source_value(
            source,
            "page_number",
            0,
        )

        text = _source_value(
            source,
            "text",
            "",
        )

    return RetrievedEvaluationSource(
        rank=int(
            _source_value(
                source,
                "rank",
                default_rank,
            )
            or default_rank
        ),
        page_number=int(
            page_number or 0
        ),
        score=float(
            _source_value(
                source,
                "score",
                0.0,
            )
            or 0.0
        ),
        text=str(
            text or ""
        ),
    )


def _extract_json_object(
    value: str,
) -> dict:
    cleaned = value.strip()

    try:
        parsed = json.loads(
            cleaned
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(
            "The judge did not return JSON."
        )

    parsed = json.loads(
        match.group(0)
    )

    if not isinstance(
        parsed,
        dict,
    ):
        raise ValueError(
            "Judge output was not a JSON object."
        )

    return parsed


def _score_0_to_5(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        score = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    return max(
        0.0,
        min(
            5.0,
            score,
        ),
    )


class RAGEvaluationRunner:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _generate_answer(
        self,
        *,
        question: str,
        context: str,
        model_spec: str,
        explanation_level: str,
    ) -> str:
        if not context.strip():
            return (
                "The retrieved paper context does "
                "not contain enough evidence to "
                "answer this question."
            )

        model = get_chat_model_from_spec(
            model_spec=model_spec,
            temperature=0.0,
            json_mode=False,
        )

        result = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are evaluating a research-paper "
                        "RAG system. Answer only from the "
                        "retrieved context. Do not invent facts. "
                        "When possible, cite evidence using "
                        "[Page N]. If the evidence is "
                        "insufficient, state that clearly."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Explanation level: "
                        f"{explanation_level}\n\n"
                        f"Question:\n{question}\n\n"
                        f"Retrieved context:\n{context}"
                    )
                ),
            ]
        )

        return message_text(
            result
        ).strip()

    def _judge_answer(
        self,
        *,
        question: str,
        answer: str,
        context: str,
        expected_answer: str,
        model_spec: str,
    ) -> EvaluationJudgeResult:
        model = get_chat_model_from_spec(
            model_spec=model_spec,
            temperature=(
                self.settings
                .evaluation_judge_temperature
            ),
            json_mode=True,
        )

        judge_prompt = {
            "question": question,
            "answer": answer,
            "retrieved_context": (
                context
            ),
            "expected_answer": (
                expected_answer
            ),
            "instructions": {
                "relevance": (
                    "Score 0 to 5 for whether the "
                    "answer directly addresses the question."
                ),
                "groundedness": (
                    "Score 0 to 5 for whether the answer "
                    "is supported by retrieved context."
                ),
                "completeness": (
                    "Score 0 to 5 for whether the answer "
                    "covers the important information."
                ),
                "output": (
                    "Return only JSON with relevance, "
                    "groundedness, completeness, "
                    "and justification."
                ),
            },
        }

        result = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a strict RAG evaluator. "
                        "Use only the supplied question, "
                        "answer, context, and optional "
                        "expected answer."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        judge_prompt,
                        ensure_ascii=False,
                    )
                ),
            ]
        )

        payload = _extract_json_object(
            message_text(
                result
            )
        )

        return EvaluationJudgeResult(
            relevance=_score_0_to_5(
                payload.get(
                    "relevance"
                )
            ),
            groundedness=_score_0_to_5(
                payload.get(
                    "groundedness"
                )
            ),
            completeness=_score_0_to_5(
                payload.get(
                    "completeness"
                )
            ),
            justification=str(
                payload.get(
                    "justification",
                    "",
                )
            ),
        )

    @staticmethod
    def _aggregate(
        results: list[
            EvaluationCaseResult
        ],
    ) -> dict[str, float]:
        metric_names: set[str] = set()

        for result in results:
            metric_names.update(
                result.metrics.keys()
            )

        aggregate: dict[
            str,
            float,
        ] = {}

        for metric_name in sorted(
            metric_names
        ):
            average = safe_mean(
                result.metrics.get(
                    metric_name
                )
                for result in results
            )

            if average is not None:
                aggregate[
                    metric_name
                ] = average

        successful_cases = sum(
            1
            for result in results
            if not result.error
        )

        aggregate[
            "successful_case_ratio"
        ] = (
            successful_cases
            / len(results)
            if results
            else 0.0
        )

        return aggregate

    def run(
        self,
        *,
        session_id: str,
        dataset_name: str,
        cases: list[
            EvaluationCase
        ],
        top_k: int,
        minimum_score: float,
        run_generation: bool,
        use_llm_judge: bool,
        model_spec: str,
        explanation_level: str,
    ) -> RAGEvaluationResponse:
        if (
            len(cases)
            > self.settings
            .evaluation_max_cases
        ):
            raise APIError(
                status_code=400,
                code=(
                    "too_many_evaluation_cases"
                ),
                detail=(
                    "The evaluation dataset exceeds "
                    f"{self.settings.evaluation_max_cases} "
                    "cases."
                ),
            )

        session = session_store.get(
            session_id
        )

        if not session.index_ready:
            raise APIError(
                status_code=409,
                code="rag_index_required",
                detail=(
                    "Build a FAISS or Pinecone RAG "
                    "index before running evaluation."
                ),
            )

        (
            model_provider,
            model_name,
            fallback_enabled,
        ) = parse_model_spec(
            model_spec
        )

        results: list[
            EvaluationCaseResult
        ] = []

        warnings: list[str] = []

        if (
            use_llm_judge
            and not run_generation
        ):
            warnings.append(
                "LLM judging was skipped because "
                "answer generation is disabled."
            )

        for index, case in enumerate(
            cases,
            start=1,
        ):
            case_id = (
                case.case_id.strip()
                or f"case-{index:03d}"
            )

            try:
                retrieval_started = (
                    time.perf_counter()
                )

                context, raw_sources = (
                    retrieve_session_context(
                        session=session,
                        query=case.question,
                        top_k=top_k,
                        minimum_score=(
                            minimum_score
                        ),
                    )
                )

                retrieval_latency_ms = (
                    (
                        time.perf_counter()
                        - retrieval_started
                    )
                    * 1000
                )

                sources = [
                    _normalize_source(
                        source,
                        default_rank=rank,
                    )
                    for rank, source
                    in enumerate(
                        raw_sources,
                        start=1,
                    )
                ]

                retrieved_pages = (
                    unique_in_order(
                        source.page_number
                        for source in sources
                        if source.page_number > 0
                    )
                )

                answer = ""
                generation_latency_ms = 0.0

                if run_generation:
                    generation_started = (
                        time.perf_counter()
                    )

                    answer = (
                        self._generate_answer(
                            question=(
                                case.question
                            ),
                            context=context,
                            model_spec=(
                                model_spec
                            ),
                            explanation_level=(
                                explanation_level
                            ),
                        )
                    )

                    generation_latency_ms = (
                        (
                            time.perf_counter()
                            - generation_started
                        )
                        * 1000
                    )

                retrieval_metrics = (
                    retrieval_page_metrics(
                        expected_pages=(
                            case.expected_pages
                        ),
                        retrieved_pages=(
                            retrieved_pages
                        ),
                        top_k=top_k,
                    )
                )

                top_score = (
                    sources[0].score
                    if sources
                    else 0.0
                )

                mean_score = (
                    sum(
                        source.score
                        for source in sources
                    )
                    / len(sources)
                    if sources
                    else 0.0
                )

                metrics: dict[
                    str,
                    float | None,
                ] = {
                    **retrieval_metrics,
                    "top_1_similarity": (
                        float(top_score)
                    ),
                    "mean_retrieved_similarity": (
                        float(mean_score)
                    ),
                    "retrieved_source_count": (
                        float(len(sources))
                    ),
                    "retrieval_latency_ms": (
                        float(
                            retrieval_latency_ms
                        )
                    ),
                    "generation_latency_ms": (
                        float(
                            generation_latency_ms
                        )
                    ),
                }

                if run_generation:
                    metrics[
                        "answer_keyword_coverage"
                    ] = answer_keyword_coverage(
                        expected_answer=(
                            case.expected_answer
                        ),
                        answer=answer,
                    )

                    metrics[
                        "answer_context_overlap"
                    ] = answer_context_overlap(
                        answer=answer,
                        context=context,
                    )

                judge_result = None

                if (
                    run_generation
                    and use_llm_judge
                ):
                    judge_result = (
                        self._judge_answer(
                            question=(
                                case.question
                            ),
                            answer=answer,
                            context=context,
                            expected_answer=(
                                case.expected_answer
                            ),
                            model_spec=(
                                model_spec
                            ),
                        )
                    )

                    metrics[
                        "judge_relevance_0_to_5"
                    ] = (
                        judge_result.relevance
                    )

                    metrics[
                        "judge_groundedness_0_to_5"
                    ] = (
                        judge_result
                        .groundedness
                    )

                    metrics[
                        "judge_completeness_0_to_5"
                    ] = (
                        judge_result
                        .completeness
                    )

                results.append(
                    EvaluationCaseResult(
                        case_id=case_id,
                        question=(
                            case.question
                        ),
                        expected_pages=(
                            case.expected_pages
                        ),
                        expected_answer=(
                            case.expected_answer
                        ),
                        retrieved_pages=(
                            retrieved_pages
                        ),
                        retrieved_sources=(
                            sources
                        ),
                        answer=answer,
                        metrics=metrics,
                        judge=judge_result,
                    )
                )

            except Exception as exc:
                results.append(
                    EvaluationCaseResult(
                        case_id=case_id,
                        question=(
                            case.question
                        ),
                        expected_pages=(
                            case.expected_pages
                        ),
                        expected_answer=(
                            case.expected_answer
                        ),
                        retrieved_pages=[],
                        retrieved_sources=[],
                        answer="",
                        metrics={},
                        judge=None,
                        error=str(exc),
                    )
                )

        aggregate_metrics = (
            self._aggregate(
                results
            )
        )

        run_name = (
            f"{dataset_name}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        dataset_payload = {
            "dataset_name": dataset_name,
            "session_id": session_id,
            "cases": [
                case.model_dump(
                    mode="json"
                )
                for case in cases
            ],
        }

        response = RAGEvaluationResponse(
            status="completed",
            session_id=session_id,
            dataset_name=dataset_name,
            mlflow_experiment_name=(
                self.settings
                .mlflow_experiment_name
            ),
            case_count=len(results),
            aggregate_metrics=(
                aggregate_metrics
            ),
            cases=results,
            warnings=warnings,
        )

        parameters = {
            "session_id": session_id,
            "dataset_name": dataset_name,
            "case_count": len(cases),
            "top_k": top_k,
            "minimum_score": (
                minimum_score
            ),
            "run_generation": (
                run_generation
            ),
            "use_llm_judge": (
                use_llm_judge
            ),
            "llm_provider": (
                model_provider.value
            ),
            "llm_model": model_name,
            "llm_fallback_enabled": (
                fallback_enabled
            ),
            "embedding_model": (
                session
                .embedding_model_name
            ),
            "embedding_device": (
                session.embedding_device
            ),
            "vector_backend": (
                session.vector_backend
            ),
            "vector_dimension": (
                session.vector_dimension
            ),
        }

        try:
            run_id = (
                mlflow_tracker
                .log_evaluation(
                    run_name=run_name,
                    parameters=parameters,
                    aggregate_metrics=(
                        aggregate_metrics
                    ),
                    dataset_payload=(
                        dataset_payload
                    ),
                    result_payload=(
                        response.model_dump(
                            mode="json"
                        )
                    ),
                    tags={
                        "application": (
                            "ResearchEase AI"
                        ),
                        "version": "11",
                        "evaluation_type": (
                            "rag"
                        ),
                        "vector_backend": (
                            session.vector_backend
                        ),
                    },
                )
            )

            response.mlflow_run_id = (
                run_id
            )

        except Exception as exc:
            response.warnings.append(
                (
                    "Evaluation completed, but "
                    "MLflow logging failed: "
                    f"{exc}"
                )
            )

        return response


rag_evaluation_runner = (
    RAGEvaluationRunner()
)