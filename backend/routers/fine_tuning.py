from __future__ import annotations

import torch

from fastapi import (
    APIRouter,
    Query,
    status,
)

from backend.core.errors import APIError
from backend.core.settings import (
    get_settings,
)
from backend.fine_tuning.dataset import (
    validate_instruction_examples,
)
from backend.fine_tuning.inference import (
    adapter_inference_service,
)
from backend.fine_tuning.jobs import (
    fine_tune_job_manager,
)
from backend.fine_tuning.registry import (
    AdapterRegistryError,
    adapter_registry,
)
from backend.fine_tuning.schemas import (
    AdapterInfo,
    AdapterListResponse,
    DatasetValidationRequest,
    DatasetValidationResponse,
    FineTuneGenerateRequest,
    FineTuneGenerateResponse,
    FineTuneHealthResponse,
    FineTuneJobListResponse,
    FineTuneJobResponse,
    FineTuneRequest,
)


router = APIRouter(
    prefix="/fine-tuning",
    tags=["LoRA fine-tuning"],
)


@router.get(
    "/health",
    response_model=(
        FineTuneHealthResponse
    ),
)
def fine_tuning_health() -> (
    FineTuneHealthResponse
):
    settings = get_settings()

    cuda_available = (
        torch.cuda.is_available()
    )

    gpu_name = ""
    gpu_memory_gb = 0.0

    if cuda_available:
        gpu_name = (
            torch.cuda
            .get_device_name(0)
        )

        properties = (
            torch.cuda
            .get_device_properties(0)
        )

        gpu_memory_gb = (
            properties.total_memory
            / 1024**3
        )

    return FineTuneHealthResponse(
        status="healthy",
        torch_available=True,
        cuda_available=(
            cuda_available
        ),
        device=(
            "cuda"
            if cuda_available
            else "cpu"
        ),
        gpu_name=gpu_name,
        gpu_memory_gb=(
            float(gpu_memory_gb)
        ),
        base_model=(
            settings
            .fine_tuning_base_model
        ),
        adapter_directory=str(
            settings
            .fine_tuning_output_dir
        ),
        detail=(
            "CUDA fine-tuning is available."
            if cuda_available
            else (
                "CUDA is unavailable. "
                "Fine-tuning will use CPU."
            )
        ),
    )


@router.get("/template")
def fine_tuning_template() -> dict:
    examples = [
        {
            "instruction": (
                "Explain the main research "
                "objective clearly."
            ),
            "input": (
                "The study develops a transformer "
                "system for academic-document analysis."
            ),
            "output": (
                "The primary objective is to build "
                "a transformer-based system that helps "
                "researchers analyze academic documents "
                "more efficiently and consistently."
            ),
        },
        {
            "instruction": (
                "Summarize the methodology."
            ),
            "input": (
                "The pipeline extracts PDF text, "
                "creates embeddings, indexes chunks, "
                "retrieves relevant evidence, and "
                "generates grounded answers."
            ),
            "output": (
                "The methodology follows five stages: "
                "PDF text extraction, semantic chunking, "
                "embedding generation, vector indexing, "
                "and evidence-grounded answer generation."
            ),
        },
        {
            "instruction": (
                "State the limitation without "
                "inventing information."
            ),
            "input": (
                "The evaluation used only twelve "
                "English-language papers."
            ),
            "output": (
                "A key limitation is the small and "
                "language-restricted evaluation set. "
                "Results from twelve English papers may "
                "not generalize to larger multilingual "
                "research collections."
            ),
        },
        {
            "instruction": (
                "Convert the result into a concise "
                "academic finding."
            ),
            "input": (
                "Retrieval accuracy increased from "
                "0.71 to 0.84 after chunk optimization."
            ),
            "output": (
                "Chunk optimization improved retrieval "
                "accuracy from 0.71 to 0.84, representing "
                "an absolute gain of 0.13."
            ),
        },
        {
            "instruction": (
                "Explain why RAG is used."
            ),
            "input": (
                "The assistant must answer from "
                "uploaded paper evidence."
            ),
            "output": (
                "RAG is used to retrieve relevant "
                "passages from the uploaded paper before "
                "generation, helping the assistant ground "
                "its response in document evidence."
            ),
        },
        {
            "instruction": (
                "Write a structured contribution list."
            ),
            "input": (
                "The work provides PDF analysis, "
                "semantic search, mathematical explanation, "
                "and literature comparison."
            ),
            "output": (
                "The work contributes: (1) automated PDF "
                "analysis, (2) semantic research-paper "
                "retrieval, (3) mathematical explanation, "
                "and (4) multi-paper literature comparison."
            ),
        },
        {
            "instruction": (
                "Identify an appropriate future direction."
            ),
            "input": (
                "The current system processes text-only PDFs."
            ),
            "output": (
                "A suitable future direction is multimodal "
                "document understanding so that figures, "
                "tables, equations, and diagrams can be "
                "analyzed alongside extracted text."
            ),
        },
        {
            "instruction": (
                "Answer conservatively when evidence "
                "is insufficient."
            ),
            "input": (
                "The provided text does not report "
                "the training dataset size."
            ),
            "output": (
                "The available evidence does not specify "
                "the training dataset size, so a reliable "
                "number cannot be provided."
            ),
        },
    ]

    return {
        "adapter_name": (
            "researchease-academic-style"
        ),
        "base_model": (
            get_settings()
            .fine_tuning_base_model
        ),
        "examples": examples,
    }


@router.post(
    "/validate",
    response_model=(
        DatasetValidationResponse
    ),
)
def validate_dataset(
    request: DatasetValidationRequest,
) -> DatasetValidationResponse:
    return validate_instruction_examples(
        request.examples
    )


@router.post(
    "/jobs",
    response_model=(
        FineTuneJobResponse
    ),
    status_code=(
        status.HTTP_202_ACCEPTED
    ),
)
def create_fine_tuning_job(
    request: FineTuneRequest,
) -> FineTuneJobResponse:
    settings = get_settings()

    if (
        len(request.examples)
        > settings
        .fine_tuning_max_examples
    ):
        raise APIError(
            status_code=400,
            code=(
                "too_many_training_examples"
            ),
            detail=(
                "The dataset exceeds the "
                f"{settings.fine_tuning_max_examples} "
                "example limit."
            ),
        )

    try:
        result = (
            fine_tune_job_manager
            .create_job(request)
        )

    except (
        ValueError,
        AdapterRegistryError,
    ) as exc:
        raise APIError(
            status_code=409,
            code=(
                "fine_tuning_job_conflict"
            ),
            detail=str(exc),
        ) from exc

    return FineTuneJobResponse(
        **result
    )


@router.get(
    "/jobs",
    response_model=(
        FineTuneJobListResponse
    ),
)
def list_fine_tuning_jobs(
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> FineTuneJobListResponse:
    return FineTuneJobListResponse(
        jobs=[
            FineTuneJobResponse(
                **job
            )
            for job
            in fine_tune_job_manager
            .list_jobs(limit)
        ]
    )


@router.get(
    "/jobs/{job_id}",
    response_model=(
        FineTuneJobResponse
    ),
)
def get_fine_tuning_job(
    job_id: str,
) -> FineTuneJobResponse:
    try:
        result = (
            fine_tune_job_manager
            .get_job(job_id)
        )

    except KeyError as exc:
        raise APIError(
            status_code=404,
            code=(
                "fine_tuning_job_not_found"
            ),
            detail=str(exc),
        ) from exc

    return FineTuneJobResponse(
        **result
    )


@router.get(
    "/adapters",
    response_model=(
        AdapterListResponse
    ),
)
def list_adapters() -> (
    AdapterListResponse
):
    return AdapterListResponse(
        adapters=[
            AdapterInfo(**adapter)
            for adapter
            in adapter_registry
            .list_adapters()
        ]
    )


@router.delete(
    "/adapters/{adapter_name}",
)
def delete_adapter(
    adapter_name: str,
) -> dict[str, str]:
    try:
        adapter_registry.delete(
            adapter_name
        )

    except AdapterRegistryError as exc:
        raise APIError(
            status_code=404,
            code="adapter_not_found",
            detail=str(exc),
        ) from exc

    return {
        "message": (
            f"Adapter '{adapter_name}' "
            "was deleted."
        )
    }


@router.post(
    "/generate",
    response_model=(
        FineTuneGenerateResponse
    ),
)
def generate_with_adapter(
    request: FineTuneGenerateRequest,
) -> FineTuneGenerateResponse:
    try:
        result = (
            adapter_inference_service
            .generate(
                adapter_name=(
                    request.adapter_name
                ),
                prompt=request.prompt,
                system_prompt=(
                    request.system_prompt
                ),
                max_new_tokens=(
                    request.max_new_tokens
                ),
                temperature=(
                    request.temperature
                ),
                top_p=request.top_p,
            )
        )

    except AdapterRegistryError as exc:
        raise APIError(
            status_code=404,
            code="adapter_not_found",
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise APIError(
            status_code=503,
            code=(
                "adapter_generation_failed"
            ),
            detail=str(exc),
        ) from exc

    return FineTuneGenerateResponse(
        **result
    )