from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class PrecisionMode(
    str,
    Enum,
):
    AUTO = "auto"
    FP32 = "fp32"
    FP16 = "fp16"


class FineTuneJobStatus(
    str,
    Enum,
):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class InstructionExample(BaseModel):
    instruction: str = Field(
        min_length=2,
        max_length=8000,
    )

    input: str = Field(
        default="",
        max_length=12000,
    )

    output: str = Field(
        min_length=2,
        max_length=16000,
    )

    system_prompt: str = Field(
        default=(
            "You are ResearchEase AI, "
            "an accurate academic research assistant."
        ),
        max_length=4000,
    )

    @field_validator(
        "instruction",
        "input",
        "output",
        "system_prompt",
    )
    @classmethod
    def strip_text(
        cls,
        value: str,
    ) -> str:
        return value.strip()


class DatasetValidationRequest(BaseModel):
    examples: list[
        InstructionExample
    ] = Field(
        min_length=1,
        max_length=500,
    )


class DatasetValidationResponse(BaseModel):
    valid: bool

    example_count: int
    duplicate_count: int
    average_instruction_characters: float
    average_output_characters: float

    warnings: list[str] = Field(
        default_factory=list
    )

    preview: list[dict] = Field(
        default_factory=list
    )


class FineTuneRequest(BaseModel):
    adapter_name: str = Field(
        min_length=2,
        max_length=80,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$",
    )

    base_model: str = Field(
        default=(
            "Qwen/Qwen2.5-0.5B-Instruct"
        ),
        min_length=2,
        max_length=300,
    )

    examples: list[
        InstructionExample
    ] = Field(
        min_length=4,
        max_length=500,
    )

    validation_split: float = Field(
        default=0.1,
        ge=0.0,
        le=0.3,
    )

    epochs: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
    )

    max_steps: int = -1

    learning_rate: float = Field(
        default=0.0002,
        gt=0.0,
        le=0.01,
    )

    per_device_batch_size: int = Field(
        default=1,
        ge=1,
        le=8,
    )

    gradient_accumulation_steps: int = Field(
        default=8,
        ge=1,
        le=128,
    )

    max_sequence_length: int = Field(
        default=256,
        ge=64,
        le=2048,
    )

    lora_r: int = Field(
        default=8,
        ge=1,
        le=128,
    )

    lora_alpha: int = Field(
        default=16,
        ge=1,
        le=512,
    )

    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
    )

    target_modules: str = Field(
        default="q_proj,v_proj",
        min_length=2,
        max_length=300,
    )

    precision: PrecisionMode = (
        PrecisionMode.AUTO
    )

    gradient_checkpointing: bool = True

    seed: int = Field(
        default=42,
        ge=0,
        le=2_147_483_647,
    )

    @field_validator("max_steps")
    @classmethod
    def validate_max_steps(
        cls,
        value: int,
    ) -> int:
        if value != -1 and value < 1:
            raise ValueError(
                "max_steps must be -1 or "
                "a positive integer."
            )

        return value


class FineTuneJobResponse(BaseModel):
    job_id: str
    adapter_name: str

    status: FineTuneJobStatus

    stage: str = ""
    progress: float = 0.0

    created_at: str
    started_at: str = ""
    completed_at: str = ""

    metrics: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    result: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    error: str = ""


class FineTuneJobListResponse(BaseModel):
    jobs: list[
        FineTuneJobResponse
    ]


class AdapterInfo(BaseModel):
    adapter_name: str
    path: str

    base_model: str = ""
    created_at: str = ""

    trainable_parameters: int = 0
    total_parameters: int = 0
    trainable_percentage: float = 0.0

    training_examples: int = 0
    evaluation_examples: int = 0

    size_bytes: int = 0

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class AdapterListResponse(BaseModel):
    adapters: list[
        AdapterInfo
    ]


class FineTuneGenerateRequest(BaseModel):
    adapter_name: str = Field(
        min_length=2,
        max_length=80,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$",
    )

    prompt: str = Field(
        min_length=2,
        max_length=8000,
    )

    system_prompt: str = Field(
        default=(
            "You are ResearchEase AI, "
            "an accurate academic research assistant."
        ),
        max_length=4000,
    )

    max_new_tokens: int = Field(
        default=256,
        ge=1,
        le=1024,
    )

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
    )

    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
    )


class FineTuneGenerateResponse(BaseModel):
    adapter_name: str
    base_model: str

    response: str

    device: str
    generation_latency_seconds: float


class FineTuneHealthResponse(BaseModel):
    status: str

    torch_available: bool
    cuda_available: bool

    device: str
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0

    base_model: str
    adapter_directory: str

    detail: str