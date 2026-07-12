from __future__ import annotations

import gc
import inspect
import json
import shutil
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any, Callable

import torch

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    __version__ as transformers_version,
)

from backend.core.settings import (
    get_settings,
)
from backend.fine_tuning.dataset import (
    prepare_instruction_datasets,
)
from backend.fine_tuning.registry import (
    adapter_registry,
)
from backend.fine_tuning.schemas import (
    FineTuneRequest,
    PrecisionMode,
)


ProgressCallback = Callable[
    [
        str,
        float,
        dict[str, Any],
    ],
    None,
]


def _json_safe(
    payload: dict[str, Any],
) -> dict[str, Any]:
    safe: dict[str, Any] = {}

    for key, value in payload.items():
        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ) or value is None:
            safe[key] = value
        else:
            try:
                safe[key] = float(value)
            except (
                TypeError,
                ValueError,
            ):
                safe[key] = str(value)

    return safe


def _transformers_major() -> int:
    try:
        return int(
            transformers_version.split(
                ".",
                maxsplit=1,
            )[0]
        )
    except (
        ValueError,
        IndexError,
    ):
        return 4


def _model_dtype_arguments(
    dtype: torch.dtype,
) -> dict[str, Any]:
    if _transformers_major() >= 5:
        return {
            "dtype": dtype
        }

    return {
        "torch_dtype": dtype
    }


def _target_modules(
    value: str,
):
    cleaned = value.strip()

    if cleaned == "all-linear":
        return "all-linear"

    modules = [
        module.strip()
        for module in cleaned.split(",")
        if module.strip()
    ]

    if not modules:
        raise ValueError(
            "At least one LoRA target module "
            "must be specified."
        )

    return modules


class TrainingProgressCallback(
    TrainerCallback
):
    def __init__(
        self,
        callback: ProgressCallback,
    ) -> None:
        self.callback = callback

    def on_train_begin(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        self.callback(
            "training",
            0.15,
            {
                "global_step": 0,
                "max_steps": (
                    state.max_steps
                ),
            },
        )

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs,
    ):
        logs = logs or {}

        if state.max_steps > 0:
            fraction = (
                state.global_step
                / state.max_steps
            )
        else:
            fraction = 0.0

        progress = min(
            0.95,
            0.15
            + (
                max(
                    0.0,
                    fraction,
                )
                * 0.8
            ),
        )

        self.callback(
            "training",
            progress,
            {
                **_json_safe(logs),
                "global_step": (
                    state.global_step
                ),
                "max_steps": (
                    state.max_steps
                ),
            },
        )


class LoRATrainerService:
    def __init__(self) -> None:
        self.settings = get_settings()

    @staticmethod
    def _device_and_precision(
        request: FineTuneRequest,
    ) -> tuple[
        str,
        torch.dtype,
        bool,
    ]:
        cuda_available = (
            torch.cuda.is_available()
        )

        if (
            request.precision
            == PrecisionMode.FP16
            and not cuda_available
        ):
            raise RuntimeError(
                "FP16 training was requested, "
                "but CUDA is unavailable."
            )

        if cuda_available:
            device = "cuda"

            use_fp16 = (
                request.precision
                != PrecisionMode.FP32
            )

            dtype = (
                torch.float16
                if use_fp16
                else torch.float32
            )

        else:
            device = "cpu"
            use_fp16 = False
            dtype = torch.float32

        return (
            device,
            dtype,
            use_fp16,
        )

    @staticmethod
    def _training_arguments(
        *,
        request: FineTuneRequest,
        output_dir: Path,
        has_evaluation: bool,
        use_fp16: bool,
        device: str,
    ) -> TrainingArguments:
        kwargs: dict[
            str,
            Any,
        ] = {
            "output_dir": str(
                output_dir
            ),
            "overwrite_output_dir": True,
            "num_train_epochs": (
                request.epochs
            ),
            "max_steps": (
                request.max_steps
            ),
            "per_device_train_batch_size": (
                request
                .per_device_batch_size
            ),
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": (
                request
                .gradient_accumulation_steps
            ),
            "learning_rate": (
                request.learning_rate
            ),
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": (
                "cosine"
            ),
            "logging_strategy": "steps",
            "logging_steps": 1,
            "logging_first_step": True,
            "save_strategy": "no",
            "report_to": [],
            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": (
                device == "cuda"
            ),
            "fp16": use_fp16,
            "bf16": False,
            "gradient_checkpointing": (
                request
                .gradient_checkpointing
            ),
            "max_grad_norm": 1.0,
            "seed": request.seed,
            "data_seed": request.seed,
            "optim": "adamw_torch",
        }

        signature = inspect.signature(
            TrainingArguments.__init__
        ).parameters

        evaluation_strategy = (
            "epoch"
            if has_evaluation
            else "no"
        )

        if "eval_strategy" in signature:
            kwargs[
                "eval_strategy"
            ] = evaluation_strategy

        elif (
            "evaluation_strategy"
            in signature
        ):
            kwargs[
                "evaluation_strategy"
            ] = evaluation_strategy

        if (
            device == "cpu"
            and "use_cpu" in signature
        ):
            kwargs["use_cpu"] = True

        elif (
            device == "cpu"
            and "no_cuda" in signature
        ):
            kwargs["no_cuda"] = True

        accepted_kwargs = {
            key: value
            for key, value
            in kwargs.items()
            if key in signature
        }

        return TrainingArguments(
            **accepted_kwargs
        )

    @staticmethod
    def _trainer(
        *,
        model,
        arguments,
        training_dataset,
        evaluation_dataset,
        data_collator,
        tokenizer,
        callbacks,
    ) -> Trainer:
        kwargs = {
            "model": model,
            "args": arguments,
            "train_dataset": (
                training_dataset
            ),
            "eval_dataset": (
                evaluation_dataset
            ),
            "data_collator": (
                data_collator
            ),
            "callbacks": callbacks,
        }

        trainer_signature = (
            inspect.signature(
                Trainer.__init__
            ).parameters
        )

        if (
            "processing_class"
            in trainer_signature
        ):
            kwargs[
                "processing_class"
            ] = tokenizer

        elif (
            "tokenizer"
            in trainer_signature
        ):
            kwargs[
                "tokenizer"
            ] = tokenizer

        return Trainer(**kwargs)

    def train(
        self,
        *,
        job_id: str,
        request: FineTuneRequest,
        progress_callback: (
            ProgressCallback
        ),
    ) -> dict[str, Any]:
        final_directory = (
            adapter_registry
            .ensure_available(
                request.adapter_name
            )
        )

        temporary_directory = (
            self.settings
            .fine_tuning_temp_dir
            / job_id
        ).resolve()

        checkpoint_directory = (
            temporary_directory
            / "checkpoints"
        )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        model = None
        tokenizer = None
        trainer = None

        try:
            progress_callback(
                "loading_tokenizer",
                0.03,
                {},
            )

            tokenizer = (
                AutoTokenizer
                .from_pretrained(
                    request.base_model,
                    use_fast=True,
                )
            )

            if (
                tokenizer.pad_token_id
                is None
            ):
                tokenizer.pad_token = (
                    tokenizer.eos_token
                )

            tokenizer.padding_side = (
                "right"
            )

            progress_callback(
                "preparing_dataset",
                0.06,
                {},
            )

            (
                training_dataset,
                evaluation_dataset,
                data_collator,
                dataset_statistics,
            ) = prepare_instruction_datasets(
                tokenizer=tokenizer,
                examples=request.examples,
                max_sequence_length=(
                    request
                    .max_sequence_length
                ),
                validation_split=(
                    request
                    .validation_split
                ),
                seed=request.seed,
            )

            (
                device,
                dtype,
                use_fp16,
            ) = self._device_and_precision(
                request
            )

            progress_callback(
                "loading_base_model",
                0.09,
                {
                    "device": device,
                    "dtype": str(dtype),
                },
            )

            model = (
                AutoModelForCausalLM
                .from_pretrained(
                    request.base_model,
                    low_cpu_mem_usage=True,
                    **_model_dtype_arguments(
                        dtype
                    ),
                )
            )

            model.config.pad_token_id = (
                tokenizer.pad_token_id
            )

            model.config.use_cache = False

            if (
                request
                .gradient_checkpointing
            ):
                if hasattr(
                    model,
                    (
                        "gradient_"
                        "checkpointing_enable"
                    ),
                ):
                    model.gradient_checkpointing_enable()

                if hasattr(
                    model,
                    (
                        "enable_input_"
                        "require_grads"
                    ),
                ):
                    model.enable_input_require_grads()

            progress_callback(
                "injecting_lora",
                0.12,
                {},
            )

            lora_configuration = (
                LoraConfig(
                    task_type=(
                        TaskType.CAUSAL_LM
                    ),
                    inference_mode=False,
                    r=request.lora_r,
                    lora_alpha=(
                        request.lora_alpha
                    ),
                    lora_dropout=(
                        request
                        .lora_dropout
                    ),
                    target_modules=(
                        _target_modules(
                            request
                            .target_modules
                        )
                    ),
                    bias="none",
                )
            )

            model = get_peft_model(
                model,
                lora_configuration,
            )

            trainable_parameters = sum(
                parameter.numel()
                for parameter
                in model.parameters()
                if parameter.requires_grad
            )

            total_parameters = sum(
                parameter.numel()
                for parameter
                in model.parameters()
            )

            trainable_percentage = (
                trainable_parameters
                / total_parameters
                * 100
                if total_parameters
                else 0.0
            )

            arguments = (
                self._training_arguments(
                    request=request,
                    output_dir=(
                        checkpoint_directory
                    ),
                    has_evaluation=(
                        evaluation_dataset
                        is not None
                    ),
                    use_fp16=use_fp16,
                    device=device,
                )
            )

            callbacks = [
                TrainingProgressCallback(
                    progress_callback
                )
            ]

            trainer = self._trainer(
                model=model,
                arguments=arguments,
                training_dataset=(
                    training_dataset
                ),
                evaluation_dataset=(
                    evaluation_dataset
                ),
                data_collator=(
                    data_collator
                ),
                tokenizer=tokenizer,
                callbacks=callbacks,
            )

            train_result = trainer.train()

            training_metrics = (
                _json_safe(
                    train_result.metrics
                )
            )

            evaluation_metrics: dict[
                str,
                Any,
            ] = {}

            if (
                evaluation_dataset
                is not None
            ):
                progress_callback(
                    "evaluating",
                    0.96,
                    {},
                )

                evaluation_metrics = (
                    _json_safe(
                        trainer.evaluate()
                    )
                )

            progress_callback(
                "saving_adapter",
                0.98,
                {},
            )

            final_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            model.config.use_cache = True

            model.save_pretrained(
                final_directory,
                safe_serialization=True,
            )

            tokenizer.save_pretrained(
                final_directory
            )

            dataset_path = (
                final_directory
                / "training_dataset.jsonl"
            )

            with dataset_path.open(
                "w",
                encoding="utf-8",
            ) as dataset_file:
                for example in (
                    request.examples
                ):
                    dataset_file.write(
                        json.dumps(
                            example.model_dump(
                                mode="json"
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            created_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            metadata = {
                "adapter_name": (
                    request.adapter_name
                ),
                "base_model": (
                    request.base_model
                ),
                "created_at": created_at,
                "device": device,
                "precision": (
                    request
                    .precision.value
                ),
                "transformers_version": (
                    transformers_version
                ),
                "trainable_parameters": (
                    trainable_parameters
                ),
                "total_parameters": (
                    total_parameters
                ),
                "trainable_percentage": (
                    trainable_percentage
                ),
                "training_examples": (
                    dataset_statistics[
                        "training_examples"
                    ]
                ),
                "evaluation_examples": (
                    dataset_statistics[
                        "evaluation_examples"
                    ]
                ),
                "dataset_statistics": (
                    dataset_statistics
                ),
                "training_metrics": (
                    training_metrics
                ),
                "evaluation_metrics": (
                    evaluation_metrics
                ),
                "training_configuration": (
                    request.model_dump(
                        mode="json",
                        exclude={
                            "examples"
                        },
                    )
                ),
            }

            adapter_registry.write_metadata(
                request.adapter_name,
                metadata,
            )

            progress_callback(
                "completed",
                1.0,
                {
                    **training_metrics,
                    **evaluation_metrics,
                },
            )

            return {
                "adapter_name": (
                    request.adapter_name
                ),
                "adapter_path": str(
                    final_directory
                ),
                "base_model": (
                    request.base_model
                ),
                "trainable_parameters": (
                    trainable_parameters
                ),
                "total_parameters": (
                    total_parameters
                ),
                "trainable_percentage": (
                    trainable_percentage
                ),
                "training_metrics": (
                    training_metrics
                ),
                "evaluation_metrics": (
                    evaluation_metrics
                ),
                "dataset_statistics": (
                    dataset_statistics
                ),
            }

        except Exception:
            if final_directory.exists():
                shutil.rmtree(
                    final_directory,
                    ignore_errors=True,
                )

            raise

        finally:
            del trainer
            del model
            del tokenizer

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )


lora_trainer_service = (
    LoRATrainerService()
)