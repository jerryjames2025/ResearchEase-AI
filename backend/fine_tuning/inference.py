from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from backend.fine_tuning.registry import (
    adapter_registry,
)
from backend.fine_tuning.trainer import (
    _model_dtype_arguments,
)


def _clear_memory() -> None:
    """
    Release Python and CUDA memory as aggressively
    as possible between inference attempts.
    """

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _is_cuda_out_of_memory(
    error: BaseException,
) -> bool:
    message = str(error).lower()

    return (
        "cuda out of memory" in message
        or "cuda error: out of memory" in message
        or "cublas_status_alloc_failed" in message
    )


class AdapterInferenceService:
    def _generate_on_device(
        self,
        *,
        adapter_name: str,
        adapter_path: Path,
        base_model_name: str,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        device: str,
    ) -> dict[str, Any]:
        tokenizer = None
        base_model = None
        model = None
        inputs = None
        generated = None

        started = time.perf_counter()

        dtype = (
            torch.float16
            if device == "cuda"
            else torch.float32
        )

        try:
            tokenizer = (
                AutoTokenizer.from_pretrained(
                    adapter_path,
                    use_fast=True,
                )
            )

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = (
                    tokenizer.eos_token
                )

            tokenizer.padding_side = "left"

            base_model = (
                AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    low_cpu_mem_usage=True,
                    **_model_dtype_arguments(
                        dtype
                    ),
                )
            )

            base_model.config.use_cache = True
            base_model.config.pad_token_id = (
                tokenizer.pad_token_id
            )

            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                is_trainable=False,
            )

            model.to(device)
            model.eval()

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            rendered_prompt = (
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            inputs = tokenizer(
                rendered_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            generation_arguments: dict[
                str,
                Any,
            ] = {
                "max_new_tokens": (
                    max_new_tokens
                ),
                "pad_token_id": (
                    tokenizer.pad_token_id
                ),
                "eos_token_id": (
                    tokenizer.eos_token_id
                ),
                "do_sample": (
                    temperature > 0
                ),
                "use_cache": True,
            }

            if temperature > 0:
                generation_arguments[
                    "temperature"
                ] = temperature

                generation_arguments[
                    "top_p"
                ] = top_p

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    **generation_arguments,
                )

            prompt_length = (
                inputs["input_ids"].shape[-1]
            )

            generated_tokens = generated[
                0,
                prompt_length:,
            ]

            response = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            latency = (
                time.perf_counter()
                - started
            )

            return {
                "adapter_name": adapter_name,
                "base_model": (
                    base_model_name
                ),
                "response": response,
                "device": device,
                "generation_latency_seconds": (
                    float(latency)
                ),
            }

        finally:
            try:
                del generated
            except Exception:
                pass

            try:
                del inputs
            except Exception:
                pass

            try:
                del model
            except Exception:
                pass

            try:
                del base_model
            except Exception:
                pass

            try:
                del tokenizer
            except Exception:
                pass

            _clear_memory()

    def generate(
        self,
        *,
        adapter_name: str,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        metadata = adapter_registry.metadata(
            adapter_name
        )

        adapter_path = (
            adapter_registry.adapter_path(
                adapter_name
            )
        )

        base_model_name = (
            metadata.get("base_model")
            or metadata.get(
                "metadata",
                {},
            ).get(
                "base_model",
                "",
            )
        )

        if not base_model_name:
            raise RuntimeError(
                "The adapter metadata does not "
                "contain its base-model name."
            )

        _clear_memory()

        if torch.cuda.is_available():
            try:
                return self._generate_on_device(
                    adapter_name=adapter_name,
                    adapter_path=adapter_path,
                    base_model_name=(
                        base_model_name
                    ),
                    prompt=prompt,
                    system_prompt=(
                        system_prompt
                    ),
                    max_new_tokens=(
                        max_new_tokens
                    ),
                    temperature=temperature,
                    top_p=top_p,
                    device="cuda",
                )

            except RuntimeError as exc:
                if not _is_cuda_out_of_memory(
                    exc
                ):
                    raise

                _clear_memory()

                result = (
                    self._generate_on_device(
                        adapter_name=(
                            adapter_name
                        ),
                        adapter_path=(
                            adapter_path
                        ),
                        base_model_name=(
                            base_model_name
                        ),
                        prompt=prompt,
                        system_prompt=(
                            system_prompt
                        ),
                        max_new_tokens=(
                            max_new_tokens
                        ),
                        temperature=(
                            temperature
                        ),
                        top_p=top_p,
                        device="cpu",
                    )
                )

                result["device"] = (
                    "cpu — CUDA OOM fallback"
                )

                return result

        return self._generate_on_device(
            adapter_name=adapter_name,
            adapter_path=adapter_path,
            base_model_name=(
                base_model_name
            ),
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device="cpu",
        )


adapter_inference_service = (
    AdapterInferenceService()
)