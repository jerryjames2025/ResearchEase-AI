from __future__ import annotations

from typing import Any

from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
)

from backend.fine_tuning.schemas import (
    DatasetValidationResponse,
    InstructionExample,
)


def validate_instruction_examples(
    examples: list[
        InstructionExample
    ],
) -> DatasetValidationResponse:
    signatures: set[
        tuple[str, str, str]
    ] = set()

    duplicate_count = 0
    warnings: list[str] = []

    instruction_lengths: list[int] = []
    output_lengths: list[int] = []

    for example in examples:
        signature = (
            example.instruction.lower(),
            example.input.lower(),
            example.output.lower(),
        )

        if signature in signatures:
            duplicate_count += 1
        else:
            signatures.add(signature)

        instruction_lengths.append(
            len(example.instruction)
        )

        output_lengths.append(
            len(example.output)
        )

    example_count = len(examples)

    average_instruction = (
        sum(instruction_lengths)
        / example_count
        if example_count
        else 0.0
    )

    average_output = (
        sum(output_lengths)
        / example_count
        if example_count
        else 0.0
    )

    if example_count < 8:
        warnings.append(
            "Fewer than eight examples are suitable "
            "only for a pipeline smoke test."
        )

    if duplicate_count:
        warnings.append(
            f"{duplicate_count} duplicate examples "
            "were detected."
        )

    if average_output < 40:
        warnings.append(
            "Outputs are very short. Fine-tuning works "
            "better when answers demonstrate the desired "
            "response structure and level of detail."
        )

    return DatasetValidationResponse(
        valid=(
            example_count > 0
            and duplicate_count
            < example_count
        ),
        example_count=example_count,
        duplicate_count=duplicate_count,
        average_instruction_characters=(
            float(average_instruction)
        ),
        average_output_characters=(
            float(average_output)
        ),
        warnings=warnings,
        preview=[
            example.model_dump(
                mode="json"
            )
            for example in examples[:3]
        ],
    )


def _user_content(
    example: InstructionExample,
) -> str:
    if example.input:
        return (
            f"{example.instruction}\n\n"
            f"Context:\n{example.input}"
        )

    return example.instruction


def _prompt_text(
    tokenizer,
    example: InstructionExample,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                example.system_prompt
            ),
        },
        {
            "role": "user",
            "content": (
                _user_content(example)
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _full_text(
    tokenizer,
    example: InstructionExample,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                example.system_prompt
            ),
        },
        {
            "role": "user",
            "content": (
                _user_content(example)
            ),
        },
        {
            "role": "assistant",
            "content": example.output,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_instruction_example(
    *,
    tokenizer,
    example: InstructionExample,
    max_sequence_length: int,
) -> dict[str, Any] | None:
    prompt_text = _prompt_text(
        tokenizer,
        example,
    )

    full_text = _full_text(
        tokenizer,
        example,
    )

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_sequence_length,
    )

    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_sequence_length,
    )

    input_ids = list(
        full_tokens["input_ids"]
    )

    attention_mask = list(
        full_tokens["attention_mask"]
    )

    labels = input_ids.copy()

    prompt_length = min(
        len(prompt_tokens["input_ids"]),
        len(labels),
    )

    labels[:prompt_length] = (
        [-100] * prompt_length
    )

    if not any(
        label != -100
        for label in labels
    ):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": (
            attention_mask
        ),
        "labels": labels,
    }


def prepare_instruction_datasets(
    *,
    tokenizer,
    examples: list[
        InstructionExample
    ],
    max_sequence_length: int,
    validation_split: float,
    seed: int,
):
    tokenized_records: list[
        dict[str, Any]
    ] = []

    dropped_examples = 0

    for example in examples:
        record = (
            tokenize_instruction_example(
                tokenizer=tokenizer,
                example=example,
                max_sequence_length=(
                    max_sequence_length
                ),
            )
        )

        if record is None:
            dropped_examples += 1
            continue

        tokenized_records.append(
            record
        )

    if not tokenized_records:
        raise ValueError(
            "All examples were truncated before "
            "their assistant answers. Increase the "
            "maximum sequence length or shorten prompts."
        )

    dataset = Dataset.from_list(
        tokenized_records
    )

    evaluation_dataset = None

    if (
        validation_split > 0
        and len(dataset) >= 8
    ):
        evaluation_count = max(
            1,
            round(
                len(dataset)
                * validation_split
            ),
        )

        evaluation_count = min(
            evaluation_count,
            len(dataset) - 1,
        )

        split = dataset.train_test_split(
            test_size=evaluation_count,
            seed=seed,
            shuffle=True,
        )

        training_dataset = split[
            "train"
        ]

        evaluation_dataset = split[
            "test"
        ]

    else:
        training_dataset = dataset

    data_collator = (
        DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )
    )

    statistics = {
        "original_examples": len(
            examples
        ),
        "usable_examples": len(
            tokenized_records
        ),
        "dropped_examples": (
            dropped_examples
        ),
        "training_examples": len(
            training_dataset
        ),
        "evaluation_examples": (
            len(evaluation_dataset)
            if evaluation_dataset
            is not None
            else 0
        ),
    }

    return (
        training_dataset,
        evaluation_dataset,
        data_collator,
        statistics,
    )