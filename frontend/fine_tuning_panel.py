from __future__ import annotations

import json

import streamlit as st

from frontend.api_client import APIClientError, ResearchEaseAPI


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _format_integer(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def _format_float(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return f"{0.0:.{digits}f}"


def render_fine_tuning_panel(api: ResearchEaseAPI) -> None:
    st.subheader("Local LoRA Fine-Tuning Lab")

    st.caption(
        "Train a small PEFT adapter locally using instruction, "
        "context, and expected-output examples."
    )

    try:
        health = api.fine_tuning_health()

        health_columns = st.columns(4)
        health_columns[0].metric(
            "Device",
            health.get("device", "unknown"),
        )
        health_columns[1].metric(
            "CUDA",
            "Available"
            if health.get("cuda_available")
            else "Unavailable",
        )
        health_columns[2].metric(
            "GPU",
            health.get("gpu_name", "CPU") or "CPU",
        )
        health_columns[3].metric(
            "GPU memory",
            f"{float(health.get('gpu_memory_gb', 0.0)):.2f} GB",
        )

        st.caption(
            "Default base model: "
            f"{health.get('base_model', DEFAULT_BASE_MODEL)}"
        )

        if not health.get("cuda_available"):
            st.warning(
                "CUDA is unavailable, so fine-tuning will run on CPU "
                "and may be very slow."
            )

    except APIClientError as exc:
        st.error(f"Fine-tuning health check failed: {exc}")
        return

    dataset_tab, jobs_tab, adapters_tab = st.tabs(
        [
            "Dataset and training",
            "Training jobs",
            "Adapter testing",
        ]
    )

    # -----------------------------------------------------
    # Dataset and training
    # -----------------------------------------------------

    with dataset_tab:
        if "fine_tuning_dataset" not in st.session_state:
            try:
                template = api.fine_tuning_template()
            except APIClientError:
                template = {
                    "adapter_name": "researchease-adapter",
                    "base_model": DEFAULT_BASE_MODEL,
                    "examples": [],
                }

            st.session_state["fine_tuning_dataset"] = json.dumps(
                template,
                indent=2,
            )

        dataset_text = st.text_area(
            "Fine-tuning dataset JSON",
            key="fine_tuning_dataset",
            height=500,
        )

        settings_columns = st.columns(4)

        epochs = settings_columns[0].number_input(
            "Epochs",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.5,
        )

        max_sequence_length = settings_columns[1].selectbox(
            "Maximum tokens",
            [128, 256, 384, 512],
            index=1,
        )

        lora_r = settings_columns[2].selectbox(
            "LoRA rank",
            [4, 8, 16, 32],
            index=1,
        )

        precision = settings_columns[3].selectbox(
            "Precision",
            ["auto", "fp16", "fp32"],
        )

        second_row = st.columns(4)

        learning_rate = second_row[0].number_input(
            "Learning rate",
            min_value=0.00001,
            max_value=0.01,
            value=0.0002,
            format="%.5f",
        )

        gradient_accumulation = second_row[1].selectbox(
            "Gradient accumulation",
            [1, 2, 4, 8, 16, 32],
            index=3,
        )

        target_modules = second_row[2].selectbox(
            "LoRA target",
            [
                "q_proj,v_proj",
                "q_proj,k_proj,v_proj,o_proj",
                "all-linear",
            ],
        )

        validation_split = second_row[3].number_input(
            "Validation split",
            min_value=0.0,
            max_value=0.3,
            value=0.1,
            step=0.05,
        )

        action_columns = st.columns(2)
        validate_clicked = action_columns[0].button(
            "Validate dataset",
            use_container_width=True,
        )
        train_clicked = action_columns[1].button(
            "Start LoRA training",
            type="primary",
            use_container_width=True,
        )

        try:
            payload = json.loads(dataset_text)

            if not isinstance(payload, dict):
                raise ValueError(
                    "The fine-tuning JSON must be an object."
                )

            adapter_name = str(payload.get("adapter_name", "")).strip()
            base_model = str(
                payload.get("base_model", DEFAULT_BASE_MODEL)
            ).strip()
            examples = payload.get("examples", [])

            if not isinstance(examples, list):
                raise ValueError("'examples' must be a JSON list.")

        except (json.JSONDecodeError, ValueError) as exc:
            if validate_clicked or train_clicked:
                st.error(f"Invalid dataset JSON: {exc}")
            return

        if validate_clicked:
            try:
                validation = api.validate_fine_tuning_dataset(examples)

                if validation.get("valid"):
                    st.success("Dataset structure is valid.")
                else:
                    st.error("Dataset validation failed.")

                example_count = int(
                    validation.get("example_count", 0)
                )
                duplicate_count = int(
                    validation.get("duplicate_count", 0)
                )
                average_instruction = float(
                    validation.get(
                        "average_instruction_characters",
                        0.0,
                    )
                )
                average_output = float(
                    validation.get(
                        "average_output_characters",
                        0.0,
                    )
                )

                metric_columns = st.columns(4)
                metric_columns[0].metric("Examples", example_count)
                metric_columns[1].metric("Duplicates", duplicate_count)
                metric_columns[2].metric(
                    "Average instruction",
                    f"{average_instruction:.0f} chars",
                )
                metric_columns[3].metric(
                    "Average output",
                    f"{average_output:.0f} chars",
                )

                for warning in validation.get("warnings", []):
                    st.warning(str(warning))

            except APIClientError as exc:
                st.error(str(exc))

        if train_clicked:
            if not adapter_name:
                st.error("The dataset must include a non-empty adapter_name.")
            elif len(examples) < 4:
                st.error(
                    "At least four examples are required to start training."
                )
            else:
                try:
                    with st.spinner("Submitting fine-tuning job..."):
                        job = api.start_fine_tuning_job(
                            {
                                "adapter_name": adapter_name,
                                "base_model": base_model,
                                "examples": examples,
                                "validation_split": float(validation_split),
                                "epochs": float(epochs),
                                "max_steps": -1,
                                "learning_rate": float(learning_rate),
                                "per_device_batch_size": 1,
                                "gradient_accumulation_steps": int(
                                    gradient_accumulation
                                ),
                                "max_sequence_length": int(
                                    max_sequence_length
                                ),
                                "lora_r": int(lora_r),
                                "lora_alpha": int(lora_r) * 2,
                                "lora_dropout": 0.05,
                                "target_modules": target_modules,
                                "precision": precision,
                                "gradient_checkpointing": True,
                                "seed": 42,
                            }
                        )

                    st.session_state[
                        "active_fine_tuning_job"
                    ] = job["job_id"]

                    st.success(
                        "Training job created: "
                        f"{job['job_id']}"
                    )
                    st.warning(
                        "Keep FastAPI running without --reload until "
                        "training finishes."
                    )

                except APIClientError as exc:
                    st.error(str(exc))

    # -----------------------------------------------------
    # Training jobs
    # -----------------------------------------------------

    with jobs_tab:
        active_job_id = st.session_state.get(
            "active_fine_tuning_job",
            "",
        )

        if st.button(
            "Refresh training jobs",
            use_container_width=True,
        ):
            st.rerun()

        try:
            jobs_result = api.fine_tuning_jobs()
            jobs = jobs_result.get("jobs", [])
        except APIClientError as exc:
            st.error(str(exc))
            jobs = []

        if active_job_id:
            try:
                active_job = api.fine_tuning_job(active_job_id)

                st.markdown("### Active job")
                st.write(
                    "**Adapter:**",
                    active_job.get("adapter_name", ""),
                )
                st.write(
                    "**Status:**",
                    active_job.get("status", ""),
                )
                st.write(
                    "**Stage:**",
                    active_job.get("stage", ""),
                )

                progress_value = float(
                    active_job.get("progress", 0.0)
                )
                st.progress(max(0.0, min(1.0, progress_value)))

                metrics = active_job.get("metrics", {})
                if metrics:
                    st.json(metrics)

                if active_job.get("result"):
                    with st.expander("Training result"):
                        st.json(active_job["result"])

                if active_job.get("error"):
                    st.error(active_job["error"])

            except APIClientError as exc:
                st.warning(str(exc))

        if jobs:
            job_rows: list[dict[str, object]] = []

            for item in jobs:
                progress_percent = (
                    float(item.get("progress", 0.0)) * 100.0
                )
                created_at = str(item.get("created_at", ""))

                job_rows.append(
                    {
                        "Job": str(item.get("job_id", ""))[:8],
                        "Adapter": item.get("adapter_name", ""),
                        "Status": item.get("status", ""),
                        "Stage": item.get("stage", ""),
                        "Progress": f"{progress_percent:.1f}%",
                        "Created": created_at[:19],
                    }
                )

            st.dataframe(
                job_rows,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No fine-tuning jobs found.")

    # -----------------------------------------------------
    # Adapter testing
    # -----------------------------------------------------

    with adapters_tab:
        try:
            adapter_result = api.fine_tuning_adapters()
            adapters = adapter_result.get("adapters", [])
        except APIClientError as exc:
            st.error(str(exc))
            adapters = []

        if not adapters:
            st.info("No completed LoRA adapters are available.")
            return

        adapter_names = [
            str(item["adapter_name"])
            for item in adapters
        ]

        selected_adapter = st.selectbox(
            "LoRA adapter",
            adapter_names,
        )

        selected_info = next(
            item
            for item in adapters
            if item["adapter_name"] == selected_adapter
        )

        trainable_parameters = selected_info.get(
            "trainable_parameters",
            0,
        )
        trainable_percentage = float(
            selected_info.get("trainable_percentage", 0.0)
        )
        training_examples = selected_info.get(
            "training_examples",
            0,
        )
        adapter_size_mb = (
            float(selected_info.get("size_bytes", 0)) / 1024**2
        )

        info_columns = st.columns(4)
        info_columns[0].metric(
            "Trainable parameters",
            _format_integer(trainable_parameters),
        )
        info_columns[1].metric(
            "Trainable %",
            f"{trainable_percentage:.4f}%",
        )
        info_columns[2].metric(
            "Training examples",
            _format_integer(training_examples),
        )
        info_columns[3].metric(
            "Adapter size",
            f"{adapter_size_mb:.2f} MB",
        )

        test_prompt = st.text_area(
            "Adapter test prompt",
            value=(
                "Explain why semantic retrieval is important "
                "in a research assistant."
            ),
        )

        if st.button(
            "Generate with adapter",
            type="primary",
            use_container_width=True,
        ):
            if not test_prompt.strip():
                st.warning("Enter an adapter test prompt.")
            else:
                try:
                    with st.spinner(
                        "Loading base model and adapter..."
                    ):
                        generated = api.generate_with_adapter(
                            {
                                "adapter_name": selected_adapter,
                                "prompt": test_prompt,
                                "max_new_tokens": 128,
                                "temperature": 0.1,
                                "top_p": 0.9,
                            }
                        )

                    st.markdown(generated.get("response", ""))

                    latency = float(
                        generated.get(
                            "generation_latency_seconds",
                            0.0,
                        )
                    )
                    device = generated.get("device", "unknown")
                    st.caption(
                        f"Device: {device} · Latency: {latency:.2f}s"
                    )

                except APIClientError as exc:
                    st.error(str(exc))

        with st.expander("Danger zone"):
            confirm_delete = st.checkbox(
                "Confirm adapter deletion"
            )

            if st.button(
                "Delete selected adapter",
                disabled=not confirm_delete,
            ):
                try:
                    api.delete_fine_tuning_adapter(selected_adapter)
                    st.success("Adapter deleted.")
                    st.rerun()
                except APIClientError as exc:
                    st.error(str(exc))