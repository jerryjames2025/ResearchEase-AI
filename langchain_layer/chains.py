from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser,
)

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

from langchain_layer.model_factory import (
    get_ollama_chat_model,
)

from langchain_layer.prompts import (
    expanded_research_prompt,
    literature_chunk_prompt,
    literature_record_prompt,
    literature_synthesis_prompt,
    math_computation_prompt,
    math_topic_prompt,
    paper_qa_prompt,
)

from langchain_layer.schemas import (
    LiteratureRecord,
)

from services.embedding_model import (
    EmbeddingService,
)

from services.vector_store import (
    FaissVectorStore,
    SearchResult,
)


NOT_FOUND_MESSAGE = (
    "This information was not found "
    "in the uploaded paper."
)


def _paper_context(
    sources: list[SearchResult],
    max_characters: int,
) -> str:
    """
    Format retrieved paper chunks for a prompt.
    """

    blocks: list[str] = []
    used_characters = 0

    for source in sources:
        block = (
            f"[Page {source.chunk.page_number} | "
            f"Similarity {source.score:.3f}]\n"
            f"{source.chunk.text.strip()}"
        )

        remaining = (
            max_characters
            - used_characters
        )

        if remaining <= 0:
            break

        if len(block) > remaining:
            block = block[
                :remaining
            ]

        blocks.append(
            block
        )

        used_characters += len(
            block
        )

    return "\n\n".join(
        blocks
    )


def build_paper_rag_chain(
    *,
    vector_store: FaissVectorStore,
    embedding_service: EmbeddingService,
    model_name: str,
    top_k: int,
    minimum_score: float,
    max_context_characters: int,
):
    """
    Build an LCEL retrieval and generation chain.

    RunnableLambda performs retrieval.

    RunnableBranch prevents the LLM from being called
    when no paper chunk passes the similarity threshold.
    """

    def retrieve(
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        question = str(
            payload.get(
                "question",
                "",
            )
        ).strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        query_embedding = (
            embedding_service
            .encode_query(
                question
            )
        )

        retrieved = vector_store.search(
            query_embedding=(
                query_embedding
            ),
            top_k=top_k,
        )

        accepted = [
            item
            for item in retrieved
            if (
                item.score
                >= minimum_score
            )
        ]

        display_sources = (
            accepted
            if accepted
            else retrieved
        )

        return {
            **payload,

            "context": _paper_context(
                accepted,
                max_context_characters,
            ),

            "sources": (
                display_sources
            ),

            "retrieval_blocked": (
                not accepted
            ),
        }

    answer_chain = (
        paper_qa_prompt()
        | get_ollama_chat_model(
            model_name=model_name,
            temperature=0.05,
        )
        | StrOutputParser()
    )

    blocked_chain = RunnableLambda(
        lambda payload: {
            **payload,
            "answer": (
                NOT_FOUND_MESSAGE
            ),
        }
    )

    generated_chain = (
        RunnablePassthrough.assign(
            answer=answer_chain
        )
    )

    return (
        RunnableLambda(
            retrieve
        )
        | RunnableBranch(
            (
                lambda payload: bool(
                    payload[
                        "retrieval_blocked"
                    ]
                ),
                blocked_chain,
            ),

            generated_chain,
        )
    )


def run_expanded_research_chain(
    *,
    question: str,
    paper_context: str,
    external_context: str,
    explanation_level: str,
    model_name: str,
) -> str:
    """
    Run the expanded academic research chain.
    """

    chain = (
        expanded_research_prompt()
        | get_ollama_chat_model(
            model_name=model_name,
            temperature=0.05,
        )
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "question": question,

            "paper_context": (
                paper_context
            ),

            "external_context": (
                external_context
            ),

            "explanation_level": (
                explanation_level
            ),
        }
    )


def run_math_computation_chain(
    *,
    model_name: str,
    explanation_level: str,
    operation: str,
    original_input: str,
    input_format: str,
    parsed_text: str,
    result_text: str,
    variable: str,
    verification: str,
    notes: str,
    user_context: str,
    paper_context: str,
) -> str:
    """
    Explain a verified SymPy computation.
    """

    chain = (
        math_computation_prompt()
        | get_ollama_chat_model(
            model_name=model_name,
            temperature=0.1,
        )
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "explanation_level": (
                explanation_level
            ),

            "operation": operation,

            "original_input": (
                original_input
            ),

            "input_format": (
                input_format
            ),

            "parsed_text": (
                parsed_text
            ),

            "result_text": (
                result_text
            ),

            "variable": (
                variable
                or "Not required"
            ),

            "verification": (
                verification
                or (
                    "No separate verification "
                    "was required."
                )
            ),

            "notes": (
                notes
                or "None"
            ),

            "user_context": (
                user_context
                or "None"
            ),

            "paper_context": (
                paper_context
                or (
                    "No uploaded-paper "
                    "context was supplied."
                )
            ),
        }
    )


def run_math_topic_chain(
    *,
    topic: str,
    model_name: str,
    explanation_level: str,
    user_context: str,
    paper_context: str,
) -> str:
    """
    Explain a mathematical topic.
    """

    chain = (
        math_topic_prompt()
        | get_ollama_chat_model(
            model_name=model_name,
            temperature=0.15,
        )
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "topic": topic,

            "explanation_level": (
                explanation_level
            ),

            "user_context": (
                user_context
                or "None"
            ),

            "paper_context": (
                paper_context
                or (
                    "No uploaded-paper "
                    "context was supplied."
                )
            ),
        }
    )


def run_literature_chunk_chain(
    *,
    text: str,
    chunk_number: int,
    total_chunks: int,
    model_name: str,
) -> str:
    """
    Extract evidence notes from one paper section.
    """

    chain = (
        literature_chunk_prompt()
        | get_ollama_chat_model(
            model_name=model_name,
            temperature=0.05,
        )
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "text": text,

            "chunk_number": (
                chunk_number
            ),

            "total_chunks": (
                total_chunks
            ),
        }
    )


def run_literature_record_chain(
    *,
    source_name: str,
    evidence_scope: str,
    evidence: str,
    model_name: str,
) -> LiteratureRecord:
    """
    Create a Pydantic-validated literature
    matrix record.
    """

    parser = PydanticOutputParser(
        pydantic_object=(
            LiteratureRecord
        )
    )

    chain = (
        literature_record_prompt(
            parser.get_format_instructions()
        )
        | get_ollama_chat_model(
            model_name=model_name,
            temperature=0.0,
            json_mode=True,
        )
        | parser
    )

    return chain.invoke(
        {
            "source_name": (
                source_name
            ),

            "evidence_scope": (
                evidence_scope
            ),

            "evidence": evidence,
        }
    )


def run_literature_synthesis_chain(
    *,
    topic: str,
    explanation_level: str,
    paper_records: str,
    model_name: str,
) -> str:
    """
    Generate a citation-grounded literature synthesis.
    """

    chain = (
        literature_synthesis_prompt()
        | get_ollama_chat_model(
            model_name=model_name,
            temperature=0.1,
        )
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "topic": (
                topic
                or (
                    "Not explicitly "
                    "supplied"
                )
            ),

            "explanation_level": (
                explanation_level
            ),

            "paper_records": (
                paper_records
            ),
        }
    )