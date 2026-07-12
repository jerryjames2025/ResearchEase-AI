from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
    status,
)

from backend.core.errors import APIError
from backend.core.files import (
    read_pdf_upload,
)
from backend.core.session_store import (
    session_store,
)
from backend.core.settings import (
    APISettings,
    get_settings,
)
from backend.llm.dependency import (
    LLMRequestConfig,
    get_llm_request_config,
)
from backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    IndexRequest,
    IndexResponse,
    MessageResponse,
    PaperUploadResponse,
    SessionListResponse,
    SessionResponse,
)
from services.embedding_model import (
    load_embedding_service,
)
from services.ollama_client import (
    analyze_paper,
)
from services.pdf_parser import (
    create_page_chunks,
    extract_pdf,
)
from services.vector_store import (
    FaissVectorStore,
)


router = APIRouter(
    prefix="/papers",
    tags=["papers"],
)


def _session_response(
    session,
) -> SessionResponse:
    return SessionResponse(
        session_id=session.session_id,
        filename=session.paper.filename,
        page_count=(
            session.paper.page_count
        ),
        extracted_characters=len(
            session.paper.text
        ),
        created_at=session.created_at,
        updated_at=session.updated_at,
        analysis_ready=bool(
            session.analysis
        ),
        index_ready=(
            session.index_ready
        ),
        chunk_count=len(
            session.paper_chunks
        ),
        chat_message_count=len(
            session.chat_history
        ),
    )


@router.get(
    "",
    response_model=(
        SessionListResponse
    ),
)
def list_sessions(
    limit: int = 100,
) -> SessionListResponse:
    summaries = session_store.list(
        limit=limit
    )

    sessions = [
        SessionResponse(
            session_id=item[
                "session_id"
            ],
            filename=item["filename"],
            page_count=item[
                "page_count"
            ],
            extracted_characters=item[
                "extracted_characters"
            ],
            created_at=item[
                "created_at"
            ],
            updated_at=item[
                "updated_at"
            ],
            analysis_ready=item[
                "analysis_ready"
            ],
            index_ready=item[
                "index_ready"
            ],
            chunk_count=item[
                "chunk_count"
            ],
            chat_message_count=len(
                session_store
                .get_chat_messages(
                    item["session_id"]
                )
            ),
        )
        for item in summaries
    ]

    return SessionListResponse(
        sessions=sessions
    )


@router.post(
    "/upload",
    response_model=(
        PaperUploadResponse
    ),
    status_code=(
        status.HTTP_201_CREATED
    ),
)
def upload_paper(
    file: UploadFile = File(...),
    settings: APISettings = Depends(
        get_settings
    ),
) -> PaperUploadResponse:
    upload = read_pdf_upload(
        file,
        settings.max_upload_mb,
    )

    paper = extract_pdf(
        upload
    )

    session = session_store.create(
        paper,
        raw_pdf=upload.data,
    )

    return PaperUploadResponse(
        session_id=(
            session.session_id
        ),
        filename=paper.filename,
        page_count=paper.page_count,
        extracted_characters=len(
            paper.text
        ),
        created_at=(
            session.created_at
        ),
    )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
)
def get_session(
    session_id: str,
) -> SessionResponse:
    return _session_response(
        session_store.get(
            session_id
        )
    )


@router.get(
    "/{session_id}/analysis",
    response_model=AnalyzeResponse,
)
def get_analysis(
    session_id: str,
) -> AnalyzeResponse:
    session = session_store.get(
        session_id
    )

    if not session.analysis:
        raise APIError(
            status_code=404,
            code="analysis_not_found",
            detail=(
                "No saved analysis exists "
                "for this session."
            ),
        )

    return AnalyzeResponse(
        session_id=(
            session.session_id
        ),
        filename=(
            session.paper.filename
        ),
        analysis=session.analysis,
        sections_processed=len(
            session.partial_summaries
        ),
    )


@router.delete(
    "/{session_id}",
    response_model=MessageResponse,
)
def delete_session(
    session_id: str,
) -> MessageResponse:
    session_store.delete(
        session_id
    )

    return MessageResponse(
        message=(
            "Research session and "
            "persisted data deleted."
        )
    )


@router.post(
    "/{session_id}/analyze",
    response_model=AnalyzeResponse,
)
def analyze_uploaded_paper(
    session_id: str,
    request: AnalyzeRequest,
    llm: LLMRequestConfig = Depends(
        get_llm_request_config
    ),
) -> AnalyzeResponse:
    session = session_store.get(
        session_id
    )

    runtime_model = llm.model_spec(
        fallback_model=(
            request.ollama_model
        )
    )

    (
        analysis,
        partial_summaries,
    ) = analyze_paper(
        text=session.paper.text,
        filename=(
            session.paper.filename
        ),
        page_count=(
            session.paper.page_count
        ),
        model=runtime_model,
        explanation_level=(
            request.explanation_level
        ),
    )

    session_store.save_analysis(
        session,
        analysis,
        partial_summaries,
    )

    return AnalyzeResponse(
        session_id=(
            session.session_id
        ),
        filename=(
            session.paper.filename
        ),
        analysis=analysis,
        sections_processed=len(
            partial_summaries
        ),
    )


@router.post(
    "/{session_id}/index",
    response_model=IndexResponse,
)
def build_paper_index(
    session_id: str,
    request: IndexRequest,
) -> IndexResponse:
    session = session_store.get(
        session_id
    )

    chunks = create_page_chunks(
        session.paper
    )

    embedding_service = (
        load_embedding_service(
            request.embedding_model,
            request.device,
        )
    )

    embeddings = (
        embedding_service
        .encode_documents(
            [
                chunk.text
                for chunk in chunks
            ]
        )
    )

    vector_store = FaissVectorStore(
        chunks=chunks,
        embeddings=embeddings,
    )

    index_path = (
        session_store.save_index(
            session,
            chunks,
            vector_store,
            request.embedding_model,
            request.device,
        )
    )

    return IndexResponse(
        session_id=(
            session.session_id
        ),
        chunk_count=(
            vector_store.size
        ),
        embedding_model=(
            request.embedding_model
        ),
        device=str(
            embedding_service.device
        ),
        index_path=index_path,
    )