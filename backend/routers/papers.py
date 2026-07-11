from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
    status,
)

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
from backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    IndexRequest,
    IndexResponse,
    MessageResponse,
    PaperUploadResponse,
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


@router.post(
    "/upload",
    response_model=PaperUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_paper(
    file: UploadFile = File(...),
    settings: APISettings = Depends(
        get_settings
    ),
) -> PaperUploadResponse:
    """
    Upload a research-paper PDF and create
    a new API session.
    """

    upload = read_pdf_upload(
        file,
        settings.max_upload_mb,
    )

    paper = extract_pdf(
        upload
    )

    session = session_store.create(
        paper
    )

    return PaperUploadResponse(
        session_id=session.session_id,
        filename=paper.filename,
        page_count=paper.page_count,
        extracted_characters=len(
            paper.text
        ),
        created_at=session.created_at,
    )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
)
def get_session(
    session_id: str,
) -> SessionResponse:
    session = session_store.get(
        session_id
    )

    with session.lock:
        return SessionResponse(
            session_id=(
                session.session_id
            ),
            filename=(
                session.paper.filename
            ),
            page_count=(
                session.paper.page_count
            ),
            created_at=(
                session.created_at
            ),
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
            "Research session deleted."
        )
    )


@router.post(
    "/{session_id}/analyze",
    response_model=AnalyzeResponse,
)
def analyze_uploaded_paper(
    session_id: str,
    request: AnalyzeRequest,
) -> AnalyzeResponse:
    session = session_store.get(
        session_id
    )

    analysis, partial_summaries = (
        analyze_paper(
            text=session.paper.text,
            filename=(
                session.paper.filename
            ),
            page_count=(
                session.paper.page_count
            ),
            model=(
                request.ollama_model
            ),
            explanation_level=(
                request.explanation_level
            ),
        )
    )

    with session.lock:
        session.analysis = analysis
        session.partial_summaries = (
            partial_summaries
        )

    return AnalyzeResponse(
        session_id=session.session_id,
        filename=session.paper.filename,
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

    with session.lock:
        session.paper_chunks = chunks
        session.vector_store = (
            vector_store
        )
        session.embedding_model_name = (
            request.embedding_model
        )
        session.embedding_device = (
            request.device
        )
        session.chat_history = []

    return IndexResponse(
        session_id=session.session_id,
        chunk_count=vector_store.size,
        embedding_model=(
            request.embedding_model
        ),
        device=(
            str(
                embedding_service.device
            )
        ),
    )