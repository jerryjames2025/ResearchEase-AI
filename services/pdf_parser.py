from __future__ import annotations

from dataclasses import dataclass
import re

import pymupdf

from config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE


@dataclass(frozen=True)
class PageText:
    """
    Stores the extracted text of one PDF page.
    """

    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedPaper:
    """
    Stores the complete extracted research paper.
    """

    filename: str
    page_count: int
    text: str
    pages: list[PageText]


@dataclass(frozen=True)
class PaperChunk:
    """
    One searchable chunk with its source-page information.
    """

    chunk_id: int
    page_number: int
    chunk_number_on_page: int
    text: str

    @property
    def source_label(self) -> str:
        return (
            f"Page {self.page_number}, "
            f"chunk {self.chunk_number_on_page}"
        )


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving useful paragraphs.
    """

    text = text.replace("\x00", " ")

    # Replace repeated spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces appearing immediately after line breaks.
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Prevent too many empty lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf(uploaded_file) -> ExtractedPaper:
    """
    Extract page-aware text from a Streamlit UploadedFile.
    """

    pdf_bytes = uploaded_file.getvalue()

    try:
        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        )
    except Exception as exc:
        raise ValueError(
            f"Unable to open the PDF: {exc}"
        ) from exc

    pages: list[PageText] = []
    combined_pages: list[str] = []

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            page_text = clean_text(
                page.get_text(
                    "text",
                    sort=True,
                )
            )

            pages.append(
                PageText(
                    page_number=page_number,
                    text=page_text,
                )
            )

            if page_text:
                combined_pages.append(
                    f"\n--- PAGE {page_number} ---\n"
                    f"{page_text}"
                )
            else:
                combined_pages.append(
                    f"\n--- PAGE {page_number} ---\n"
                    "[No selectable text was found on this page.]"
                )

    finally:
        document.close()

    if not any(page.text for page in pages):
        raise ValueError(
            "No selectable text was found. "
            "This may be a scanned PDF. "
            "OCR support is not included in Version 2."
        )

    return ExtractedPaper(
        filename=uploaded_file.name,
        page_count=len(pages),
        text="\n".join(combined_pages).strip(),
        pages=pages,
    )


def split_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split text into overlapping chunks.

    The function tries to end chunks near paragraph or sentence
    boundaries instead of cutting the text at an arbitrary character.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must satisfy 0 <= overlap < chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        hard_end = min(
            start + chunk_size,
            len(text),
        )

        end = hard_end

        if hard_end < len(text):
            paragraph_break = text.rfind(
                "\n\n",
                start,
                hard_end,
            )

            sentence_break = text.rfind(
                ". ",
                start,
                hard_end,
            )

            preferred_break = max(
                paragraph_break,
                sentence_break,
            )

            # Avoid creating extremely small chunks.
            if preferred_break > start + (chunk_size // 2):
                if preferred_break == paragraph_break:
                    end = preferred_break + 2
                else:
                    end = preferred_break + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - overlap

        # Protection against an infinite loop.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def create_page_chunks(
    paper: ExtractedPaper,
    chunk_size: int = RAG_CHUNK_SIZE,
    overlap: int = RAG_CHUNK_OVERLAP,
) -> list[PaperChunk]:
    """
    Create searchable paper chunks while retaining page numbers.
    """

    chunks: list[PaperChunk] = []
    global_chunk_id = 0

    for page in paper.pages:
        if not page.text:
            continue

        page_parts = split_text(
            text=page.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for page_chunk_number, part in enumerate(
            page_parts,
            start=1,
        ):
            chunks.append(
                PaperChunk(
                    chunk_id=global_chunk_id,
                    page_number=page.page_number,
                    chunk_number_on_page=page_chunk_number,
                    text=part,
                )
            )

            global_chunk_id += 1

    if not chunks:
        raise ValueError(
            "The PDF did not produce any searchable chunks."
        )

    return chunks