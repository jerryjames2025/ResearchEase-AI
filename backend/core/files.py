from __future__ import annotations

from dataclasses import dataclass

from fastapi import UploadFile

from backend.core.errors import APIError


@dataclass(frozen=True)
class UploadedBytes:
    """
    Adapter matching the interface expected by
    services.pdf_parser.extract_pdf().
    """

    name: str
    data: bytes

    def getvalue(self) -> bytes:
        return self.data


def read_pdf_upload(
    upload: UploadFile,
    max_upload_mb: int,
) -> UploadedBytes:
    """
    Read and validate a FastAPI PDF upload.
    """

    filename = (
        upload.filename
        or "uploaded-paper.pdf"
    )

    if not filename.lower().endswith(
        ".pdf"
    ):
        raise APIError(
            status_code=400,
            code="invalid_file_type",
            detail="Only PDF files are supported.",
        )

    max_bytes = (
        max_upload_mb
        * 1024
        * 1024
    )

    content = upload.file.read(
        max_bytes + 1
    )

    if not content:
        raise APIError(
            status_code=400,
            code="empty_file",
            detail="The uploaded PDF is empty.",
        )

    if len(content) > max_bytes:
        raise APIError(
            status_code=413,
            code="file_too_large",
            detail=(
                "The uploaded file exceeds the "
                f"{max_upload_mb} MB limit."
            ),
        )

    if not content.startswith(
        b"%PDF"
    ):
        raise APIError(
            status_code=400,
            code="invalid_pdf",
            detail=(
                "The uploaded file does not appear "
                "to contain valid PDF data."
            ),
        )

    return UploadedBytes(
        name=filename,
        data=content,
    )