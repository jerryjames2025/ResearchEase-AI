from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AcademicPaper:
    """
    Normalized academic-paper metadata returned by
    any supported academic provider.
    """

    source: str
    source_id: str
    title: str

    authors: list[str] = field(
        default_factory=list
    )

    year: int | None = None
    abstract: str = ""
    url: str = ""
    doi: str = ""
    venue: str = ""

    citation_count: int | None = None

    publication_date: str = ""
    pdf_url: str = ""

    relevance_score: float | None = None

    @property
    def author_text(self) -> str:
        """
        Return a readable author string.
        """

        if not self.authors:
            return "Authors not available"

        if len(self.authors) <= 4:
            return ", ".join(
                self.authors
            )

        return (
            ", ".join(
                self.authors[:4]
            )
            + ", et al."
        )

    @property
    def identity_key(self) -> str:
        """
        Create a key used to remove duplicate papers.
        """

        if self.doi:
            return (
                f"doi:{self.doi.lower().strip()}"
            )

        normalized_title = "".join(
            character.lower()
            for character in self.title
            if character.isalnum()
        )

        return (
            f"title:{normalized_title}"
        )

    def evidence_text(self) -> str:
        """
        Create the text that will be supplied to the LLM
        as external academic evidence.
        """

        parts = [
            f"Title: {self.title}"
        ]

        if self.authors:
            parts.append(
                f"Authors: {self.author_text}"
            )

        if self.year:
            parts.append(
                f"Year: {self.year}"
            )

        if self.venue:
            parts.append(
                f"Venue: {self.venue}"
            )

        if self.doi:
            parts.append(
                f"DOI: {self.doi}"
            )

        if self.abstract:
            parts.append(
                f"Abstract: {self.abstract}"
            )

        return "\n".join(parts)


@dataclass(frozen=True)
class AcademicSearchResponse:
    """
    Combined academic search results and provider warnings.
    """

    papers: list[AcademicPaper]
    warnings: list[str]