from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class LiteraturePaperSummary:
    """
    Normalized summary used in the literature-review matrix.
    """

    citation_key: str
    title: str
    authors: str
    year: str
    source_type: str
    evidence_scope: str

    research_problem: str
    objective: str
    methodology: str
    dataset: str
    models_or_methods: str
    evaluation_metrics: str
    main_findings: str
    limitations: str
    future_work: str
    keywords: str

    source_url: str = ""
    doi: str = ""

    notes: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        """
        Convert the dataclass into session-safe data.
        """

        return asdict(self)

    def matrix_row(self) -> dict[str, str]:
        """
        Return the columns used in the literature matrix.
        """

        return {
            "ID": self.citation_key,
            "Title": self.title,
            "Authors": self.authors,
            "Year": self.year,
            "Evidence": self.evidence_scope,
            "Research problem": self.research_problem,
            "Objective": self.objective,
            "Methodology": self.methodology,
            "Dataset": self.dataset,
            "Models / methods": self.models_or_methods,
            "Metrics": self.evaluation_metrics,
            "Main findings": self.main_findings,
            "Limitations": self.limitations,
            "Future work": self.future_work,
            "Keywords": self.keywords,
        }