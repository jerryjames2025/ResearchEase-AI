from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MathComputation:
    """
    Verified symbolic result produced by SymPy.
    """

    original_input: str
    input_format: str
    operation: str

    parsed_text: str
    parsed_latex: str

    result_text: str
    result_latex: str

    variable: str = ""
    verification: str = ""

    notes: list[str] = field(
        default_factory=list
    )

    substitutions: dict[str, str] = field(
        default_factory=dict
    )