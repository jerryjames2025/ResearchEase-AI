from __future__ import annotations

import re

import sympy as sp

from sympy.parsing.latex import parse_latex

from sympy.parsing.sympy_parser import (
    convert_xor,
    function_exponentiation,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


class MathInputError(ValueError):
    """
    Raised when mathematical input cannot be parsed.
    """


TRANSFORMATIONS = (
    standard_transformations
    + (
        implicit_multiplication_application,
        function_exponentiation,
        convert_xor,
    )
)


ALLOWED_FUNCTIONS: dict[str, object] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "factorial": sp.factorial,
    "gamma": sp.gamma,
    "Min": sp.Min,
    "Max": sp.Max,
    "pi": sp.pi,
    "E": sp.E,
    "I": sp.I,
    "oo": sp.oo,
}


GLOBAL_DICTIONARY: dict[str, object] = {
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Add": sp.Add,
    "Mul": sp.Mul,
    "Pow": sp.Pow,
    "Eq": sp.Eq,
    "Tuple": sp.Tuple,
}


BLOCKED_WORDS = {
    "import",
    "exec",
    "eval",
    "open",
    "compile",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "lambda",
    "class",
    "def",
    "yield",
}


def _strip_math_delimiters(
    text: str,
) -> str:
    """
    Remove common LaTeX display delimiters.
    """

    text = text.strip()

    delimiter_pairs = (
        ("$$", "$$"),
        ("\\[", "\\]"),
        ("\\(", "\\)"),
        ("$", "$"),
    )

    for start, end in delimiter_pairs:
        if (
            text.startswith(start)
            and text.endswith(end)
        ):
            return text[
                len(start):
                -len(end)
            ].strip()

    return text


def _looks_like_latex(
    text: str,
) -> bool:
    """
    Detect common LaTeX commands.
    """

    indicators = (
        "\\frac",
        "\\sqrt",
        "\\sum",
        "\\int",
        "\\lim",
        "\\sin",
        "\\cos",
        "\\tan",
        "\\left",
        "\\right",
        "^{",
        "_{",
    )

    return any(
        indicator in text
        for indicator in indicators
    )


def _validate_plain_expression(
    text: str,
) -> None:
    """
    Reject obvious programming and code-like input before
    using SymPy's parser.
    """

    lowered = text.lower()

    if "__" in text:
        raise MathInputError(
            "Double-underscore names are not allowed."
        )

    for word in BLOCKED_WORDS:
        if re.search(
            rf"\b{word}\b",
            lowered,
        ):
            raise MathInputError(
                "The input contains a blocked "
                "programming keyword."
            )

    if any(
        character in text
        for character in "[]{};:'\""
    ):
        raise MathInputError(
            "Lists, dictionaries, quotes, semicolons, "
            "and code-like syntax are not allowed."
        )

    if re.search(
        r"[A-Za-z_]\w*\s*\.",
        text,
    ):
        raise MathInputError(
            "Attribute access is not allowed."
        )

    if not re.fullmatch(
        r"[A-Za-z0-9_+\-*/^().,=<>!\s]+",
        text,
    ):
        raise MathInputError(
            "The expression contains unsupported characters. "
            "Use standard mathematical syntax."
        )


def _make_local_dictionary(
    text: str,
) -> dict[str, object]:
    """
    Create symbols for variable names while preserving
    supported mathematical functions.
    """

    local_dictionary = dict(
        ALLOWED_FUNCTIONS
    )

    identifiers = set(
        re.findall(
            r"\b[A-Za-z_]\w*\b",
            text,
        )
    )

    for name in identifiers:
        if name not in local_dictionary:
            local_dictionary[name] = (
                sp.Symbol(name)
            )

    return local_dictionary


def _parse_plain_side(
    text: str,
) -> sp.Basic:
    """
    Parse one side of a normal mathematical expression.
    """

    _validate_plain_expression(
        text
    )

    try:
        return parse_expr(
            text,
            local_dict=(
                _make_local_dictionary(
                    text
                )
            ),
            global_dict=(
                GLOBAL_DICTIONARY
            ),
            transformations=(
                TRANSFORMATIONS
            ),
            evaluate=True,
        )

    except Exception as exc:
        raise MathInputError(
            f"Unable to parse the expression: {exc}"
        ) from exc


def parse_plain_math(
    text: str,
) -> sp.Basic:
    """
    Parse SymPy-style or normal plain-text mathematics.
    """

    text = _strip_math_delimiters(
        text
    )

    if not text:
        raise MathInputError(
            "Enter an equation or mathematical expression."
        )

    contains_single_equals = (
        text.count("=") == 1
        and not any(
            operator in text
            for operator in (
                "<=",
                ">=",
                "!=",
                "==",
            )
        )
    )

    if contains_single_equals:
        left_text, right_text = (
            text.split(
                "=",
                1,
            )
        )

        return sp.Eq(
            _parse_plain_side(
                left_text
            ),
            _parse_plain_side(
                right_text
            ),
        )

    return _parse_plain_side(
        text
    )


def parse_latex_math(
    text: str,
) -> sp.Basic:
    """
    Parse LaTeX using SymPy's Lark backend.
    """

    text = _strip_math_delimiters(
        text
    )

    if not text:
        raise MathInputError(
            "Enter a LaTeX equation or expression."
        )

    try:
        parsed = parse_latex(
            text,
            backend="lark",
        )

    except Exception as exc:
        raise MathInputError(
            "Unable to parse the LaTeX input. "
            "Check braces, commands, and equation syntax. "
            f"Technical detail: {exc}"
        ) from exc

    if parsed is None:
        raise MathInputError(
            "The LaTeX parser returned no "
            "mathematical expression."
        )

    return parsed


def parse_math_input(
    text: str,
    input_format: str = "Auto detect",
) -> tuple[sp.Basic, str]:
    """
    Parse either plain text or LaTeX.
    """

    selected_format = (
        input_format
        .strip()
        .lower()
    )

    if selected_format == "latex":
        return (
            parse_latex_math(
                text
            ),
            "LaTeX",
        )

    if selected_format in {
        "sympy / plain text",
        "plain text",
        "sympy",
    }:
        return (
            parse_plain_math(
                text
            ),
            "SymPy / plain text",
        )

    if _looks_like_latex(
        text
    ):
        try:
            return (
                parse_latex_math(
                    text
                ),
                "LaTeX",
            )

        except MathInputError:
            pass

    return (
        parse_plain_math(
            text
        ),
        "SymPy / plain text",
    )


def parse_scalar(
    text: str,
) -> sp.Basic:
    """
    Parse a limit point, integration bound,
    or substitution value.
    """

    value = text.strip()

    if not value:
        raise MathInputError(
            "A required mathematical value is empty."
        )

    return parse_plain_math(
        value
    )