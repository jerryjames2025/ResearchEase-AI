from __future__ import annotations

from collections.abc import Iterable

import sympy as sp

from mathematics.models import (
    MathComputation,
)

from mathematics.parser import (
    MathInputError,
    parse_math_input,
    parse_scalar,
)


SUPPORTED_OPERATIONS = (
    "Explain and simplify",
    "Solve equation",
    "Differentiate",
    "Integrate",
    "Factor",
    "Expand",
    "Evaluate substitutions",
    "Calculate limit",
)


def _sorted_symbols(
    expression: sp.Basic,
) -> list[sp.Symbol]:
    """
    Return symbols in a stable order.
    """

    return sorted(
        expression.free_symbols,
        key=lambda symbol: symbol.name,
    )


def _resolve_variable(
    expression: sp.Basic,
    variable_name: str,
) -> sp.Symbol:
    """
    Select the requested variable or automatically
    use the first variable found.
    """

    requested_name = (
        variable_name.strip()
    )

    if requested_name:
        if not requested_name.replace(
            "_",
            "",
        ).isalnum():
            raise MathInputError(
                "Variable names may contain letters, "
                "numbers, and underscores."
            )

        return sp.Symbol(
            requested_name
        )

    symbols = _sorted_symbols(
        expression
    )

    if not symbols:
        raise MathInputError(
            "This operation requires a variable. "
            "Enter one in the variable field."
        )

    return symbols[0]


def _apply_to_equation(
    expression: sp.Basic,
    function,
) -> sp.Basic:
    """
    Apply an operation to both sides of an equation.
    """

    if isinstance(
        expression,
        sp.Equality,
    ):
        return sp.Eq(
            function(
                expression.lhs
            ),
            function(
                expression.rhs
            ),
        )

    return function(
        expression
    )


def _parse_substitutions(
    raw_text: str,
) -> tuple[
    dict[sp.Symbol, sp.Basic],
    dict[str, str],
]:
    """
    Parse substitutions such as x=2, y=3.
    """

    symbolic: dict[
        sp.Symbol,
        sp.Basic,
    ] = {}

    display: dict[
        str,
        str,
    ] = {}

    if not raw_text.strip():
        raise MathInputError(
            "Enter substitutions such as x=2, y=3."
        )

    for item in raw_text.split(
        ","
    ):
        if "=" not in item:
            raise MathInputError(
                f"Invalid substitution '{item.strip()}'. "
                "Use name=value."
            )

        name, value_text = (
            item.split(
                "=",
                1,
            )
        )

        name = name.strip()
        value_text = (
            value_text.strip()
        )

        if (
            not name
            or not name.replace(
                "_",
                "",
            ).isalnum()
        ):
            raise MathInputError(
                f"Invalid substitution variable '{name}'."
            )

        value = parse_scalar(
            value_text
        )

        symbolic[
            sp.Symbol(name)
        ] = value

        display[name] = str(
            value
        )

    return (
        symbolic,
        display,
    )


def _format_solution_result(
    solutions,
) -> sp.Basic:
    """
    Convert solve output into a printable SymPy object.
    """

    if isinstance(
        solutions,
        dict,
    ):
        return sp.Dict(
            solutions
        )

    if isinstance(
        solutions,
        (
            list,
            tuple,
            set,
        ),
    ):
        return sp.FiniteSet(
            *solutions
        )

    return solutions


def _verify_solutions(
    original: sp.Basic,
    variable: sp.Symbol,
    solutions,
) -> str:
    """
    Substitute symbolic solutions back into the equation.
    """

    if isinstance(
        solutions,
        dict,
    ):
        candidates: Iterable = [
            solutions
        ]

    elif isinstance(
        solutions,
        (
            list,
            tuple,
            set,
            sp.FiniteSet,
        ),
    ):
        candidates = solutions

    else:
        candidates = [
            solutions
        ]

    statuses: list[str] = []

    for candidate in candidates:
        try:
            if isinstance(
                candidate,
                dict,
            ):
                substituted = (
                    original.subs(
                        candidate
                    )
                )

            else:
                substituted = (
                    original.subs(
                        variable,
                        candidate,
                    )
                )

            if isinstance(
                substituted,
                sp.Equality,
            ):
                residual = sp.simplify(
                    substituted.lhs
                    - substituted.rhs
                )

            else:
                residual = sp.simplify(
                    substituted
                )

            if (
                residual == 0
                or residual is sp.S.true
            ):
                status = "verified"

            else:
                status = (
                    "not conclusively verified"
                )

            statuses.append(
                f"{candidate}: {status}"
            )

        except Exception:
            statuses.append(
                f"{candidate}: verification unavailable"
            )

    return "; ".join(
        statuses
    )


def compute_math(
    source_text: str,
    input_format: str,
    operation: str,
    variable_name: str = "",
    derivative_order: int = 1,
    lower_bound: str = "",
    upper_bound: str = "",
    limit_point: str = "0",
    limit_direction: str = "+-",
    substitutions_text: str = "",
) -> MathComputation:
    """
    Parse input and perform a verified symbolic operation.
    """

    expression, detected_format = (
        parse_math_input(
            source_text,
            input_format,
        )
    )

    notes: list[str] = []
    verification = ""
    substitutions_display: dict[
        str,
        str,
    ] = {}

    variable = ""

    if operation == "Explain and simplify":
        result = _apply_to_equation(
            expression,
            sp.simplify,
        )

        notes.append(
            "SymPy simplify was used as a "
            "general-purpose simplification step."
        )

    elif operation == "Solve equation":
        symbol = _resolve_variable(
            expression,
            variable_name,
        )

        variable = symbol.name

        if isinstance(
            expression,
            sp.Equality,
        ):
            target = expression

        else:
            target = sp.Eq(
                expression,
                0,
            )

        solutions = sp.solve(
            target,
            symbol,
        )

        result = _format_solution_result(
            solutions
        )

        verification = _verify_solutions(
            target,
            symbol,
            solutions,
        )

        if not solutions:
            notes.append(
                "SymPy did not find a symbolic solution "
                "for the selected variable."
            )

    elif operation == "Differentiate":
        symbol = _resolve_variable(
            expression,
            variable_name,
        )

        variable = symbol.name

        order = max(
            1,
            int(
                derivative_order
            ),
        )

        result = _apply_to_equation(
            expression,
            lambda item: sp.diff(
                item,
                symbol,
                order,
            ),
        )

        notes.append(
            f"Derivative order: {order}."
        )

    elif operation == "Integrate":
        symbol = _resolve_variable(
            expression,
            variable_name,
        )

        variable = symbol.name

        if (
            lower_bound.strip()
            or upper_bound.strip()
        ):
            if (
                not lower_bound.strip()
                or not upper_bound.strip()
            ):
                raise MathInputError(
                    "Enter both lower and upper bounds "
                    "for a definite integral."
                )

            lower = parse_scalar(
                lower_bound
            )

            upper = parse_scalar(
                upper_bound
            )

            result = _apply_to_equation(
                expression,
                lambda item: sp.integrate(
                    item,
                    (
                        symbol,
                        lower,
                        upper,
                    ),
                ),
            )

            notes.append(
                f"Definite integral from {lower} to {upper}."
            )

        else:
            result = _apply_to_equation(
                expression,
                lambda item: sp.integrate(
                    item,
                    symbol,
                ),
            )

            notes.append(
                "Indefinite integral; an arbitrary "
                "constant is not added automatically."
            )

    elif operation == "Factor":
        result = _apply_to_equation(
            expression,
            sp.factor,
        )

    elif operation == "Expand":
        result = _apply_to_equation(
            expression,
            sp.expand,
        )

    elif operation == "Evaluate substitutions":
        (
            symbolic_substitutions,
            substitutions_display,
        ) = _parse_substitutions(
            substitutions_text
        )

        substituted = expression.subs(
            symbolic_substitutions
        )

        if substituted.free_symbols:
            result = sp.simplify(
                substituted
            )

        else:
            result = sp.N(
                substituted
            )

        verification = (
            "Values were substituted directly into "
            "the parsed symbolic expression."
        )

    elif operation == "Calculate limit":
        symbol = _resolve_variable(
            expression,
            variable_name,
        )

        variable = symbol.name

        point = parse_scalar(
            limit_point
        )

        if limit_direction in {
            "+",
            "-",
            "+-",
        }:
            direction = limit_direction

        else:
            direction = "+-"

        if isinstance(
            expression,
            sp.Equality,
        ):
            result = sp.Eq(
                sp.limit(
                    expression.lhs,
                    symbol,
                    point,
                    dir=direction,
                ),
                sp.limit(
                    expression.rhs,
                    symbol,
                    point,
                    dir=direction,
                ),
            )

        else:
            result = sp.limit(
                expression,
                symbol,
                point,
                dir=direction,
            )

        notes.append(
            f"Limit as {symbol} approaches "
            f"{point}, direction {direction}."
        )

    else:
        raise MathInputError(
            f"Unsupported operation: {operation}"
        )

    return MathComputation(
        original_input=source_text,
        input_format=detected_format,
        operation=operation,
        parsed_text=str(
            expression
        ),
        parsed_latex=sp.latex(
            expression
        ),
        result_text=str(
            result
        ),
        result_latex=sp.latex(
            result
        ),
        variable=variable,
        verification=verification,
        notes=notes,
        substitutions=(
            substitutions_display
        ),
    )