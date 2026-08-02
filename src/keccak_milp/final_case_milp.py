"""
Primitivas MILP del caso final de Keccak reducido.

Este módulo proporciona formulaciones exactas para:

1. La S-box Chi* mediante su forma normal algebraica compacta.
2. La capa adicional Mu mediante un XOR de tres entradas.

Los modelos V1 permanecen separados y no son modificados aquí.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pulp


CHI_STAR_QUADRATIC_PAIRS: tuple[
    tuple[int, int],
    ...,
] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)

CHI_STAR_PARITY_UPPER_BOUNDS: tuple[
    int,
    ...,
] = (
    2,
    3,
    3,
    3,
    3,
)


@dataclass(frozen=True)
class ChiStarMILPVariables:
    """Variables auxiliares creadas para una S-box Chi*."""

    products: dict[
        tuple[int, int],
        pulp.LpVariable,
    ]

    parity: tuple[
        pulp.LpVariable,
        ...,
    ]


@dataclass(frozen=True)
class MuMILPVariables:
    """Variable de paridad creada para un XOR triple."""

    parity: pulp.LpVariable


def _validate_problem(
    problem: object,
) -> pulp.LpProblem:
    """Valida una instancia de problema PuLP."""

    if not isinstance(
        problem,
        pulp.LpProblem,
    ):
        raise TypeError(
            "problem debe ser una instancia de pulp.LpProblem."
        )

    return problem


def _validate_prefix(
    prefix: object,
) -> str:
    """Valida un prefijo no vacío para nombres MILP."""

    if not isinstance(prefix, str):
        raise TypeError(
            "prefix debe ser una cadena de texto."
        )

    normalized = prefix.strip()

    if not normalized:
        raise ValueError(
            "prefix no puede estar vacío."
        )

    return normalized


def _validate_variable_sequence(
    variables: object,
    *,
    expected_length: int,
    parameter_name: str,
) -> tuple[pulp.LpVariable, ...]:
    """Valida una secuencia de variables PuLP."""

    if isinstance(
        variables,
        (str, bytes),
    ):
        raise TypeError(
            f"{parameter_name} debe ser una secuencia "
            "de variables PuLP."
        )

    try:
        normalized = tuple(variables)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(
            f"{parameter_name} debe ser una secuencia "
            "de variables PuLP."
        ) from error

    if len(normalized) != expected_length:
        raise ValueError(
            f"{parameter_name} debe contener exactamente "
            f"{expected_length} variables."
        )

    for variable in normalized:
        if not isinstance(
            variable,
            pulp.LpVariable,
        ):
            raise TypeError(
                f"Todos los elementos de {parameter_name} "
                "deben ser variables PuLP."
            )

    return normalized


def _add_constraint(
    problem: pulp.LpProblem,
    constraint: pulp.LpConstraint,
    name: str,
) -> None:
    """Agrega una restricción con nombre explícito."""

    problem.addConstraint(
        constraint,
        name=name,
    )


def create_binary_variable(
    problem: pulp.LpProblem,
    name: str,
) -> pulp.LpVariable:
    """Crea y adjunta una variable binaria al problema."""

    validated_problem = _validate_problem(
        problem
    )

    validated_name = _validate_prefix(
        name
    )

    return validated_problem.add_variable(
        validated_name,
        lowBound=0,
        upBound=1,
        cat=pulp.LpBinary,
    )


def create_integer_variable(
    problem: pulp.LpProblem,
    name: str,
    *,
    low_bound: int,
    upper_bound: int,
) -> pulp.LpVariable:
    """Crea y adjunta una variable entera acotada."""

    validated_problem = _validate_problem(
        problem
    )

    validated_name = _validate_prefix(
        name
    )

    if (
        isinstance(low_bound, bool)
        or not isinstance(low_bound, int)
    ):
        raise TypeError(
            "low_bound debe ser un entero."
        )

    if (
        isinstance(upper_bound, bool)
        or not isinstance(upper_bound, int)
    ):
        raise TypeError(
            "upper_bound debe ser un entero."
        )

    if low_bound > upper_bound:
        raise ValueError(
            "low_bound no puede ser mayor que upper_bound."
        )

    return validated_problem.add_variable(
        validated_name,
        lowBound=low_bound,
        upBound=upper_bound,
        cat=pulp.LpInteger,
    )


def add_and_equivalence(
    problem: pulp.LpProblem,
    left: pulp.LpVariable,
    right: pulp.LpVariable,
    output: pulp.LpVariable,
    *,
    prefix: str,
) -> None:
    """
    Agrega la linealización exacta:

        output = left AND right.
    """

    validated_problem = _validate_problem(
        problem
    )

    validated_prefix = _validate_prefix(
        prefix
    )

    left_variable, right_variable, output_variable = (
        _validate_variable_sequence(
            (
                left,
                right,
                output,
            ),
            expected_length=3,
            parameter_name="variables",
        )
    )

    _add_constraint(
        validated_problem,
        output_variable <= left_variable,
        f"{validated_prefix}_upper_left",
    )

    _add_constraint(
        validated_problem,
        output_variable <= right_variable,
        f"{validated_prefix}_upper_right",
    )

    _add_constraint(
        validated_problem,
        (
            output_variable
            >= left_variable + right_variable - 1
        ),
        f"{validated_prefix}_lower",
    )


def _chi_star_anf_terms(
    inputs: tuple[pulp.LpVariable, ...],
    products: dict[
        tuple[int, int],
        pulp.LpVariable,
    ],
) -> tuple[
    tuple[pulp.LpVariable, ...],
    ...,
]:
    """Construye los términos ANF de las cinco coordenadas."""

    x0, x1, x2, x3, x4 = inputs

    q01 = products[0, 1]
    q02 = products[0, 2]
    q03 = products[0, 3]
    q04 = products[0, 4]

    q12 = products[1, 2]
    q13 = products[1, 3]
    q14 = products[1, 4]

    q23 = products[2, 3]
    q24 = products[2, 4]
    q34 = products[3, 4]

    return (
        (
            x0,
            q12,
            q24,
            q34,
        ),
        (
            x1,
            q03,
            q13,
            x4,
            q24,
            q34,
        ),
        (
            x0,
            q01,
            q12,
            x3,
            q03,
            q23,
            q34,
        ),
        (
            x1,
            q01,
            x2,
            q02,
            x3,
            q04,
            q24,
        ),
        (
            q01,
            q02,
            q12,
            x3,
            q13,
            q14,
        ),
    )


def add_chi_star_sbox_constraints(
    problem: pulp.LpProblem,
    inputs: Sequence[pulp.LpVariable],
    outputs: Sequence[pulp.LpVariable],
    *,
    prefix: str,
) -> ChiStarMILPVariables:
    """
    Modela exactamente una S-box Chi* mediante su ANF.

    La función recibe cinco variables binarias de entrada y cinco
    variables binarias de salida. Crea:

        10 variables binarias de productos AND;
         5 variables enteras de paridad.

    Se agregan 35 restricciones:

        30 para los diez productos AND;
         5 para las coordenadas ANF.
    """

    validated_problem = _validate_problem(
        problem
    )

    validated_prefix = _validate_prefix(
        prefix
    )

    input_variables = _validate_variable_sequence(
        inputs,
        expected_length=5,
        parameter_name="inputs",
    )

    output_variables = _validate_variable_sequence(
        outputs,
        expected_length=5,
        parameter_name="outputs",
    )

    products = {
        pair: create_binary_variable(
            validated_problem,
            (
                f"{validated_prefix}"
                f"_product_{pair[0]}{pair[1]}"
            ),
        )
        for pair in CHI_STAR_QUADRATIC_PAIRS
    }

    parity = tuple(
        create_integer_variable(
            validated_problem,
            (
                f"{validated_prefix}"
                f"_parity_{index}"
            ),
            low_bound=0,
            upper_bound=(
                CHI_STAR_PARITY_UPPER_BOUNDS[
                    index
                ]
            ),
        )
        for index in range(5)
    )

    for first, second in CHI_STAR_QUADRATIC_PAIRS:
        add_and_equivalence(
            validated_problem,
            input_variables[first],
            input_variables[second],
            products[first, second],
            prefix=(
                f"{validated_prefix}"
                f"_and_{first}{second}"
            ),
        )

    coordinate_terms = _chi_star_anf_terms(
        input_variables,
        products,
    )

    for output_index, terms in enumerate(
        coordinate_terms
    ):
        _add_constraint(
            validated_problem,
            (
                pulp.lpSum(terms)
                == (
                    output_variables[output_index]
                    + 2 * parity[output_index]
                )
            ),
            (
                f"{validated_prefix}"
                f"_output_{output_index}"
            ),
        )

    return ChiStarMILPVariables(
        products=products,
        parity=parity,
    )


def add_mu_xor3_constraints(
    problem: pulp.LpProblem,
    inputs: Sequence[pulp.LpVariable],
    output: pulp.LpVariable,
    *,
    prefix: str,
) -> MuMILPVariables:
    """
    Modela exactamente:

        output = inputs[0] XOR inputs[1] XOR inputs[2].

    Se crea una variable binaria de paridad q y se agrega:

        inputs[0] + inputs[1] + inputs[2]
            =
        output + 2 q.
    """

    validated_problem = _validate_problem(
        problem
    )

    validated_prefix = _validate_prefix(
        prefix
    )

    input_variables = _validate_variable_sequence(
        inputs,
        expected_length=3,
        parameter_name="inputs",
    )

    output_variable = _validate_variable_sequence(
        (output,),
        expected_length=1,
        parameter_name="output",
    )[0]

    parity = create_binary_variable(
        validated_problem,
        f"{validated_prefix}_parity",
    )

    _add_constraint(
        validated_problem,
        (
            pulp.lpSum(input_variables)
            == output_variable + 2 * parity
        ),
        f"{validated_prefix}_xor3",
    )

    return MuMILPVariables(
        parity=parity,
    )


__all__ = [
    "CHI_STAR_PARITY_UPPER_BOUNDS",
    "CHI_STAR_QUADRATIC_PAIRS",
    "ChiStarMILPVariables",
    "MuMILPVariables",
    "add_and_equivalence",
    "add_chi_star_sbox_constraints",
    "add_mu_xor3_constraints",
    "create_binary_variable",
    "create_integer_variable",
]
