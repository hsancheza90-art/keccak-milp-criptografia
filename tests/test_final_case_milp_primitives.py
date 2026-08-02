"""Pruebas de las primitivas MILP del caso final."""

from __future__ import annotations

from collections.abc import Iterator

import pulp
import pytest

import keccak_milp.final_case_milp as final_case_milp
from keccak_milp.final_case_milp import (
    CHI_STAR_PARITY_UPPER_BOUNDS,
    CHI_STAR_QUADRATIC_PAIRS,
    add_chi_star_sbox_constraints,
    add_mu_xor3_constraints,
    create_binary_variable,
)
from keccak_milp.nonlinear import CHI_STAR_TABLE


def add_constraint(
    problem: pulp.LpProblem,
    constraint: pulp.LpConstraint,
    name: str,
) -> None:
    problem.addConstraint(
        constraint,
        name=name,
    )


def build_chi_problem(
    prefix: str,
) -> tuple[
    pulp.LpProblem,
    tuple[pulp.LpVariable, ...],
    tuple[pulp.LpVariable, ...],
]:
    problem = pulp.LpProblem(
        f"{prefix}_problem",
        pulp.LpMinimize,
    )

    inputs = tuple(
        create_binary_variable(
            problem,
            f"{prefix}_input_{index}",
        )
        for index in range(5)
    )

    outputs = tuple(
        create_binary_variable(
            problem,
            f"{prefix}_output_variable_{index}",
        )
        for index in range(5)
    )

    add_chi_star_sbox_constraints(
        problem,
        inputs,
        outputs,
        prefix=f"{prefix}_chi",
    )

    return (
        problem,
        inputs,
        outputs,
    )


def build_mu_problem(
    prefix: str,
) -> tuple[
    pulp.LpProblem,
    tuple[pulp.LpVariable, ...],
    pulp.LpVariable,
]:
    problem = pulp.LpProblem(
        f"{prefix}_problem",
        pulp.LpMinimize,
    )

    inputs = tuple(
        create_binary_variable(
            problem,
            f"{prefix}_input_{index}",
        )
        for index in range(3)
    )

    output = create_binary_variable(
        problem,
        f"{prefix}_output",
    )

    add_mu_xor3_constraints(
        problem,
        inputs,
        output,
        prefix=f"{prefix}_mu",
    )

    return (
        problem,
        inputs,
        output,
    )


@pytest.fixture(scope="module")
def solver() -> Iterator[pulp.LpSolver]:
    cbc = pulp.PULP_CBC_CMD(
        msg=False,
    )

    if not cbc.available():
        pytest.skip(
            "CBC no está disponible."
        )

    yield cbc


def solve_status(
    problem: pulp.LpProblem,
    solver: pulp.LpSolver,
) -> str:
    problem.setObjective(
        pulp.LpAffineExpression()
    )

    problem.solve(solver)

    return pulp.LpStatus[
        problem.status
    ]


def extract_output(
    outputs: tuple[pulp.LpVariable, ...],
) -> int:
    result = 0

    for index, variable in enumerate(outputs):
        value = variable.value()

        assert value is not None

        result |= (
            int(value > 0.5)
            << index
        )

    return result


def test_public_interface() -> None:
    expected = {
        "CHI_STAR_PARITY_UPPER_BOUNDS",
        "CHI_STAR_QUADRATIC_PAIRS",
        "ChiStarMILPVariables",
        "MuMILPVariables",
        "add_and_equivalence",
        "add_chi_star_sbox_constraints",
        "add_mu_xor3_constraints",
        "create_binary_variable",
        "create_integer_variable",
    }

    assert set(
        final_case_milp.__all__
    ) == expected

    assert len(
        final_case_milp.__all__
    ) == len(
        set(final_case_milp.__all__)
    )

    for name in final_case_milp.__all__:
        assert hasattr(
            final_case_milp,
            name,
        )


def test_chi_constants() -> None:
    assert len(
        CHI_STAR_QUADRATIC_PAIRS
    ) == 10

    assert set(
        CHI_STAR_QUADRATIC_PAIRS
    ) == {
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
    }

    assert CHI_STAR_PARITY_UPPER_BOUNDS == (
        2,
        3,
        3,
        3,
        3,
    )


def test_chi_structure() -> None:
    problem, _, _ = build_chi_problem(
        "structure"
    )

    assert problem.numVariables() == 25
    assert problem.numConstraints() == 35

    variable_names = {
        variable.name
        for variable in problem.variables()
    }

    product_names = {
        (
            f"structure_chi"
            f"_product_{first}{second}"
        )
        for first, second
        in CHI_STAR_QUADRATIC_PAIRS
    }

    parity_names = {
        f"structure_chi_parity_{index}"
        for index in range(5)
    }

    assert product_names <= variable_names
    assert parity_names <= variable_names


def test_chi_matches_table_exhaustively(
    solver: pulp.LpSolver,
) -> None:
    for input_value in range(32):
        problem, inputs, outputs = (
            build_chi_problem(
                f"valid_{input_value:02d}"
            )
        )

        for index, variable in enumerate(inputs):
            add_constraint(
                problem,
                (
                    variable
                    == (
                        (input_value >> index)
                        & 1
                    )
                ),
                f"fix_input_{index}",
            )

        assert solve_status(
            problem,
            solver,
        ) == "Optimal"

        assert extract_output(
            outputs
        ) == CHI_STAR_TABLE[input_value]


def test_chi_rejects_all_wrong_outputs(
    solver: pulp.LpSolver,
) -> None:
    for input_value in range(32):
        problem, inputs, outputs = (
            build_chi_problem(
                f"forbidden_{input_value:02d}"
            )
        )

        for index, variable in enumerate(inputs):
            add_constraint(
                problem,
                (
                    variable
                    == (
                        (input_value >> index)
                        & 1
                    )
                ),
                f"fix_input_{index}",
            )

        expected_output = CHI_STAR_TABLE[
            input_value
        ]

        differences = []

        for index, output in enumerate(outputs):
            expected_bit = (
                expected_output >> index
            ) & 1

            differences.append(
                1 - output
                if expected_bit
                else output
            )

        add_constraint(
            problem,
            pulp.lpSum(differences) >= 1,
            "force_alternative_output",
        )

        assert solve_status(
            problem,
            solver,
        ) == "Infeasible"


def test_mu_structure() -> None:
    problem, _, _ = build_mu_problem(
        "mu_structure"
    )

    assert problem.numVariables() == 5
    assert problem.numConstraints() == 1

    variable_names = {
        variable.name
        for variable in problem.variables()
    }

    assert (
        "mu_structure_mu_parity"
        in variable_names
    )


def test_mu_matches_xor_exhaustively(
    solver: pulp.LpSolver,
) -> None:
    for input_value in range(8):
        problem, inputs, output = (
            build_mu_problem(
                f"mu_valid_{input_value}"
            )
        )

        bits = tuple(
            (input_value >> index) & 1
            for index in range(3)
        )

        for index, variable in enumerate(inputs):
            add_constraint(
                problem,
                variable == bits[index],
                f"fix_input_{index}",
            )

        assert solve_status(
            problem,
            solver,
        ) == "Optimal"

        value = output.value()

        assert value is not None

        assert int(value > 0.5) == (
            bits[0]
            ^ bits[1]
            ^ bits[2]
        )


def test_mu_rejects_wrong_outputs(
    solver: pulp.LpSolver,
) -> None:
    for input_value in range(8):
        problem, inputs, output = (
            build_mu_problem(
                f"mu_wrong_{input_value}"
            )
        )

        bits = tuple(
            (input_value >> index) & 1
            for index in range(3)
        )

        expected_output = (
            bits[0]
            ^ bits[1]
            ^ bits[2]
        )

        for index, variable in enumerate(inputs):
            add_constraint(
                problem,
                variable == bits[index],
                f"fix_input_{index}",
            )

        add_constraint(
            problem,
            output == 1 - expected_output,
            "force_wrong_output",
        )

        assert solve_status(
            problem,
            solver,
        ) == "Infeasible"


def test_prefixes_create_disjoint_names() -> None:
    problem = pulp.LpProblem(
        "disjoint",
        pulp.LpMinimize,
    )

    first_inputs = tuple(
        create_binary_variable(
            problem,
            f"first_input_{index}",
        )
        for index in range(5)
    )

    first_outputs = tuple(
        create_binary_variable(
            problem,
            f"first_output_{index}",
        )
        for index in range(5)
    )

    second_inputs = tuple(
        create_binary_variable(
            problem,
            f"second_input_{index}",
        )
        for index in range(5)
    )

    second_outputs = tuple(
        create_binary_variable(
            problem,
            f"second_output_{index}",
        )
        for index in range(5)
    )

    add_chi_star_sbox_constraints(
        problem,
        first_inputs,
        first_outputs,
        prefix="first_chi",
    )

    add_chi_star_sbox_constraints(
        problem,
        second_inputs,
        second_outputs,
        prefix="second_chi",
    )

    names = [
        variable.name
        for variable in problem.variables()
    ]

    assert len(names) == len(set(names))
    assert problem.numVariables() == 50
    assert problem.numConstraints() == 70


@pytest.mark.parametrize(
    "invalid_prefix",
    [
        None,
        1,
        "",
        "   ",
    ],
)
def test_chi_rejects_invalid_prefix(
    invalid_prefix: object,
) -> None:
    problem = pulp.LpProblem(
        "invalid_prefix"
    )

    inputs = tuple(
        create_binary_variable(
            problem,
            f"input_{index}",
        )
        for index in range(5)
    )

    outputs = tuple(
        create_binary_variable(
            problem,
            f"output_{index}",
        )
        for index in range(5)
    )

    with pytest.raises(
        (TypeError, ValueError),
    ):
        add_chi_star_sbox_constraints(
            problem,
            inputs,
            outputs,
            prefix=invalid_prefix,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "input_count",
    [
        4,
        6,
    ],
)
def test_chi_rejects_wrong_input_count(
    input_count: int,
) -> None:
    problem = pulp.LpProblem(
        f"wrong_input_{input_count}"
    )

    inputs = tuple(
        create_binary_variable(
            problem,
            f"input_{index}",
        )
        for index in range(input_count)
    )

    outputs = tuple(
        create_binary_variable(
            problem,
            f"output_{index}",
        )
        for index in range(5)
    )

    with pytest.raises(
        ValueError,
        match="exactamente 5",
    ):
        add_chi_star_sbox_constraints(
            problem,
            inputs,
            outputs,
            prefix="chi",
        )


@pytest.mark.parametrize(
    "output_count",
    [
        4,
        6,
    ],
)
def test_chi_rejects_wrong_output_count(
    output_count: int,
) -> None:
    problem = pulp.LpProblem(
        f"wrong_output_{output_count}"
    )

    inputs = tuple(
        create_binary_variable(
            problem,
            f"input_{index}",
        )
        for index in range(5)
    )

    outputs = tuple(
        create_binary_variable(
            problem,
            f"output_{index}",
        )
        for index in range(output_count)
    )

    with pytest.raises(
        ValueError,
        match="exactamente 5",
    ):
        add_chi_star_sbox_constraints(
            problem,
            inputs,
            outputs,
            prefix="chi",
        )


def test_chi_rejects_nonvariable_input() -> None:
    problem = pulp.LpProblem(
        "nonvariable_input"
    )

    inputs: tuple[object, ...] = (
        0,
        *(
            create_binary_variable(
                problem,
                f"input_{index}",
            )
            for index in range(1, 5)
        ),
    )

    outputs = tuple(
        create_binary_variable(
            problem,
            f"output_{index}",
        )
        for index in range(5)
    )

    with pytest.raises(
        TypeError,
        match="variables PuLP",
    ):
        add_chi_star_sbox_constraints(
            problem,
            inputs,  # type: ignore[arg-type]
            outputs,
            prefix="chi",
        )


def test_chi_rejects_nonvariable_output() -> None:
    problem = pulp.LpProblem(
        "nonvariable_output"
    )

    inputs = tuple(
        create_binary_variable(
            problem,
            f"input_{index}",
        )
        for index in range(5)
    )

    outputs: tuple[object, ...] = (
        0,
        *(
            create_binary_variable(
                problem,
                f"output_{index}",
            )
            for index in range(1, 5)
        ),
    )

    with pytest.raises(
        TypeError,
        match="variables PuLP",
    ):
        add_chi_star_sbox_constraints(
            problem,
            inputs,
            outputs,  # type: ignore[arg-type]
            prefix="chi",
        )


@pytest.mark.parametrize(
    "input_count",
    [
        2,
        4,
    ],
)
def test_mu_rejects_wrong_input_count(
    input_count: int,
) -> None:
    problem = pulp.LpProblem(
        f"mu_wrong_count_{input_count}"
    )

    inputs = tuple(
        create_binary_variable(
            problem,
            f"input_{index}",
        )
        for index in range(input_count)
    )

    output = create_binary_variable(
        problem,
        "output",
    )

    with pytest.raises(
        ValueError,
        match="exactamente 3",
    ):
        add_mu_xor3_constraints(
            problem,
            inputs,
            output,
            prefix="mu",
        )


def test_mu_rejects_nonvariable_output() -> None:
    problem = pulp.LpProblem(
        "mu_nonvariable_output"
    )

    inputs = tuple(
        create_binary_variable(
            problem,
            f"input_{index}",
        )
        for index in range(3)
    )

    with pytest.raises(
        TypeError,
        match="variables PuLP",
    ):
        add_mu_xor3_constraints(
            problem,
            inputs,
            0,  # type: ignore[arg-type]
            prefix="mu",
        )
