"""Pruebas del modelo MILP permanente de una ronda final."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pulp
import pytest

import keccak_milp.final_case_round_milp as round_milp
from keccak_milp.diffusion import (
    linear_layer_final_case,
)
from keccak_milp.final_case import (
    keccak_round_final_case,
)
from keccak_milp.final_case_round_milp import (
    FinalCaseRoundMILPModel,
    build_final_case_linear_matrix,
    final_case_state_positions,
)


VALID_CASES = tuple(
    (
        z,
        security_level,
        domain_id,
        round_index,
    )
    for z in (4, 8)
    for security_level in (0, 1, 2)
    for domain_id in (0, 1, 2, 3)
    for round_index in range(
        security_level + 1
    )
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


def gf2_rank(
    matrix: np.ndarray,
) -> int:
    """Calcula el rango de una matriz binaria."""

    reduced = (
        matrix.copy().astype(np.uint8)
        & 1
    )

    rows, columns = reduced.shape

    pivot_row = 0
    pivot_column = 0

    while (
        pivot_row < rows
        and pivot_column < columns
    ):
        candidates = np.flatnonzero(
            reduced[
                pivot_row:,
                pivot_column,
            ]
        )

        if candidates.size == 0:
            pivot_column += 1
            continue

        selected = (
            pivot_row
            + int(candidates[0])
        )

        if selected != pivot_row:
            reduced[
                [
                    pivot_row,
                    selected,
                ],
                :,
            ] = reduced[
                [
                    selected,
                    pivot_row,
                ],
                :,
            ]

        for row_index in range(rows):
            if (
                row_index != pivot_row
                and reduced[
                    row_index,
                    pivot_column,
                ]
            ):
                reduced[row_index, :] ^= (
                    reduced[pivot_row, :]
                )

        pivot_row += 1
        pivot_column += 1

    return pivot_row


def matrix_apply(
    matrix: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Aplica una matriz sobre GF(2)."""

    vector = state.reshape(
        -1
    ).astype(
        np.uint16
    )

    output = (
        matrix.astype(np.uint16)
        @ vector
    ) % 2

    return output.astype(
        np.int64
    ).reshape(
        state.shape
    )


def build_ready_model(
    *,
    z: int,
    security_level: int,
    domain_id: int,
    round_index: int,
    name: str,
) -> FinalCaseRoundMILPModel:
    model = FinalCaseRoundMILPModel(
        z=z,
        security_level=security_level,
        domain_id=domain_id,
        round_index=round_index,
        name=name,
    )

    model.build_model()
    model.set_feasibility_objective()

    return model


def test_public_interface() -> None:
    expected = {
        "FinalCaseRoundMILPModel",
        "FinalCaseRoundMILPStatistics",
        "StatePosition",
        "build_final_case_linear_matrix",
        "final_case_state_positions",
    }

    assert set(
        round_milp.__all__
    ) == expected

    assert len(
        round_milp.__all__
    ) == len(
        set(round_milp.__all__)
    )

    for name in round_milp.__all__:
        assert hasattr(
            round_milp,
            name,
        )


@pytest.mark.parametrize(
    "z, expected_row_weights, expected_column_weights",
    [
        (
            4,
            (19, 33),
            (25, 31),
        ),
        (
            8,
            (23, 33),
            (25, 33),
        ),
    ],
)
def test_linear_matrix_structure(
    z: int,
    expected_row_weights: tuple[int, int],
    expected_column_weights: tuple[int, int],
) -> None:
    matrix = build_final_case_linear_matrix(
        z
    )

    state_bits = 25 * z

    assert matrix.shape == (
        state_bits,
        state_bits,
    )

    assert matrix.dtype == np.uint8
    assert gf2_rank(matrix) == state_bits

    row_weights = matrix.sum(
        axis=1
    )

    column_weights = matrix.sum(
        axis=0
    )

    assert (
        int(row_weights.min()),
        int(row_weights.max()),
    ) == expected_row_weights

    assert (
        int(column_weights.min()),
        int(column_weights.max()),
    ) == expected_column_weights

    matrix[
        0,
        0,
    ] ^= 1

    fresh_matrix = build_final_case_linear_matrix(
        z
    )

    assert not np.array_equal(
        matrix,
        fresh_matrix,
    )


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_linear_matrix_matches_functional_layer(
    z: int,
) -> None:
    matrix = build_final_case_linear_matrix(
        z
    )

    rng = np.random.default_rng(
        88000 + z
    )

    for _ in range(12):
        state = rng.integers(
            low=0,
            high=2,
            size=(
                5,
                5,
                z,
            ),
            dtype=np.int64,
        )

        expected = linear_layer_final_case(
            state
        )

        obtained = matrix_apply(
            matrix,
            state,
        )

        assert np.array_equal(
            obtained,
            expected,
        )


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_model_structure(
    z: int,
) -> None:
    model = FinalCaseRoundMILPModel(
        z=z,
        security_level=2,
        domain_id=3,
        round_index=2,
        name=f"structure_z{z}",
    )

    model.build_model()

    statistics = model.statistics()

    assert statistics.z == z
    assert statistics.state_bits == 25 * z
    assert statistics.sboxes == 5 * z

    assert statistics.variables == 200 * z
    assert statistics.constraints == 225 * z

    assert (
        statistics.linear_parity_variables
        == 25 * z
    )

    assert (
        statistics.chi_auxiliary_variables
        == 75 * z
    )

    assert model.problem.numVariables() == (
        200 * z
    )

    assert model.problem.numConstraints() == (
        225 * z
    )

    assert len(model.input_state) == 25 * z
    assert len(model.linear_output) == 25 * z
    assert len(model.chi_output) == 25 * z
    assert len(model.round_output) == 25 * z
    assert len(model.linear_parity) == 25 * z
    assert len(model.chi_auxiliary) == 5 * z


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_build_model_is_idempotent(
    z: int,
) -> None:
    model = FinalCaseRoundMILPModel(
        z=z,
        security_level=1,
        domain_id=2,
        round_index=1,
        name=f"idempotent_z{z}",
    )

    model.build_model()

    first_variables = (
        model.problem.numVariables()
    )

    first_constraints = (
        model.problem.numConstraints()
    )

    model.build_model()

    assert (
        model.problem.numVariables()
        == first_variables
    )

    assert (
        model.problem.numConstraints()
        == first_constraints
    )

    model.set_feasibility_objective()
    model.set_feasibility_objective()


@pytest.mark.parametrize(
    (
        "z",
        "security_level",
        "domain_id",
        "round_index",
    ),
    VALID_CASES,
)
def test_round_model_matches_functional_reference(
    z: int,
    security_level: int,
    domain_id: int,
    round_index: int,
    solver: pulp.LpSolver,
) -> None:
    seed = (
        90000
        + 1000 * z
        + 100 * security_level
        + 10 * domain_id
        + round_index
    )

    rng = np.random.default_rng(
        seed
    )

    state = rng.integers(
        low=0,
        high=2,
        size=(
            5,
            5,
            z,
        ),
        dtype=np.int64,
    )

    expected = keccak_round_final_case(
        state,
        round_index=round_index,
        security_level=security_level,
        domain_id=domain_id,
    )

    model = build_ready_model(
        z=z,
        security_level=security_level,
        domain_id=domain_id,
        round_index=round_index,
        name=(
            f"valid_z{z}"
            f"_s{security_level}"
            f"_d{domain_id}"
            f"_r{round_index}"
        ),
    )

    model.fix_input_values(
        state
    )

    assert model.solve(
        solver
    ) == "Optimal"

    obtained = model.output_values()

    assert np.array_equal(
        obtained,
        expected,
    )


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_model_rejects_wrong_output(
    z: int,
    solver: pulp.LpSolver,
) -> None:
    rng = np.random.default_rng(
        93000 + z
    )

    state = rng.integers(
        low=0,
        high=2,
        size=(
            5,
            5,
            z,
        ),
        dtype=np.int64,
    )

    expected = keccak_round_final_case(
        state,
        round_index=2,
        security_level=2,
        domain_id=3,
    )

    wrong_output = expected.copy()
    wrong_output[
        0,
        0,
        0,
    ] ^= 1

    model = build_ready_model(
        z=z,
        security_level=2,
        domain_id=3,
        round_index=2,
        name=f"wrong_output_z{z}",
    )

    model.fix_input_values(
        state
    )

    model.fix_output_values(
        wrong_output
    )

    assert model.solve(
        solver
    ) == "Infeasible"


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_fix_input_does_not_mutate_state(
    z: int,
) -> None:
    rng = np.random.default_rng(
        94000 + z
    )

    state = rng.integers(
        low=0,
        high=2,
        size=(
            5,
            5,
            z,
        ),
        dtype=np.int64,
    )

    original = state.copy()

    model = build_ready_model(
        z=z,
        security_level=0,
        domain_id=0,
        round_index=0,
        name=f"no_mutation_z{z}",
    )

    model.fix_input_values(
        state
    )

    model.fix_input_values(
        state.copy()
    )

    assert np.array_equal(
        state,
        original,
    )

    different = state.copy()
    different[
        0,
        0,
        0,
    ] ^= 1

    with pytest.raises(
        RuntimeError,
        match="valores diferentes",
    ):
        model.fix_input_values(
            different
        )


@pytest.mark.parametrize(
    "invalid_z",
    [
        True,
        4.0,
        "4",
        None,
        1,
        16,
    ],
)
def test_rejects_invalid_word_size(
    invalid_z: object,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        FinalCaseRoundMILPModel(
            z=invalid_z,  # type: ignore[arg-type]
            security_level=0,
            domain_id=0,
            round_index=0,
        )


@pytest.mark.parametrize(
    "invalid_security",
    [
        True,
        0.0,
        "0",
        None,
        -1,
        3,
    ],
)
def test_rejects_invalid_security_level(
    invalid_security: object,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        FinalCaseRoundMILPModel(
            z=4,
            security_level=invalid_security,  # type: ignore[arg-type]
            domain_id=0,
            round_index=0,
        )


@pytest.mark.parametrize(
    "invalid_domain",
    [
        True,
        0.0,
        "0",
        None,
        -1,
        4,
    ],
)
def test_rejects_invalid_domain_id(
    invalid_domain: object,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        FinalCaseRoundMILPModel(
            z=4,
            security_level=0,
            domain_id=invalid_domain,  # type: ignore[arg-type]
            round_index=0,
        )


@pytest.mark.parametrize(
    "invalid_round",
    [
        True,
        0.0,
        "0",
        None,
        -1,
        1,
    ],
)
def test_rejects_invalid_round_index(
    invalid_round: object,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
    ):
        FinalCaseRoundMILPModel(
            z=4,
            security_level=0,
            domain_id=0,
            round_index=invalid_round,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_case",
    [
        "not_array",
        "wrong_shape",
        "floating",
        "nonbinary",
    ],
)
def test_rejects_invalid_input_state(
    invalid_case: str,
) -> None:
    model = build_ready_model(
        z=4,
        security_level=0,
        domain_id=0,
        round_index=0,
        name=f"invalid_state_{invalid_case}",
    )

    if invalid_case == "not_array":
        values: object = [
            0,
            1,
        ]
    elif invalid_case == "wrong_shape":
        values = np.zeros(
            (
                5,
                5,
                8,
            ),
            dtype=np.int64,
        )
    elif invalid_case == "floating":
        values = np.zeros(
            (
                5,
                5,
                4,
            ),
            dtype=np.float64,
        )
    else:
        values = np.zeros(
            (
                5,
                5,
                4,
            ),
            dtype=np.int64,
        )

        values[
            0,
            0,
            0,
        ] = 2

    with pytest.raises(
        (TypeError, ValueError),
    ):
        model.fix_input_values(
            values
        )


def test_lifecycle_guards(
    solver: pulp.LpSolver,
) -> None:
    model = FinalCaseRoundMILPModel(
        z=4,
        security_level=0,
        domain_id=0,
        round_index=0,
        name="lifecycle",
    )

    state = np.zeros(
        (
            5,
            5,
            4,
        ),
        dtype=np.int64,
    )

    with pytest.raises(
        RuntimeError,
        match=r"build_model\(\)",
    ):
        model.fix_input_values(
            state
        )

    with pytest.raises(
        RuntimeError,
        match=r"build_model\(\)",
    ):
        model.output_variable(
            0,
            0,
            0,
        )

    model.build_model()

    with pytest.raises(
        RuntimeError,
        match="función objetivo",
    ):
        model.solve(
            solver
        )

    with pytest.raises(
        RuntimeError,
        match="solución",
    ):
        model.output_values()
