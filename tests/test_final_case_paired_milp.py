"""Pruebas del modelo diferencial emparejado del caso final."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pulp
import pytest

import keccak_milp.final_case_paired_milp as paired_milp
from keccak_milp.diffusion import (
    linear_layer_final_case,
)
from keccak_milp.final_case import (
    keccak_round_final_case,
    security_level_to_rounds,
)
from keccak_milp.final_case_paired_milp import (
    FinalCasePairedMILPModel,
)


VALID_CASES = tuple(
    (
        z,
        security_level,
        domain_id,
    )
    for z in (
        4,
        8,
    )
    for security_level in (
        0,
        1,
        2,
    )
    for domain_id in (
        0,
        1,
        2,
        3,
    )
)


@dataclass(frozen=True)
class ReferenceTrace:
    """Traza diferencial funcional de referencia."""

    left_final: np.ndarray
    right_final: np.ndarray
    initial_difference: np.ndarray

    chi_input_differences: tuple[
        np.ndarray,
        ...,
    ]

    active_sboxes: tuple[
        np.ndarray,
        ...,
    ]


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


def build_ready_model(
    *,
    z: int,
    security_level: int,
    domain_id: int,
    name: str,
) -> FinalCasePairedMILPModel:
    """Construye el modelo y define el objetivo activo."""

    model = FinalCasePairedMILPModel(
        z=z,
        security_level=security_level,
        domain_id=domain_id,
        name=name,
    )

    model.build_model()
    model.set_active_sbox_objective()

    return model


def expected_structure(
    *,
    z: int,
    rounds: int,
) -> tuple[int, int]:
    """Calcula las dimensiones del modelo emparejado."""

    variables = (
        430 * z * rounds
        + 25 * z
    )

    constraints = (
        630 * z * rounds
        + 50 * z
        + 1
    )

    return (
        variables,
        constraints,
    )


def invalid_state(
    *,
    invalid_case: str,
    z: int,
) -> object:
    """Genera un estado deliberadamente inválido."""

    if invalid_case == "not_array":
        return [
            0,
            1,
        ]

    if invalid_case == "wrong_shape":
        wrong_z = (
            8
            if z == 4
            else 4
        )

        return np.zeros(
            (
                5,
                5,
                wrong_z,
            ),
            dtype=np.int64,
        )

    if invalid_case == "floating":
        return np.zeros(
            (
                5,
                5,
                z,
            ),
            dtype=np.float64,
        )

    values = np.zeros(
        (
            5,
            5,
            z,
        ),
        dtype=np.int64,
    )

    values[
        0,
        0,
        0,
    ] = 2

    return values


def functional_trace(
    left_state: np.ndarray,
    right_state: np.ndarray,
    *,
    security_level: int,
    domain_id: int,
) -> ReferenceTrace:
    """Calcula una traza funcional completa."""

    rounds = security_level_to_rounds(
        security_level
    )

    left_current = left_state.copy()
    right_current = right_state.copy()

    initial_difference = np.bitwise_xor(
        left_current,
        right_current,
    ).astype(
        np.int64
    )

    chi_differences: list[
        np.ndarray
    ] = []

    active_patterns: list[
        np.ndarray
    ] = []

    for round_index in range(
        rounds
    ):
        left_linear = linear_layer_final_case(
            left_current
        )

        right_linear = linear_layer_final_case(
            right_current
        )

        difference = np.bitwise_xor(
            left_linear,
            right_linear,
        ).astype(
            np.int64
        )

        active = np.any(
            difference != 0,
            axis=0,
        ).astype(
            np.int64
        )

        chi_differences.append(
            difference
        )

        active_patterns.append(
            active
        )

        left_current = keccak_round_final_case(
            left_current,
            round_index=round_index,
            security_level=security_level,
            domain_id=domain_id,
        )

        right_current = keccak_round_final_case(
            right_current,
            round_index=round_index,
            security_level=security_level,
            domain_id=domain_id,
        )

    return ReferenceTrace(
        left_final=left_current,
        right_final=right_current,
        initial_difference=initial_difference,
        chi_input_differences=tuple(
            chi_differences
        ),
        active_sboxes=tuple(
            active_patterns
        ),
    )


def variable_bit(
    variable: pulp.LpVariable,
) -> int:
    """Recupera el valor binario de una variable."""

    value = variable.value()

    assert value is not None

    return int(
        value > 0.5
    )


def test_public_interface() -> None:
    expected = {
        "FinalCasePairedMILPModel",
        "FinalCasePairedMILPStatistics",
    }

    assert set(
        paired_milp.__all__
    ) == expected

    assert len(
        paired_milp.__all__
    ) == len(
        set(paired_milp.__all__)
    )

    for name in paired_milp.__all__:
        assert hasattr(
            paired_milp,
            name,
        )


@pytest.mark.parametrize(
    "z, security_level",
    [
        (
            z,
            security_level,
        )
        for z in (
            4,
            8,
        )
        for security_level in (
            0,
            1,
            2,
        )
    ],
)
def test_model_structure(
    z: int,
    security_level: int,
) -> None:
    model = FinalCasePairedMILPModel(
        z=z,
        security_level=security_level,
        domain_id=3,
        name=(
            f"structure_z{z}"
            f"_s{security_level}"
        ),
    )

    model.build_model()

    statistics = model.statistics()

    rounds = security_level_to_rounds(
        security_level
    )

    (
        expected_variables,
        expected_constraints,
    ) = expected_structure(
        z=z,
        rounds=rounds,
    )

    branch_variables = (
        200 * z * rounds
    )

    branch_constraints = (
        225 * z * rounds
        + 25 * z * max(
            rounds - 1,
            0,
        )
    )

    assert statistics.z == z
    assert (
        statistics.security_level
        == security_level
    )
    assert statistics.domain_id == 3
    assert statistics.rounds == rounds
    assert statistics.state_bits == 25 * z

    assert (
        statistics.variables
        == expected_variables
    )

    assert (
        statistics.constraints
        == expected_constraints
    )

    assert (
        statistics.left_variables
        == branch_variables
    )

    assert (
        statistics.right_variables
        == branch_variables
    )

    assert (
        statistics.left_constraints
        == branch_constraints
    )

    assert (
        statistics.right_constraints
        == branch_constraints
    )

    assert (
        statistics.initial_difference_variables
        == 25 * z
    )

    assert (
        statistics.chi_input_difference_variables
        == 25 * z * rounds
    )

    assert (
        statistics.active_sbox_variables
        == 5 * z * rounds
    )

    variable_names = [
        variable.name
        for variable
        in model.problem.variables()
    ]

    assert len(variable_names) == len(
        set(variable_names)
    )

    constraint_names = list(
        model.problem.constraints.keys()
    )

    assert len(constraint_names) == len(
        set(constraint_names)
    )


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
    model = FinalCasePairedMILPModel(
        z=z,
        security_level=2,
        domain_id=2,
        name=f"idempotent_z{z}",
    )

    model.build_model()

    first_variables = (
        model.problem.numVariables()
    )

    first_constraints = (
        model.problem.numConstraints()
    )

    left_id = id(
        model.left
    )

    right_id = id(
        model.right
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

    assert id(model.left) == left_id
    assert id(model.right) == right_id

    model.set_active_sbox_objective()
    model.set_active_sbox_objective()


@pytest.mark.parametrize(
    (
        "z",
        "security_level",
        "domain_id",
    ),
    VALID_CASES,
)
def test_paired_model_matches_functional_reference(
    z: int,
    security_level: int,
    domain_id: int,
    solver: pulp.LpSolver,
) -> None:
    seed = (
        130000
        + 1000 * z
        + 100 * security_level
        + domain_id
    )

    rng = np.random.default_rng(
        seed
    )

    left_state = rng.integers(
        low=0,
        high=2,
        size=(
            5,
            5,
            z,
        ),
        dtype=np.int64,
    )

    right_state = left_state.copy()

    flip_count = (
        security_level + 1
    )

    flip_indices = rng.choice(
        25 * z,
        size=flip_count,
        replace=False,
    )

    right_state.reshape(
        -1
    )[
        flip_indices
    ] ^= 1

    left_original = left_state.copy()
    right_original = right_state.copy()

    trace = functional_trace(
        left_state,
        right_state,
        security_level=security_level,
        domain_id=domain_id,
    )

    model = build_ready_model(
        z=z,
        security_level=security_level,
        domain_id=domain_id,
        name=(
            f"valid_z{z}"
            f"_s{security_level}"
            f"_d{domain_id}"
        ),
    )

    model.fix_left_input_values(
        left_state
    )

    model.fix_right_input_values(
        right_state
    )

    assert model.solve(
        solver
    ) == "Optimal"

    assert np.array_equal(
        model.branch_initial_state_values(
            "left"
        ),
        left_state,
    )

    assert np.array_equal(
        model.branch_initial_state_values(
            "right"
        ),
        right_state,
    )

    assert np.array_equal(
        model.branch_final_state_values(
            "left"
        ),
        trace.left_final,
    )

    assert np.array_equal(
        model.branch_final_state_values(
            "right"
        ),
        trace.right_final,
    )

    assert np.array_equal(
        model.initial_difference_values(),
        trace.initial_difference,
    )

    expected_counts: list[int] = []

    for round_index in range(
        model.number_of_rounds
    ):
        expected_difference = (
            trace.chi_input_differences[
                round_index
            ]
        )

        expected_active = (
            trace.active_sboxes[
                round_index
            ]
        )

        assert np.array_equal(
            model.chi_input_difference_values(
                round_index
            ),
            expected_difference,
        )

        assert np.array_equal(
            model.active_sbox_values(
                round_index
            ),
            expected_active,
        )

        expected_counts.append(
            int(
                expected_active.sum()
            )
        )

        left_round = model.left.round_model(
            round_index
        )

        right_round = model.right.round_model(
            round_index
        )

        for position in model.positions:
            left_chi = variable_bit(
                left_round.chi_output_variable(
                    *position
                )
            )

            right_chi = variable_bit(
                right_round.chi_output_variable(
                    *position
                )
            )

            left_output = variable_bit(
                left_round.output_variable(
                    *position
                )
            )

            right_output = variable_bit(
                right_round.output_variable(
                    *position
                )
            )

            assert (
                left_chi ^ right_chi
            ) == (
                left_output ^ right_output
            )

    assert model.active_sbox_counts() == tuple(
        expected_counts
    )

    assert model.objective_value() == sum(
        expected_counts
    )

    assert np.array_equal(
        left_state,
        left_original,
    )

    assert np.array_equal(
        right_state,
        right_original,
    )


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_identical_inputs_are_infeasible(
    z: int,
    solver: pulp.LpSolver,
) -> None:
    rng = np.random.default_rng(
        134000 + z
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

    model = build_ready_model(
        z=z,
        security_level=0,
        domain_id=0,
        name=f"identical_z{z}",
    )

    model.fix_left_input_values(
        state
    )

    model.fix_right_input_values(
        state.copy()
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
def test_active_objective_contains_each_sbox_once(
    z: int,
) -> None:
    model = FinalCasePairedMILPModel(
        z=z,
        security_level=2,
        domain_id=3,
        name=f"objective_z{z}",
    )

    model.build_model()
    model.set_active_sbox_objective()

    coefficients = dict(
        model.problem.objective.items()
    )

    assert len(coefficients) == (
        15 * z
    )

    assert set(
        coefficients
    ) == set(
        model.active_sboxes.values()
    )

    assert all(
        coefficient == 1
        for coefficient
        in coefficients.values()
    )

    assert (
        model.problem.objective.constant
        == 0
    )


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_fix_left_input_is_idempotent_and_nonmutating(
    z: int,
) -> None:
    rng = np.random.default_rng(
        135000 + z
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
        security_level=1,
        domain_id=2,
        name=f"fix_left_z{z}",
    )

    before = model.problem.numConstraints()

    model.fix_left_input_values(
        state
    )

    after = model.problem.numConstraints()

    assert after == before + 25 * z

    model.fix_left_input_values(
        state.copy()
    )

    assert (
        model.problem.numConstraints()
        == after
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
        model.fix_left_input_values(
            different
        )


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_fix_right_input_is_idempotent_and_nonmutating(
    z: int,
) -> None:
    rng = np.random.default_rng(
        136000 + z
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
        security_level=1,
        domain_id=1,
        name=f"fix_right_z{z}",
    )

    before = model.problem.numConstraints()

    model.fix_right_input_values(
        state
    )

    after = model.problem.numConstraints()

    assert after == before + 25 * z

    model.fix_right_input_values(
        state.copy()
    )

    assert (
        model.problem.numConstraints()
        == after
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
        model.fix_right_input_values(
            different
        )


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_accessors_return_canonical_variables(
    z: int,
) -> None:
    model = FinalCasePairedMILPModel(
        z=z,
        security_level=2,
        domain_id=3,
        name=f"accessors_z{z}",
    )

    model.build_model()

    assert (
        model.branch_model("left")
        is model.left
    )

    assert (
        model.branch_model("right")
        is model.right
    )

    assert (
        model.initial_difference_variable(
            0,
            0,
            0,
        )
        is model.initial_difference[
            (
                0,
                0,
                0,
            )
        ]
    )

    assert (
        model.chi_input_difference_variable(
            2,
            0,
            0,
            0,
        )
        is model.chi_input_difference[
            (
                2,
                0,
                0,
                0,
            )
        ]
    )

    assert (
        model.active_sbox_variable(
            2,
            0,
            0,
        )
        is model.active_sboxes[
            (
                2,
                0,
                0,
            )
        ]
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
        (
            TypeError,
            ValueError,
        ),
    ):
        FinalCasePairedMILPModel(
            z=invalid_z,  # type: ignore[arg-type]
            security_level=0,
            domain_id=0,
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
        (
            TypeError,
            ValueError,
        ),
    ):
        FinalCasePairedMILPModel(
            z=4,
            security_level=invalid_security,  # type: ignore[arg-type]
            domain_id=0,
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
        (
            TypeError,
            ValueError,
        ),
    ):
        FinalCasePairedMILPModel(
            z=4,
            security_level=0,
            domain_id=invalid_domain,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_name",
    [
        1,
        True,
        "",
        "   ",
    ],
)
def test_rejects_invalid_name(
    invalid_name: object,
) -> None:
    with pytest.raises(
        (
            TypeError,
            ValueError,
        ),
    ):
        FinalCasePairedMILPModel(
            z=4,
            security_level=0,
            domain_id=0,
            name=invalid_name,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_branch",
    [
        None,
        1,
        "",
        "middle",
    ],
)
def test_rejects_invalid_branch(
    invalid_branch: object,
) -> None:
    model = FinalCasePairedMILPModel(
        z=4,
        security_level=0,
        domain_id=0,
        name="invalid_branch",
    )

    model.build_model()

    with pytest.raises(
        (
            TypeError,
            ValueError,
        ),
    ):
        model.branch_model(
            invalid_branch  # type: ignore[arg-type]
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
def test_rejects_invalid_left_state(
    invalid_case: str,
) -> None:
    model = build_ready_model(
        z=4,
        security_level=1,
        domain_id=0,
        name=(
            "invalid_left_"
            f"{invalid_case}"
        ),
    )

    with pytest.raises(
        (
            TypeError,
            ValueError,
        ),
    ):
        model.fix_left_input_values(
            invalid_state(
                invalid_case=invalid_case,
                z=4,
            )
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
def test_rejects_invalid_right_state(
    invalid_case: str,
) -> None:
    model = build_ready_model(
        z=4,
        security_level=2,
        domain_id=0,
        name=(
            "invalid_right_"
            f"{invalid_case}"
        ),
    )

    with pytest.raises(
        (
            TypeError,
            ValueError,
        ),
    ):
        model.fix_right_input_values(
            invalid_state(
                invalid_case=invalid_case,
                z=4,
            )
        )


@pytest.mark.parametrize(
    "invalid_round_index",
    [
        True,
        0.0,
        "0",
        -1,
        3,
    ],
)
def test_rejects_invalid_round_index(
    invalid_round_index: object,
) -> None:
    model = FinalCasePairedMILPModel(
        z=4,
        security_level=2,
        domain_id=0,
        name="invalid_round_index",
    )

    model.build_model()

    with pytest.raises(
        (
            TypeError,
            ValueError,
        ),
    ):
        model.active_sbox_variable(
            invalid_round_index,  # type: ignore[arg-type]
            0,
            0,
        )


def test_lifecycle_guards(
    solver: pulp.LpSolver,
) -> None:
    model = FinalCasePairedMILPModel(
        z=4,
        security_level=0,
        domain_id=0,
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
        model.fix_left_input_values(
            state
        )

    with pytest.raises(
        RuntimeError,
        match=r"build_model\(\)",
    ):
        model.initial_difference_variable(
            0,
            0,
            0,
        )

    with pytest.raises(
        RuntimeError,
        match=r"build_model\(\)",
    ):
        model.statistics()

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
        match="función objetivo",
    ):
        model.objective_value()

    model.set_active_sbox_objective()

    with pytest.raises(
        RuntimeError,
        match="solución",
    ):
        model.initial_difference_values()

    with pytest.raises(
        RuntimeError,
        match="solución",
    ):
        model.objective_value()
