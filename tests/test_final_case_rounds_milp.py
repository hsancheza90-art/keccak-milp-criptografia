"""Pruebas del modelo MILP multirronda del caso final."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pulp
import pytest

import keccak_milp.final_case_rounds_milp as rounds_milp
from keccak_milp.final_case import (
    keccak_rounds_final_case,
    security_level_to_rounds,
)
from keccak_milp.final_case_rounds_milp import (
    FinalCaseRoundsMILPModel,
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
) -> FinalCaseRoundsMILPModel:
    """Construye un modelo listo para resolver."""

    model = FinalCaseRoundsMILPModel(
        z=z,
        security_level=security_level,
        domain_id=domain_id,
        name=name,
    )

    model.build_model()
    model.set_feasibility_objective()

    return model


def expected_structure(
    *,
    z: int,
    rounds: int,
) -> tuple[int, int, int]:
    """Calcula variables, restricciones y conexiones."""

    variables = (
        rounds
        * 200
        * z
    )

    boundaries = (
        max(
            rounds - 1,
            0,
        )
        * 25
        * z
    )

    constraints = (
        rounds
        * 225
        * z
        + boundaries
    )

    return (
        variables,
        constraints,
        boundaries,
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


def test_public_interface() -> None:
    expected = {
        "FinalCaseRoundsMILPModel",
        "FinalCaseRoundsMILPStatistics",
    }

    assert set(
        rounds_milp.__all__
    ) == expected

    assert len(
        rounds_milp.__all__
    ) == len(
        set(rounds_milp.__all__)
    )

    for name in rounds_milp.__all__:
        assert hasattr(
            rounds_milp,
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
    model = FinalCaseRoundsMILPModel(
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
        expected_boundaries,
    ) = expected_structure(
        z=z,
        rounds=rounds,
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
        statistics.boundary_constraints
        == expected_boundaries
    )

    assert statistics.round_variables == (
        200 * z,
    ) * rounds

    assert statistics.round_constraints == (
        225 * z,
    ) * rounds

    assert (
        model.problem.numVariables()
        == expected_variables
    )

    assert (
        model.problem.numConstraints()
        == expected_constraints
    )

    assert len(
        model.round_models
    ) == rounds

    assert len(
        model.boundary_constraint_names
    ) == expected_boundaries

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
    model = FinalCaseRoundsMILPModel(
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

    first_round_models = tuple(
        id(round_model)
        for round_model
        in model.round_models
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

    assert tuple(
        id(round_model)
        for round_model
        in model.round_models
    ) == first_round_models

    model.set_feasibility_objective()
    model.set_feasibility_objective()


@pytest.mark.parametrize(
    (
        "z",
        "security_level",
        "domain_id",
    ),
    VALID_CASES,
)
def test_multiround_matches_functional_reference(
    z: int,
    security_level: int,
    domain_id: int,
    solver: pulp.LpSolver,
) -> None:
    seed = (
        110000
        + 1000 * z
        + 100 * security_level
        + domain_id
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

    original = state.copy()

    expected = keccak_rounds_final_case(
        state,
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

    model.fix_initial_state_values(
        state
    )

    assert model.solve(
        solver
    ) == "Optimal"

    obtained = model.final_state_values()

    assert np.array_equal(
        obtained,
        expected,
    )

    assert np.array_equal(
        state,
        original,
    )


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_model_rejects_wrong_final_output(
    z: int,
    solver: pulp.LpSolver,
) -> None:
    rng = np.random.default_rng(
        113000 + z
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

    expected = keccak_rounds_final_case(
        state,
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
        name=f"wrong_final_z{z}",
    )

    model.fix_initial_state_values(
        state
    )

    model.fix_final_state_values(
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
def test_fix_initial_state_is_idempotent_and_nonmutating(
    z: int,
) -> None:
    rng = np.random.default_rng(
        114000 + z
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
        name=f"fix_initial_z{z}",
    )

    initial_constraints = (
        model.problem.numConstraints()
    )

    model.fix_initial_state_values(
        state
    )

    fixed_constraints = (
        model.problem.numConstraints()
    )

    assert (
        fixed_constraints
        == initial_constraints + 25 * z
    )

    model.fix_initial_state_values(
        state.copy()
    )

    assert (
        model.problem.numConstraints()
        == fixed_constraints
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
        model.fix_initial_state_values(
            different
        )


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_fix_final_state_is_idempotent_and_nonmutating(
    z: int,
) -> None:
    rng = np.random.default_rng(
        115000 + z
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
        security_level=2,
        domain_id=1,
        name=f"fix_final_z{z}",
    )

    initial_constraints = (
        model.problem.numConstraints()
    )

    model.fix_final_state_values(
        state
    )

    fixed_constraints = (
        model.problem.numConstraints()
    )

    assert (
        fixed_constraints
        == initial_constraints + 25 * z
    )

    model.fix_final_state_values(
        state.copy()
    )

    assert (
        model.problem.numConstraints()
        == fixed_constraints
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
        model.fix_final_state_values(
            different
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
def test_round_model_accessor(
    z: int,
    security_level: int,
) -> None:
    model = FinalCaseRoundsMILPModel(
        z=z,
        security_level=security_level,
        domain_id=0,
        name=(
            f"accessor_z{z}"
            f"_s{security_level}"
        ),
    )

    model.build_model()

    rounds = security_level_to_rounds(
        security_level
    )

    for round_index in range(
        rounds
    ):
        round_model = model.round_model(
            round_index
        )

        assert (
            round_model.round_index
            == round_index
        )

        assert round_model.z == z
        assert (
            round_model.security_level
            == security_level
        )

        assert (
            model.initial_state_variable(
                0,
                0,
                0,
            )
            is model.round_model(
                0
            ).input_variable(
                0,
                0,
                0,
            )
        )

        assert (
            model.final_state_variable(
                0,
                0,
                0,
            )
            is model.round_model(
                rounds - 1
            ).output_variable(
                0,
                0,
                0,
            )
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
        FinalCaseRoundsMILPModel(
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
        FinalCaseRoundsMILPModel(
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
        FinalCaseRoundsMILPModel(
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
        FinalCaseRoundsMILPModel(
            z=4,
            security_level=0,
            domain_id=0,
            name=invalid_name,  # type: ignore[arg-type]
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
def test_rejects_invalid_initial_state(
    invalid_case: str,
) -> None:
    model = build_ready_model(
        z=4,
        security_level=1,
        domain_id=0,
        name=(
            f"invalid_initial_"
            f"{invalid_case}"
        ),
    )

    with pytest.raises(
        (
            TypeError,
            ValueError,
        ),
    ):
        model.fix_initial_state_values(
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
def test_rejects_invalid_final_state(
    invalid_case: str,
) -> None:
    model = build_ready_model(
        z=4,
        security_level=2,
        domain_id=0,
        name=(
            f"invalid_final_"
            f"{invalid_case}"
        ),
    )

    with pytest.raises(
        (
            TypeError,
            ValueError,
        ),
    ):
        model.fix_final_state_values(
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
def test_rejects_invalid_round_model_index(
    invalid_round_index: object,
) -> None:
    model = FinalCaseRoundsMILPModel(
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
        model.round_model(
            invalid_round_index  # type: ignore[arg-type]
        )


def test_lifecycle_guards(
    solver: pulp.LpSolver,
) -> None:
    model = FinalCaseRoundsMILPModel(
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
        model.fix_initial_state_values(
            state
        )

    with pytest.raises(
        RuntimeError,
        match=r"build_model\(\)",
    ):
        model.initial_state_variable(
            0,
            0,
            0,
        )

    with pytest.raises(
        RuntimeError,
        match=r"build_model\(\)",
    ):
        model.round_model(
            0
        )

    model.build_model()

    with pytest.raises(
        RuntimeError,
        match="función objetivo",
    ):
        model.solve(
            solver
        )

    model.set_feasibility_objective()

    with pytest.raises(
        RuntimeError,
        match="solución",
    ):
        model.final_state_values()
