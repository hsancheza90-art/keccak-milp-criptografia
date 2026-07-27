import numpy as np
import pytest

import keccak_milp.layers as layers

from keccak_milp.config import ExperimentConfig
from keccak_milp.differential import (
    PairedKeccakMILPModel,
)


def build_paired_model(
    z: int = 4,
    rounds: int = 1,
) -> PairedKeccakMILPModel:
    """Construye un modelo emparejado completo."""
    config = ExperimentConfig(
        z=z,
        rounds=rounds,
        solver="cbc",
        time_limit_seconds=60,
        verbose=False,
    )

    model = PairedKeccakMILPModel(config)
    model.build_paired_model()

    return model


def fix_input_pair(
    model: PairedKeccakMILPModel,
    left_state: np.ndarray,
    right_state: np.ndarray,
) -> None:
    """Fija los estados iniciales de ambas ejecuciones."""
    expected_shape = (
        5,
        5,
        model.config.z,
    )

    assert left_state.shape == expected_shape
    assert right_state.shape == expected_shape

    for x in range(5):
        for y in range(5):
            for k in range(model.config.z):
                model.problem += (
                    model.left.state_variable(
                        0,
                        x,
                        y,
                        k,
                    )
                    == int(left_state[x, y, k]),
                    f"fix_left_input_{x}_{y}_{k}",
                )

                model.problem += (
                    model.right.state_variable(
                        0,
                        x,
                        y,
                        k,
                    )
                    == int(right_state[x, y, k]),
                    f"fix_right_input_{x}_{y}_{k}",
                )


@pytest.mark.parametrize(
    ("z", "rounds"),
    [
        (4, 1),
        (8, 1),
        (4, 2),
    ],
)
def test_paired_model_has_expected_size(
    z: int,
    rounds: int,
) -> None:
    """El modelo presenta el tamaño estructural esperado."""
    model = build_paired_model(
        z=z,
        rounds=rounds,
    )

    single_variables = (
        25 * z
        + 195 * z * rounds
    )

    difference_variables = (
        2
        * (rounds + 1)
        * 25
        * z
    )

    expected_variables = (
        2 * single_variables
        + difference_variables
    )

    expected_constraints = (
        2 * 185 * z * rounds
        + (rounds + 1) * 25 * z
    )

    assert (
        model.declared_variable_count()
        == expected_variables
    )

    assert (
        model.constraint_count()
        == expected_constraints
    )


def test_build_paired_model_is_idempotent() -> None:
    """La construcción no duplica el problema."""
    model = build_paired_model()

    variables_before = (
        model.declared_variable_count()
    )

    constraints_before = (
        model.constraint_count()
    )

    model.build_paired_model()

    assert (
        model.declared_variable_count()
        == variables_before
    )

    assert (
        model.constraint_count()
        == constraints_before
    )


def test_difference_access_requires_build() -> None:
    """No se accede a diferencias antes de construir."""
    config = ExperimentConfig(
        z=4,
        rounds=1,
        solver="cbc",
        verbose=False,
    )

    model = PairedKeccakMILPModel(config)

    with pytest.raises(
        RuntimeError,
        match="build_paired_model",
    ):
        model.difference_variable(
            0,
            0,
            0,
            0,
        )

@pytest.mark.parametrize(
    ("seed_left", "seed_right"),
    [
        (7, 2026),
        (640, 701),
    ],
)
def test_fixed_pair_matches_exact_xor(
    seed_left: int,
    seed_right: int,
) -> None:
    """Las diferencias MILP coinciden con el XOR exacto."""
    z = 4
    rounds = 1

    rng_left = np.random.default_rng(
        seed_left
    )

    rng_right = np.random.default_rng(
        seed_right
    )

    left_input = rng_left.integers(
        0,
        2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    right_input = rng_right.integers(
        0,
        2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    model = build_paired_model(
        z=z,
        rounds=rounds,
    )

    fix_input_pair(
        model,
        left_input,
        right_input,
    )

    model.set_boundary_difference_weight_objective(
        boundary_index=0
    )

    status = model.solve()

    assert status == "Optimal"

    left_output = layers.keccak_rounds(
        left_input,
        number_of_rounds=rounds,
    )

    right_output = layers.keccak_rounds(
        right_input,
        number_of_rounds=rounds,
    )

    expected_input_difference = np.bitwise_xor(
        left_input,
        right_input,
    )

    expected_output_difference = np.bitwise_xor(
        left_output,
        right_output,
    )

    obtained_input_difference = (
        model.difference_state_values(0)
    )

    obtained_output_difference = (
        model.difference_state_values(rounds)
    )

    assert np.array_equal(
        obtained_input_difference,
        expected_input_difference,
    )

    assert np.array_equal(
        obtained_output_difference,
        expected_output_difference,
    )

    assert model.objective_value() == pytest.approx(
        float(expected_input_difference.sum())
    )


def test_identical_inputs_produce_zero_differences() -> None:
    """Dos ejecuciones iguales permanecen iguales."""
    z = 4

    rng = np.random.default_rng(2026)

    state = rng.integers(
        0,
        2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    model = build_paired_model(
        z=z,
        rounds=1,
    )

    fix_input_pair(
        model,
        state,
        state.copy(),
    )

    model.set_input_output_difference_objective()

    status = model.solve()

    assert status == "Optimal"

    assert int(
        model.difference_state_values(0).sum()
    ) == 0

    assert int(
        model.difference_state_values(1).sum()
    ) == 0

    assert model.objective_value() == pytest.approx(
        0.0
    )


def test_nonzero_input_difference_constraint_is_idempotent() -> None:
    """La exclusión de diferencia nula no se duplica."""
    model = build_paired_model()

    constraints_before = (
        model.constraint_count()
    )

    model.add_nonzero_input_difference_constraint()

    constraints_after_first = (
        model.constraint_count()
    )

    model.add_nonzero_input_difference_constraint()

    constraints_after_second = (
        model.constraint_count()
    )

    assert (
        constraints_after_first
        == constraints_before + 1
    )

    assert (
        constraints_after_second
        == constraints_after_first
    )