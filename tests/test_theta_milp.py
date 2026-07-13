"""
Validación de la formulación MILP de la capa theta.
"""

import numpy as np
import pytest

from keccak_milp.config import ExperimentConfig
from keccak_milp.layers import (
    create_single_active_bit_state,
    theta,
)
from keccak_milp.model import KeccakMILPModel


def numpy_to_nested_list(
    state: np.ndarray,
) -> list[list[list[int]]]:
    """Convierte un estado NumPy a listas enteras."""

    return state.astype(int).tolist()


def solve_theta_for_state(
    state: np.ndarray,
) -> tuple[KeccakMILPModel, np.ndarray]:
    """Resuelve theta MILP para un estado fijo."""

    z = state.shape[2]

    config = ExperimentConfig(
        z=z,
        rounds=1,
        solver="cbc",
        time_limit_seconds=60,
        mip_gap=0.0,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    model.add_theta_layer(round_index=0)
    model.fix_state_values(
        round_index=0,
        values=numpy_to_nested_list(state),
    )
    model.set_feasibility_objective()

    status = model.solve()

    assert status == "Optimal"

    milp_output = np.asarray(
        model.theta_output_values(0),
        dtype=np.int64,
    )

    return model, milp_output


@pytest.mark.parametrize("z", [4, 8])
def test_theta_milp_zero_state(z: int) -> None:
    state = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    _, milp_output = solve_theta_for_state(state)

    assert np.array_equal(
        milp_output,
        theta(state),
    )


@pytest.mark.parametrize("z", [4, 8])
def test_theta_milp_single_active_bit(z: int) -> None:
    state = create_single_active_bit_state(
        z=z,
        x=2,
        y=3,
        k=1,
    )

    _, milp_output = solve_theta_for_state(state)

    expected = theta(state)

    assert np.array_equal(
        milp_output,
        expected,
    )

    assert int(np.sum(milp_output)) == 11


@pytest.mark.parametrize("z", [4, 8])
def test_theta_milp_random_state(z: int) -> None:
    rng = np.random.default_rng(7200 + z)

    state = rng.integers(
        0,
        2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    _, milp_output = solve_theta_for_state(state)

    assert np.array_equal(
        milp_output,
        theta(state),
    )


def test_theta_variable_count_z4() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    # Dos estados de frontera:
    # 25 × 4 × 2 = 200.
    assert model.declared_variable_count() == 200

    model.add_theta_layer(0)

    # Variables theta:
    #
    # C, QC, D, QD = 4 × 5 × 4 = 80
    # T, QT         = 2 × 25 × 4 = 200
    #
    # Total adicional = 280.
    assert model.declared_variable_count() == 480


def test_theta_constraint_count_z4() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)
    model.add_theta_layer(0)

    # Paridades C:
    # 5 × 4 = 20
    #
    # Efectos D:
    # 5 × 4 = 20
    #
    # Salidas T:
    # 25 × 4 = 100
    #
    # Total = 140.
    assert model.constraint_count() == 140


def test_add_theta_is_idempotent() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    model.add_theta_layer(0)

    variables_first = model.declared_variable_count()
    constraints_first = model.constraint_count()

    model.add_theta_layer(0)

    assert model.declared_variable_count() == variables_first
    assert model.constraint_count() == constraints_first


def test_invalid_theta_round() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    with pytest.raises(ValueError):
        model.add_theta_layer(1)