"""
Validación MILP de las capas rho y pi.
"""

import numpy as np
import pytest

from keccak_milp.config import ExperimentConfig
from keccak_milp.layers import (
    create_single_active_bit_state,
    rho_pi,
    theta,
)
from keccak_milp.model import KeccakMILPModel


def solve_linear_layers(
    state: np.ndarray,
) -> tuple[KeccakMILPModel, np.ndarray]:
    """
    Resuelve theta, rho y pi mediante MILP para una entrada fija.
    """

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

    model.add_linear_layers(round_index=0)

    model.fix_state_values(
        round_index=0,
        values=state.astype(int).tolist(),
    )

    model.set_feasibility_objective()

    status = model.solve()

    assert status == "Optimal"

    output = np.asarray(
        model.rho_pi_output_values(0),
        dtype=np.int64,
    )

    return model, output


@pytest.mark.parametrize("z", [4, 8])
def test_rho_pi_milp_zero_state(z: int) -> None:
    state = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    _, milp_output = solve_linear_layers(state)

    expected = rho_pi(theta(state))

    assert np.array_equal(
        milp_output,
        expected,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_rho_pi_milp_single_active_bit(z: int) -> None:
    state = create_single_active_bit_state(
        z=z,
        x=2,
        y=3,
        k=1,
    )

    _, milp_output = solve_linear_layers(state)

    expected = rho_pi(theta(state))

    assert np.array_equal(
        milp_output,
        expected,
    )

    # Rho y pi son permutaciones y deben conservar el peso.
    assert int(np.sum(milp_output)) == int(
        np.sum(theta(state))
    )


@pytest.mark.parametrize("z", [4, 8])
def test_rho_pi_milp_random_state(z: int) -> None:
    rng = np.random.default_rng(8100 + z)

    state = rng.integers(
        0,
        2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    _, milp_output = solve_linear_layers(state)

    expected = rho_pi(theta(state))

    assert np.array_equal(
        milp_output,
        expected,
    )


def test_rho_pi_requires_theta() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    with pytest.raises(RuntimeError):
        model.add_rho_pi_layers(0)


def test_rho_pi_variable_count_z4() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    # Estados de frontera.
    assert model.declared_variable_count() == 200

    model.add_theta_layer(0)

    # Estados + theta.
    assert model.declared_variable_count() == 480

    model.add_rho_pi_layers(0)

    # Rho-pi añade 25 × 4 = 100.
    assert model.declared_variable_count() == 580


def test_rho_pi_constraint_count_z4() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    model.add_linear_layers(0)

    # Theta: 35z = 140.
    # Rho-pi: 25z = 100.
    #
    # Total: 240.
    assert model.constraint_count() == 240


def test_rho_pi_is_idempotent() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    model.add_linear_layers(0)

    variables_first = model.declared_variable_count()
    constraints_first = model.constraint_count()

    model.add_rho_pi_layers(0)

    assert model.declared_variable_count() == variables_first
    assert model.constraint_count() == constraints_first


@pytest.mark.parametrize("z", [4, 8])
def test_rho_pi_output_preserves_theta_weight(z: int) -> None:
    rng = np.random.default_rng(9200 + z)

    state = rng.integers(
        0,
        2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    _, output = solve_linear_layers(state)

    theta_output = theta(state)

    assert int(np.sum(output)) == int(
        np.sum(theta_output)
    )