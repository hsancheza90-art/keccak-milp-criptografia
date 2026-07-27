# ============================================================
# PRUEBAS DE RONDAS COMPLETAS DE KECCAK
# ============================================================

import numpy as np
import pytest

from keccak_milp.config import ExperimentConfig
from keccak_milp.layers import (
    chi,
    iota,
    keccak_round,
    keccak_rounds,
    rho_pi,
    theta,
)
from keccak_milp.model import KeccakMILPModel


@pytest.mark.parametrize(
    ("z", "round_index", "seed"),
    [
        (4, 0, 7),
        (4, 1, 2026),
        (8, 0, 640),
        (8, 2, 701),
    ],
)
def test_keccak_round_matches_manual_composition(
    z: int,
    round_index: int,
    seed: int,
) -> None:
    """Una ronda coincide con la composición manual."""
    rng = np.random.default_rng(seed)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    expected = theta(state.copy())
    expected = rho_pi(expected)
    expected = chi(expected)

    expected = iota(
        expected,
        round_index=round_index,
    )

    obtained = keccak_round(
        state,
        round_index=round_index,
    )

    assert np.array_equal(
        obtained,
        expected,
    )


@pytest.mark.parametrize(
    ("z", "number_of_rounds", "seed"),
    [
        (4, 2, 7),
        (8, 2, 2026),
        (8, 3, 640),
    ],
)
def test_keccak_rounds_matches_manual_loop(
    z: int,
    number_of_rounds: int,
    seed: int,
) -> None:
    """Varias rondas coinciden con un ciclo explícito."""
    rng = np.random.default_rng(seed)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    expected = state.copy()

    for round_index in range(
        number_of_rounds
    ):
        expected = keccak_round(
            expected,
            round_index,
        )

    obtained = keccak_rounds(
        state,
        number_of_rounds,
    )

    assert np.array_equal(
        obtained,
        expected,
    )


def test_zero_rounds_return_independent_copy() -> None:
    """Cero rondas conserva el valor, pero devuelve una copia."""
    state = np.zeros(
        (5, 5, 8),
        dtype=np.int64,
    )

    output = keccak_rounds(
        state,
        number_of_rounds=0,
    )

    assert np.array_equal(output, state)
    assert output is not state


@pytest.mark.parametrize(
    "number_of_rounds",
    [-1, -2],
)
def test_keccak_rounds_rejects_negative_count(
    number_of_rounds: int,
) -> None:
    """La cantidad de rondas no puede ser negativa."""
    state = np.zeros(
        (5, 5, 8),
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="no puede ser negativo",
    ):
        keccak_rounds(
            state,
            number_of_rounds,
        )


def test_keccak_rounds_rejects_constant_overflow() -> None:
    """No pueden solicitarse más de 24 constantes."""
    state = np.zeros(
        (5, 5, 8),
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="excede las constantes",
    ):
        keccak_rounds(
            state,
            number_of_rounds=2,
            start_round=23,
        )


def test_add_round_matches_manual_construction() -> None:
    """add_round crea la misma estructura que las llamadas manuales."""
    config = ExperimentConfig(
        z=8,
        rounds=1,
        solver="cbc",
        verbose=False,
    )

    automatic_model = KeccakMILPModel(config)
    automatic_model.add_round(0)

    manual_model = KeccakMILPModel(config)
    manual_model.add_theta_layer(0)
    manual_model.add_rho_pi_layers(0)
    manual_model.add_chi_layer(0)
    manual_model.add_iota_layer(0)

    assert (
        automatic_model.declared_variable_count()
        == manual_model.declared_variable_count()
    )

    assert (
        automatic_model.constraint_count()
        == manual_model.constraint_count()
    )


def test_add_round_is_idempotent() -> None:
    """Agregar dos veces una ronda no duplica el modelo."""
    config = ExperimentConfig(
        z=8,
        rounds=1,
        solver="cbc",
        verbose=False,
    )

    model = KeccakMILPModel(config)
    model.add_round(0)

    variables_before = (
        model.declared_variable_count()
    )

    constraints_before = (
        model.constraint_count()
    )

    model.add_round(0)

    assert (
        model.declared_variable_count()
        == variables_before
    )

    assert (
        model.constraint_count()
        == constraints_before
    )


@pytest.mark.parametrize(
    ("z", "rounds"),
    [
        (4, 2),
        (8, 2),
        (8, 3),
    ],
)
def test_add_all_rounds_has_expected_size(
    z: int,
    rounds: int,
) -> None:
    """El modelo completo presenta el tamaño esperado."""
    config = ExperimentConfig(
        z=z,
        rounds=rounds,
        solver="cbc",
        verbose=False,
    )

    model = KeccakMILPModel(config)
    model.add_all_rounds()

    expected_variables = (
        25 * z
        + 195 * z * rounds
    )

    expected_constraints = (
        185 * z * rounds
    )

    assert (
        model.declared_variable_count()
        == expected_variables
    )

    assert (
        model.constraint_count()
        == expected_constraints
    )


@pytest.mark.parametrize(
    ("z", "rounds", "seed"),
    [
        (4, 2, 2026),
        (8, 2, 640),
    ],
)
def test_multi_round_milp_matches_reference(
    z: int,
    rounds: int,
    seed: int,
) -> None:
    """El MILP de varias rondas coincide con la referencia."""
    config = ExperimentConfig(
        z=z,
        rounds=rounds,
        solver="cbc",
        verbose=False,
    )

    model = KeccakMILPModel(config)
    model.add_all_rounds()

    rng = np.random.default_rng(seed)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    for x in range(5):
        for y in range(5):
            for k in range(z):
                variable = model.state_variable(
                    round_index=0,
                    x=x,
                    y=y,
                    k=k,
                )

                model.problem += (
                    variable == int(state[x, y, k]),
                    f"fix_multi_input_{x}_{y}_{k}",
                )

    model.set_smoke_test_objective()

    status = model.solve()

    assert status == "Optimal"

    obtained = np.asarray(
        model.iota_output_values(
            rounds - 1
        ),
        dtype=np.int64,
    )

    expected = keccak_rounds(
        state,
        number_of_rounds=rounds,
    )

    assert np.array_equal(
        obtained,
        expected,
    )