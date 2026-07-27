# ============================================================
# PRUEBAS DE LA FORMULACIÓN MILP DE IOTA
# ============================================================

import numpy as np
import pytest

from keccak_milp.config import ExperimentConfig
from keccak_milp.layers import iota
from keccak_milp.model import KeccakMILPModel


def build_model_until_chi(
    z: int = 8,
    rounds: int = 1,
) -> KeccakMILPModel:
    """Construye un modelo hasta la salida de chi."""
    config = ExperimentConfig(
        z=z,
        rounds=rounds,
        solver="cbc",
        verbose=False,
    )

    model = KeccakMILPModel(config)

    model.add_theta_layer(0)
    model.add_rho_pi_layers(0)
    model.add_chi_layer(0)

    return model


def fix_chi_state(
    model: KeccakMILPModel,
    state: np.ndarray,
    round_index: int = 0,
) -> None:
    """Fija completamente la entrada de iota."""
    z = model.config.z

    assert state.shape == (5, 5, z)

    for x in range(5):
        for y in range(5):
            for k in range(z):
                variable = model.chi_output_variable(
                    round_index=round_index,
                    x=x,
                    y=y,
                    k=k,
                )

                model.problem += (
                    variable == int(state[x, y, k]),
                    (
                        f"fix_chi_r{round_index}"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )


def test_iota_requires_chi() -> None:
    """Iota no puede agregarse antes de chi."""
    config = ExperimentConfig(
        z=8,
        rounds=1,
        solver="cbc",
        verbose=False,
    )

    model = KeccakMILPModel(config)

    with pytest.raises(
        RuntimeError,
        match="chi antes de iota",
    ):
        model.add_iota_layer(0)


@pytest.mark.parametrize(
    "invalid_round",
    [-1, 1, 2],
)
def test_iota_rejects_invalid_round(
    invalid_round: int,
) -> None:
    """La ronda debe pertenecer al rango configurado."""
    model = build_model_until_chi(
        z=8,
        rounds=1,
    )

    with pytest.raises(
        ValueError,
        match="ronda iota",
    ):
        model.add_iota_layer(
            invalid_round
        )


@pytest.mark.parametrize("z", [4, 8])
def test_iota_adds_no_new_variables(
    z: int,
) -> None:
    """Iota reutiliza el siguiente estado de frontera."""
    model = build_model_until_chi(z=z)

    declared_before = (
        model.declared_variable_count()
    )

    model.add_iota_layer(0)

    declared_after = (
        model.declared_variable_count()
    )

    assert declared_after == declared_before


@pytest.mark.parametrize("z", [4, 8])
def test_iota_adds_one_constraint_per_bit(
    z: int,
) -> None:
    """Iota agrega exactamente 25 × z restricciones."""
    model = build_model_until_chi(z=z)

    constraints_before = (
        model.constraint_count()
    )

    model.add_iota_layer(0)

    constraints_after = (
        model.constraint_count()
    )

    assert (
        constraints_after - constraints_before
        == 25 * z
    )


def test_iota_is_idempotent() -> None:
    """Agregar Iota dos veces no duplica restricciones."""
    model = build_model_until_chi(z=8)

    model.add_iota_layer(0)

    variables_before = (
        model.declared_variable_count()
    )

    constraints_before = (
        model.constraint_count()
    )

    model.add_iota_layer(0)

    assert (
        model.declared_variable_count()
        == variables_before
    )

    assert (
        model.constraint_count()
        == constraints_before
    )


def test_iota_output_variable_requires_layer() -> None:
    """No se puede acceder a una salida aún no conectada."""
    model = build_model_until_chi(z=8)

    with pytest.raises(
        KeyError,
        match="add_iota_layer",
    ):
        model.iota_output_variable(
            round_index=0,
            x=0,
            y=0,
            k=0,
        )


def test_iota_output_is_next_boundary_state() -> None:
    """La salida de Iota reutiliza state[r + 1]."""
    model = build_model_until_chi(z=8)
    model.add_iota_layer(0)

    iota_variable = (
        model.iota_output_variable(
            round_index=0,
            x=2,
            y=3,
            k=4,
        )
    )

    next_state_variable = (
        model.state_variable(
            round_index=1,
            x=2,
            y=3,
            k=4,
        )
    )

    assert iota_variable is next_state_variable


def test_iota_output_values_require_solution() -> None:
    """Los valores solo se recuperan después de resolver."""
    model = build_model_until_chi(z=8)
    model.add_iota_layer(0)

    with pytest.raises(
        RuntimeError,
        match="resolverse",
    ):
        model.iota_output_values(0)


@pytest.mark.parametrize(
    ("z", "round_index"),
    [
        (4, 0),
        (4, 1),
        (8, 0),
        (8, 1),
        (8, 2),
    ],
)
def test_iota_milp_matches_reference_for_zero_state(
    z: int,
    round_index: int,
) -> None:
    """Iota MILP coincide con la referencia sobre cero."""
    rounds = round_index + 1

    config = ExperimentConfig(
        z=z,
        rounds=rounds,
        solver="cbc",
        verbose=False,
    )

    model = KeccakMILPModel(config)

    model.add_theta_layer(round_index)
    model.add_rho_pi_layers(round_index)
    model.add_chi_layer(round_index)
    model.add_iota_layer(round_index)

    state = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    fix_chi_state(
        model,
        state,
        round_index=round_index,
    )

    model.set_smoke_test_objective()

    status = model.solve()

    assert status == "Optimal"

    obtained = np.asarray(
        model.iota_output_values(
            round_index
        ),
        dtype=np.int64,
    )

    expected = iota(
        state,
        round_index=round_index,
    )

    assert np.array_equal(
        obtained,
        expected,
    )


@pytest.mark.parametrize(
    ("seed", "z", "round_index"),
    [
        (7, 4, 0),
        (2026, 4, 1),
        (640, 8, 0),
        (225, 8, 1),
        (701, 8, 2),
    ],
)
def test_iota_milp_matches_reference_for_random_state(
    seed: int,
    z: int,
    round_index: int,
) -> None:
    """Iota MILP coincide con la referencia."""
    rounds = round_index + 1

    config = ExperimentConfig(
        z=z,
        rounds=rounds,
        solver="cbc",
        verbose=False,
    )

    model = KeccakMILPModel(config)

    model.add_theta_layer(round_index)
    model.add_rho_pi_layers(round_index)
    model.add_chi_layer(round_index)
    model.add_iota_layer(round_index)

    rng = np.random.default_rng(seed)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    fix_chi_state(
        model,
        state,
        round_index=round_index,
    )

    model.set_smoke_test_objective()

    status = model.solve()

    assert status == "Optimal"

    obtained = np.asarray(
        model.iota_output_values(
            round_index
        ),
        dtype=np.int64,
    )

    expected = iota(
        state,
        round_index=round_index,
    )

    assert np.array_equal(
        obtained,
        expected,
    )