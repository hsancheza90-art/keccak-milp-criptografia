# ============================================================
# PRUEBAS DE LA FORMULACIÓN MILP DE CHI
# ============================================================

import numpy as np
import pytest

from keccak_milp.config import ExperimentConfig
from keccak_milp.layers import chi
from keccak_milp.model import KeccakMILPModel


def build_model_until_rho_pi(
    z: int = 8,
) -> KeccakMILPModel:
    """Construye un modelo hasta la salida de rho y pi."""
    config = ExperimentConfig(
        z=z,
        rounds=1,
        solver="cbc",
        verbose=False,
    )

    model = KeccakMILPModel(config)
    model.add_theta_layer(0)
    model.add_rho_pi_layers(0)

    return model


def fix_rho_pi_state(
    model: KeccakMILPModel,
    state: np.ndarray,
) -> None:
    """Fija completamente la entrada de chi."""
    z = model.config.z

    assert state.shape == (5, 5, z)

    for x in range(5):
        for y in range(5):
            for k in range(z):
                variable = model.rho_pi_output_variable(
                    round_index=0,
                    x=x,
                    y=y,
                    k=k,
                )

                model.problem += (
                    variable == int(state[x, y, k]),
                    f"fix_rho_pi_{x}_{y}_{k}",
                )


def test_chi_requires_rho_pi() -> None:
    """Chi no puede agregarse antes de rho y pi."""
    config = ExperimentConfig(
        z=8,
        rounds=1,
        solver="cbc",
        verbose=False,
    )

    model = KeccakMILPModel(config)

    with pytest.raises(
        RuntimeError,
        match="rho y pi",
    ):
        model.add_chi_layer(0)


def test_chi_creates_expected_variables() -> None:
    """Chi crea tres variables binarias por bit."""
    z = 8
    model = build_model_until_rho_pi(z)

    declared_before = model.declared_variable_count()

    model.add_chi_layer(0)

    declared_after = model.declared_variable_count()

    assert len(model.chi_and) == 25 * z
    assert len(model.chi_output) == 25 * z
    assert len(model.chi_q) == 25 * z

    assert (
        declared_after - declared_before
        == 3 * 25 * z
    )


def test_chi_creates_expected_constraints() -> None:
    """Chi crea cuatro restricciones por bit."""
    z = 8
    model = build_model_until_rho_pi(z)

    constraints_before = model.constraint_count()

    model.add_chi_layer(0)

    constraints_after = model.constraint_count()

    assert (
        constraints_after - constraints_before
        == 4 * 25 * z
    )


def test_chi_is_idempotent() -> None:
    """Agregar Chi dos veces no duplica el modelo."""
    model = build_model_until_rho_pi(z=8)

    model.add_chi_layer(0)

    variables_before = model.declared_variable_count()
    constraints_before = model.constraint_count()

    model.add_chi_layer(0)

    assert (
        model.declared_variable_count()
        == variables_before
    )

    assert (
        model.constraint_count()
        == constraints_before
    )


def test_chi_output_variable_requires_layer() -> None:
    """No se puede acceder a una salida inexistente."""
    model = build_model_until_rho_pi(z=8)

    with pytest.raises(
        KeyError,
        match="add_chi_layer",
    ):
        model.chi_output_variable(
            round_index=0,
            x=0,
            y=0,
            k=0,
        )


def test_chi_output_values_require_solution() -> None:
    """Los valores solo pueden recuperarse después de resolver."""
    model = build_model_until_rho_pi(z=8)
    model.add_chi_layer(0)

    with pytest.raises(
        RuntimeError,
        match="resolverse",
    ):
        model.chi_output_values(0)


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ],
)
def test_chi_milp_matches_truth_table(
    a: int,
    b: int,
    c: int,
) -> None:
    """
    La formulación MILP reproduce la tabla local de verdad.

    Se utiliza z=4 porque es el menor tamaño de palabra admitido
    por ExperimentConfig. La combinación se evalúa en k=0.
    """
    z = 4

    state = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    # Variables locales de la expresión:
    #
    # d = a XOR ((NOT b) AND c)
    state[0, 0, 0] = a
    state[1, 0, 0] = b
    state[2, 0, 0] = c

    model = build_model_until_rho_pi(z)
    model.add_chi_layer(0)

    fix_rho_pi_state(model, state)

    # Registrar el objetivo requerido por KeccakMILPModel.solve().
    model.set_smoke_test_objective()

    status = model.solve()

    assert status == "Optimal"

    output = np.asarray(
        model.chi_output_values(0),
        dtype=np.int64,
    )

    expected = chi(state)

    # Comparación completa del estado.
    assert np.array_equal(output, expected)

    # Comprobación explícita de la posición asociada a a, b y c.
    expected_local_bit = a ^ ((1 - b) & c)

    assert int(output[0, 0, 0]) == expected_local_bit

@pytest.mark.parametrize("seed", [7, 2026, 640])
def test_chi_milp_matches_reference_for_random_state(
    seed: int,
) -> None:
    """Chi MILP coincide con la referencia en estados aleatorios."""
    z = 8
    rng = np.random.default_rng(seed)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    model = build_model_until_rho_pi(z)
    model.add_chi_layer(0)

    fix_rho_pi_state(model, state)
    model.set_smoke_test_objective()

    status = model.solve()

    assert status == "Optimal"

    obtained = np.asarray(
        model.chi_output_values(0),
        dtype=np.int64,
    )

    expected = chi(state)

    assert np.array_equal(obtained, expected)