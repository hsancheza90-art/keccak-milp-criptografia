"""
Pruebas del esqueleto inicial del modelo MILP.
"""

import pytest

from keccak_milp.config import ExperimentConfig
from keccak_milp.model import KeccakMILPModel


# ============================================================
# CREACIÓN DE VARIABLES
# ============================================================

@pytest.mark.parametrize(
    ("z", "rounds", "expected_variables"),
    [
        (4, 1, 200),
        (4, 2, 300),
        (4, 3, 400),
        (8, 1, 400),
        (8, 2, 600),
        (8, 3, 800),
    ],
)
def test_state_variable_count(
    z: int,
    rounds: int,
    expected_variables: int,
) -> None:
    """
    Comprueba:

        variables = 25 × z × (rounds + 1)
    """

    config = ExperimentConfig(
        z=z,
        rounds=rounds,
        solver="cbc",
        verbose=False,
    )

    model = KeccakMILPModel(config)

    assert len(model.state) == expected_variables
    assert model.declared_variable_count() == expected_variables

        # Antes de agregar objetivo o restricciones, ninguna variable
        # necesariamente aparece todavía en problem.variables().
    assert model.attached_variable_count() == 0


def test_initial_and_final_state_sizes() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=3,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    assert len(model.initial_state_variables()) == 100
    assert len(model.final_state_variables()) == 100


def test_state_variable_access() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    variable = model.state_variable(
        round_index=0,
        x=1,
        y=2,
        k=3,
    )

    assert variable.name == "a_r0_x1_y2_k3"


def test_invalid_state_variable_access() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    with pytest.raises(KeyError):
        model.state_variable(
            round_index=5,
            x=0,
            y=0,
            k=0,
        )


# ============================================================
# CONSTRUCCIÓN DEL MODELO
# ============================================================

def test_skeleton_adds_one_constraint() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    assert model.constraint_count() == 0
    assert not model.has_constraint(
        "entrada_diferencial_no_nula"
    )

    model.build_skeleton()

    assert model.constraint_count() == 1
    assert model.has_constraint(
        "entrada_diferencial_no_nula"
    )


def test_build_skeleton_is_idempotent() -> None:
    """
    Ejecutar dos veces build_skeleton no debe duplicar restricciones.
    """

    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    model.build_skeleton()
    model.build_skeleton()

    assert model.constraint_count() == 1


# ============================================================
# RESOLUCIÓN
# ============================================================

@pytest.mark.parametrize(
    ("z", "rounds"),
    [
        (4, 1),
        (4, 3),
        (8, 1),
    ],
)
def test_skeleton_solves_to_one(
    z: int,
    rounds: int,
) -> None:
    """
    La entrada debe ser no nula y se minimiza su peso.

    Por tanto, el óptimo esperado es 1.
    """

    config = ExperimentConfig(
        z=z,
        rounds=rounds,
        solver="cbc",
        time_limit_seconds=60,
        mip_gap=0.0,
        verbose=False,
    )

    model = KeccakMILPModel(config)
    model.build_skeleton()

    status = model.solve()

    assert status == "Optimal"
    assert model.objective_value() == pytest.approx(1.0)
    assert len(model.active_initial_positions()) == 1


def test_solve_without_objective_raises_error() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    with pytest.raises(RuntimeError):
        model.solve()


# ============================================================
# ESTADÍSTICAS
# ============================================================

def test_model_statistics() -> None:
    config = ExperimentConfig(
        z=8,
        rounds=3,
        verbose=False,
    )

    model = KeccakMILPModel(config)

    # Antes de construir el esqueleto:
    # - existen 800 variables declaradas;
    # - ninguna está todavía conectada a expresiones.
    initial_statistics = model.statistics()

    assert initial_statistics.z == 8
    assert initial_statistics.rounds == 3
    assert initial_statistics.state_bits == 200
    assert initial_statistics.boundary_states == 4
    assert initial_statistics.declared_variables == 800
    assert initial_statistics.attached_variables == 0
    assert initial_statistics.total_constraints == 0

    model.build_skeleton()

    # El objetivo provisional y la restricción solo utilizan
    # las 200 variables del estado inicial.
    built_statistics = model.statistics()

    assert built_statistics.declared_variables == 800
    assert built_statistics.attached_variables == 200
    assert built_statistics.total_constraints == 1