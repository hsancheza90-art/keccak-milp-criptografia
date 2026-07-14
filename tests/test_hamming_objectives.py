# ============================================================
# PRUEBAS DE OBJETIVOS DE PESO DE HAMMING
# ============================================================

import numpy as np
import pytest

from keccak_milp.config import ExperimentConfig
from keccak_milp.model import KeccakMILPModel


def build_model(
    z: int = 4,
    rounds: int = 1,
) -> KeccakMILPModel:
    """Construye un modelo de varias rondas."""
    config = ExperimentConfig(
        z=z,
        rounds=rounds,
        solver="cbc",
        time_limit_seconds=60,
        verbose=False,
    )

    model = KeccakMILPModel(config)
    model.add_all_rounds()

    return model


def boundary_weight(
    model: KeccakMILPModel,
    boundary_index: int,
) -> int:
    """Calcula el peso de un estado de frontera resuelto."""
    total = 0

    for x in range(5):
        for y in range(5):
            for k in range(model.config.z):
                variable = model.state_variable(
                    round_index=boundary_index,
                    x=x,
                    y=y,
                    k=k,
                )

                value = variable.value()

                if value is None:
                    raise RuntimeError(
                        "La variable no tiene valor."
                    )

                total += int(value > 0.5)

    return total


def test_boundary_objective_rejects_non_integer() -> None:
    """El índice del estado debe ser entero."""
    model = build_model()

    with pytest.raises(
        TypeError,
        match="debe ser un entero",
    ):
        model.set_boundary_hamming_weight_objective(
            0.5  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_index",
    [-1, 2, 3],
)
def test_boundary_objective_rejects_invalid_index(
    invalid_index: int,
) -> None:
    """Solo existen los estados 0 y 1 para una ronda."""
    model = build_model(
        z=4,
        rounds=1,
    )

    with pytest.raises(
        ValueError,
        match="estado de frontera",
    ):
        model.set_boundary_hamming_weight_objective(
            invalid_index
        )


def test_objective_value_requires_objective() -> None:
    """No se recupera un objetivo inexistente."""
    model = build_model()

    with pytest.raises(
        RuntimeError,
        match="no tiene una función objetivo",
    ):
        model.objective_value()


def test_objective_value_requires_solution() -> None:
    """El objetivo solo tiene valor después de resolver."""
    model = build_model()

    model.set_boundary_hamming_weight_objective(
        0
    )

    with pytest.raises(
        RuntimeError,
        match="debe resolverse",
    ):
        model.objective_value()


@pytest.mark.parametrize("z", [4, 8])
def test_minimum_nonzero_input_weight_is_one(
    z: int,
) -> None:
    """
    Si se impide la entrada nula y se minimiza HW(A_0),
    el peso mínimo debe ser uno.
    """
    model = build_model(
        z=z,
        rounds=1,
    )

    model.add_nonzero_input_constraint()

    model.set_boundary_hamming_weight_objective(
        boundary_index=0
    )

    status = model.solve()

    assert status == "Optimal"

    input_weight = boundary_weight(
        model,
        boundary_index=0,
    )

    assert input_weight == 1
    assert model.objective_value() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("z", "rounds"),
    [
        (4, 1),
        (8, 1),
    ],
)

def test_final_boundary_objective_matches_solution_weight(
    z: int,
    rounds: int,
) -> None:
    """El valor objetivo coincide con HW(A_R)."""
    model = build_model(
        z=z,
        rounds=rounds,
    )

    model.add_nonzero_input_constraint()

    model.set_boundary_hamming_weight_objective(
        boundary_index=rounds
    )

    status = model.solve()

    assert status == "Optimal"

    final_weight = boundary_weight(
        model,
        boundary_index=rounds,
    )

    assert model.objective_value() == pytest.approx(
        float(final_weight)
    )


@pytest.mark.parametrize(
    ("z", "rounds"),
    [
        (4, 1),
        (8, 1),
    ],
)


def test_input_output_objective_matches_solution(
    z: int,
    rounds: int,
) -> None:
    """
    El valor objetivo coincide con:

        HW(A_0) + HW(A_R)
    """
    model = build_model(
        z=z,
        rounds=rounds,
    )

    model.add_nonzero_input_constraint()

    model.set_input_output_hamming_weight_objective()

    status = model.solve()

    assert status == "Optimal"

    input_weight = boundary_weight(
        model,
        boundary_index=0,
    )

    output_weight = boundary_weight(
        model,
        boundary_index=rounds,
    )

    expected_value = (
        input_weight + output_weight
    )

    assert model.objective_value() == pytest.approx(
        float(expected_value)
    )


def test_hamming_objective_replaces_smoke_objective() -> None:
    """El objetivo real puede reemplazar al provisional."""
    model = build_model(
        z=4,
        rounds=1,
    )

    model.set_smoke_test_objective()

    model.add_nonzero_input_constraint()

    model.set_boundary_hamming_weight_objective(
        boundary_index=0
    )

    status = model.solve()

    assert status == "Optimal"
    assert model.objective_value() == pytest.approx(1.0)


def test_fixed_input_objective_returns_fixed_weight() -> None:
    """Una entrada fijada conserva su peso en el objetivo."""
    z = 4

    model = build_model(
        z=z,
        rounds=1,
    )

    state = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    active_bits = [
        (0, 0, 0),
        (1, 2, 1),
        (4, 4, 3),
    ]

    for x, y, k in active_bits:
        state[x, y, k] = 1

    for x in range(5):
        for y in range(5):
            for k in range(z):
                model.problem += (
                    model.state_variable(
                        0,
                        x,
                        y,
                        k,
                    )
                    == int(state[x, y, k]),
                    f"fix_weight_input_{x}_{y}_{k}",
                )

    model.set_boundary_hamming_weight_objective(
        boundary_index=0
    )

    status = model.solve()

    assert status == "Optimal"

    assert model.objective_value() == pytest.approx(
        float(len(active_bits))
    )

def fix_input_state(
    model: KeccakMILPModel,
    state: np.ndarray,
) -> None:
    """Fija completamente el estado de frontera A_0."""
    expected_shape = (
        5,
        5,
        model.config.z,
    )

    if state.shape != expected_shape:
        raise ValueError(
            "La entrada debe tener forma "
            f"{expected_shape}."
        )

    for x in range(5):
        for y in range(5):
            for k in range(model.config.z):
                model.problem += (
                    model.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(state[x, y, k]),
                    f"fix_objective_input_{x}_{y}_{k}",
                )

def test_two_round_final_objective_with_fixed_input() -> None:
    """
    El objetivo final de dos rondas coincide con HW(A_2)
    cuando la entrada está completamente fijada.
    """
    z = 4
    rounds = 2

    model = build_model(
        z=z,
        rounds=rounds,
    )

    state = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    active_bits = [
        (0, 0, 0),
        (1, 2, 1),
        (3, 4, 2),
    ]

    for x, y, k in active_bits:
        state[x, y, k] = 1

    fix_input_state(
        model,
        state,
    )

    model.set_boundary_hamming_weight_objective(
        boundary_index=rounds
    )

    status = model.solve()

    assert status == "Optimal"

    final_weight = boundary_weight(
        model,
        boundary_index=rounds,
    )

    assert model.objective_value() == pytest.approx(
        float(final_weight)
    )


def test_two_round_input_output_objective_with_fixed_input() -> None:
    """
    El objetivo combinado de dos rondas coincide con:

        HW(A_0) + HW(A_2)

    para una entrada completamente fijada.
    """
    z = 4
    rounds = 2

    model = build_model(
        z=z,
        rounds=rounds,
    )

    state = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    active_bits = [
        (0, 0, 0),
        (1, 2, 1),
        (3, 4, 2),
    ]

    for x, y, k in active_bits:
        state[x, y,k] = 1

    fix_input_state(
        model,
        state,
    )

    model.set_input_output_hamming_weight_objective()

    status = model.solve()

    assert status == "Optimal"

    input_weight = boundary_weight(
        model,
        boundary_index=0,
    )

    final_weight = boundary_weight(
        model,
        boundary_index=rounds,
    )

    assert input_weight == len(active_bits)

    assert model.objective_value() == pytest.approx(
        float(input_weight + final_weight)
    )