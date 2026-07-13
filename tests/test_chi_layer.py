# ============================================================
# PRUEBAS DE LA CAPA CHI DE REFERENCIA
# ============================================================

import numpy as np
import pytest
from numpy.typing import NDArray

from keccak_milp.layers import chi


def build_zero_state(
    z: int,
) -> NDArray[np.int64]:
    """Construye un estado binario nulo."""
    return np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )


@pytest.mark.parametrize("z", [1, 4, 8])
def test_chi_preserves_state_shape(
    z: int,
) -> None:
    """Chi conserva la forma 5 × 5 × z."""
    state = build_zero_state(z)

    output = chi(state)

    assert output.shape == (5, 5, z)
    assert output.dtype == np.int64


def test_chi_zero_state_remains_zero() -> None:
    """El estado nulo es un punto fijo de chi."""
    state = build_zero_state(z=8)

    output = chi(state)

    assert np.array_equal(output, state)
    assert int(output.sum()) == 0


def test_chi_all_one_state_remains_one() -> None:
    """
    Si todos los bits son uno, el término no lineal es cero:

        NOT 1 AND 1 = 0
    """
    state = np.ones(
        (5, 5, 8),
        dtype=np.int64,
    )

    output = chi(state)

    assert np.array_equal(output, state)
    assert int(output.sum()) == 25 * 8


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
def test_chi_matches_local_truth_table(
    a: int,
    b: int,
    c: int,
) -> None:
    """
    Comprueba exhaustivamente:

        d = a XOR ((NOT b) AND c)
    """
    state = build_zero_state(z=1)

    state[0, 0, 0] = a
    state[1, 0, 0] = b
    state[2, 0, 0] = c

    output = chi(state)

    expected = a ^ ((1 - b) & c)

    assert int(output[0, 0, 0]) == expected


def test_chi_does_not_modify_input() -> None:
    """La función debe crear una salida independiente."""
    rng = np.random.default_rng(2026)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, 8),
        dtype=np.int64,
    )

    original = state.copy()

    _ = chi(state)

    assert np.array_equal(state, original)


def test_chi_rejects_non_numpy_state() -> None:
    """La entrada debe ser un arreglo NumPy."""
    invalid_state = [
        [
            [0 for _ in range(8)]
            for _ in range(5)
        ]
        for _ in range(5)
    ]

    with pytest.raises(
        TypeError,
        match="arreglo NumPy",
    ):
        chi(invalid_state)  # type: ignore[arg-type]


def test_chi_rejects_non_binary_state() -> None:
    """Chi no admite valores distintos de cero y uno."""
    state = build_zero_state(z=8)
    state[2, 3, 4] = 2

    with pytest.raises(
        ValueError,
        match="estado binario",
    ):
        chi(state)