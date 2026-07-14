# ============================================================
# PRUEBAS DE LA CAPA IOTA DE REFERENCIA
# ============================================================

import numpy as np
import pytest

from keccak_milp.layers import (
    ROUND_CONSTANTS_64,
    iota,
    round_constant,
)


@pytest.mark.parametrize(
    ("round_index", "z", "expected"),
    [
        (0, 4, 0x1),
        (0, 8, 0x01),
        (1, 4, 0x2),
        (1, 8, 0x82),
        (2, 4, 0xA),
        (2, 8, 0x8A),
        (3, 4, 0x0),
        (3, 8, 0x00),
        (9, 8, 0x88),
        (23, 8, 0x08),
    ],
)
def test_round_constant_is_truncated(
    round_index: int,
    z: int,
    expected: int,
) -> None:
    """La constante se trunca a los z bits inferiores."""
    assert (
        round_constant(round_index, z)
        == expected
    )


def test_round_constants_table_has_24_entries() -> None:
    """La tabla contiene las 24 constantes oficiales."""
    assert len(ROUND_CONSTANTS_64) == 24


@pytest.mark.parametrize(
    "round_index",
    [-1, 24, 25],
)
def test_round_constant_rejects_invalid_round(
    round_index: int,
) -> None:
    """No se admiten índices fuera de la tabla."""
    with pytest.raises(
        ValueError,
        match="índice de ronda",
    ):
        round_constant(round_index, z=8)


@pytest.mark.parametrize(
    "z",
    [0, 65],
)
def test_round_constant_rejects_invalid_z(
    z: int,
) -> None:
    """La longitud del lane debe estar entre 1 y 64."""
    with pytest.raises(
        ValueError,
        match="longitud del lane",
    ):
        round_constant(0, z)


def test_iota_round_zero_toggles_first_bit() -> None:
    """RC[0] activa únicamente el bit cero."""
    state = np.zeros(
        (5, 5, 8),
        dtype=np.int64,
    )

    output = iota(
        state,
        round_index=0,
    )

    expected = state.copy()
    expected[0, 0, 0] = 1

    assert np.array_equal(
        output,
        expected,
    )


def test_iota_round_one_for_z8() -> None:
    """
    Para z=8:

        RC[1] = 0x82 = 10000010₂

    Se modifican los bits k=1 y k=7.
    """
    state = np.zeros(
        (5, 5, 8),
        dtype=np.int64,
    )

    output = iota(
        state,
        round_index=1,
    )

    expected = state.copy()
    expected[0, 0, 1] = 1
    expected[0, 0, 7] = 1

    assert np.array_equal(
        output,
        expected,
    )


def test_iota_round_one_is_truncated_for_z4() -> None:
    """
    Para z=4, RC[1] se trunca:

        0x82 -> 0x2
    """
    state = np.zeros(
        (5, 5, 4),
        dtype=np.int64,
    )

    output = iota(
        state,
        round_index=1,
    )

    expected = state.copy()
    expected[0, 0, 1] = 1

    assert np.array_equal(
        output,
        expected,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_iota_only_changes_lane_zero_zero(
    z: int,
) -> None:
    """Los otros 24 lanes permanecen sin cambios."""
    rng = np.random.default_rng(2026)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    output = iota(
        state,
        round_index=2,
    )

    for x in range(5):
        for y in range(5):
            if (x, y) == (0, 0):
                continue

            assert np.array_equal(
                output[x, y, :],
                state[x, y, :],
            )


def test_iota_matches_explicit_xor() -> None:
    """El lane (0,0) coincide con un XOR bit a bit."""
    z = 8
    round_index = 2

    rng = np.random.default_rng(640)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    output = iota(
        state,
        round_index,
    )

    constant = round_constant(
        round_index,
        z,
    )

    expected_lane = state[0, 0, :].copy()

    for k in range(z):
        expected_lane[k] ^= (
            constant >> k
        ) & 1

    assert np.array_equal(
        output[0, 0, :],
        expected_lane,
    )


def test_iota_does_not_modify_input() -> None:
    """La implementación devuelve una copia independiente."""
    rng = np.random.default_rng(7)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, 8),
        dtype=np.int64,
    )

    original = state.copy()

    _ = iota(
        state,
        round_index=1,
    )

    assert np.array_equal(
        state,
        original,
    )


def test_iota_rejects_non_binary_state() -> None:
    """La entrada debe contener solamente cero y uno."""
    state = np.zeros(
        (5, 5, 8),
        dtype=np.int64,
    )

    state[1, 2, 3] = 2

    with pytest.raises(
        ValueError,
        match="estado binario",
    ):
        iota(
            state,
            round_index=0,
        )


def test_iota_rejects_non_numpy_state() -> None:
    """La entrada debe ser un arreglo NumPy."""
    state = [
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
        iota(
            state,  # type: ignore[arg-type]
            round_index=0,
        )