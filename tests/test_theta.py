"""
Pruebas funcionales de la capa theta de Keccak.
"""

import numpy as np
import pytest

from keccak_milp.layers import (
    column_parities,
    create_single_active_bit_state,
    hamming_weight,
    theta,
    theta_effect,
)


# ============================================================
# PARIDADES DE COLUMNA
# ============================================================

@pytest.mark.parametrize("z", [4, 8])
def test_zero_state_has_zero_parities(z: int) -> None:
    state = np.zeros((5, 5, z), dtype=np.int64)

    parities = column_parities(state)

    assert parities.shape == (5, z)
    assert np.all(parities == 0)


def test_single_active_bit_creates_one_column_parity() -> None:
    state = create_single_active_bit_state(
        z=4,
        x=2,
        y=3,
        k=1,
    )

    parities = column_parities(state)

    assert hamming_weight(state) == 1
    assert int(np.sum(parities)) == 1
    assert parities[2, 1] == 1


def test_two_active_bits_same_column_cancel_parity() -> None:
    state = np.zeros((5, 5, 4), dtype=np.int64)

    state[2, 1, 3] = 1
    state[2, 4, 3] = 1

    parities = column_parities(state)

    assert parities[2, 3] == 0
    assert int(np.sum(parities)) == 0


def test_three_active_bits_same_column_have_odd_parity() -> None:
    state = np.zeros((5, 5, 4), dtype=np.int64)

    state[1, 0, 2] = 1
    state[1, 2, 2] = 1
    state[1, 4, 2] = 1

    parities = column_parities(state)

    assert parities[1, 2] == 1
    assert int(np.sum(parities)) == 1


# ============================================================
# MATRIZ D
# ============================================================

def test_theta_effect_for_single_active_bit() -> None:
    z = 4

    state = create_single_active_bit_state(
        z=z,
        x=2,
        y=3,
        k=1,
    )

    effect = theta_effect(state)

    # La paridad activa se encuentra en C[2,1].
    #
    # Esta paridad interviene en:
    #
    # D[3,1] por C[x-1,k]
    # D[1,2] por C[x+1,k-1]
    assert effect[3, 1] == 1
    assert effect[1, 2] == 1

    assert int(np.sum(effect)) == 2


@pytest.mark.parametrize("z", [4, 8])
def test_zero_state_has_zero_theta_effect(z: int) -> None:
    state = np.zeros((5, 5, z), dtype=np.int64)

    effect = theta_effect(state)

    assert effect.shape == (5, z)
    assert np.all(effect == 0)


# ============================================================
# TRANSFORMACIÓN THETA
# ============================================================

@pytest.mark.parametrize("z", [4, 8])
def test_theta_preserves_zero_state(z: int) -> None:
    state = np.zeros((5, 5, z), dtype=np.int64)

    transformed = theta(state)

    assert transformed.shape == state.shape
    assert np.array_equal(transformed, state)


def test_theta_single_active_bit_expected_weight() -> None:
    state = create_single_active_bit_state(
        z=4,
        x=2,
        y=3,
        k=1,
    )

    transformed = theta(state)

    # D tiene dos posiciones activas.
    # Cada D[x,k] afecta los cinco valores de y.
    #
    # La posición inicial no coincide con esas dos columnas afectadas,
    # por lo que el peso final es:
    #
    # 1 + 5 + 5 = 11
    assert hamming_weight(transformed) == 11


def test_theta_output_is_binary() -> None:
    rng = np.random.default_rng(2026)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, 4),
        dtype=np.int64,
    )

    transformed = theta(state)

    assert np.all(np.isin(transformed, [0, 1]))


@pytest.mark.parametrize("z", [4, 8])
def test_theta_is_linear_over_xor(z: int) -> None:
    """
    Comprueba:

        theta(A XOR B) = theta(A) XOR theta(B)
    """

    rng = np.random.default_rng(2026 + z)

    state_a = rng.integers(
        0,
        2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    state_b = rng.integers(
        0,
        2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    left_side = theta(
        np.bitwise_xor(state_a, state_b)
    )

    right_side = np.bitwise_xor(
        theta(state_a),
        theta(state_b),
    )

    assert np.array_equal(
        left_side,
        right_side,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_theta_is_invertible_for_sampled_states(z: int) -> None:
    """
    Una transformación lineal de Keccak debe ser biyectiva.

    Esta prueba no construye todavía la inversa formal. Comprueba que
    una muestra de estados distintos no produzca colisiones.
    """

    rng = np.random.default_rng(4000 + z)

    outputs: set[bytes] = set()

    for _ in range(100):
        state = rng.integers(
            0,
            2,
            size=(5, 5, z),
            dtype=np.int64,
        )

        transformed = theta(state)
        encoded = transformed.astype(np.uint8).tobytes()

        assert encoded not in outputs
        outputs.add(encoded)


# ============================================================
# VALIDACIÓN DE ENTRADAS
# ============================================================

def test_theta_rejects_nonbinary_state() -> None:
    state = np.zeros((5, 5, 4), dtype=np.int64)
    state[0, 0, 0] = 2

    with pytest.raises(ValueError):
        theta(state)