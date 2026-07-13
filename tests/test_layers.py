"""
Pruebas de las transformaciones rho y pi.
"""

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from keccak_milp.layers import (
    create_labeled_state,
    is_permutation,
    pi,
    pi_destination,
    rho,
    rho_offset,
    rho_pi,
    rho_pi_destination,
    validate_state_shape,
)


# ============================================================
# PRUEBAS DEL ESTADO
# ============================================================

@pytest.mark.parametrize(
    ("z", "expected_bits"),
    [
        (4, 100),
        (8, 200),
    ],
)
def test_create_labeled_state(
    z: int,
    expected_bits: int,
) -> None:
    state = create_labeled_state(z)

    assert state.shape == (5, 5, z)
    assert state.size == expected_bits
    assert len(np.unique(state)) == expected_bits


def test_invalid_state_shape() -> None:
    invalid_state = np.zeros((4, 5, 4), dtype=int)

    with pytest.raises(ValueError):
        validate_state_shape(invalid_state)


# ============================================================
# PRUEBAS DE RHO
# ============================================================

def test_rho_offset_reduced_modulo_z() -> None:
    # El desplazamiento oficial para (x=0, y=1) es 36.
    assert rho_offset(0, 1, 4) == 0
    assert rho_offset(0, 1, 8) == 4

    # El desplazamiento oficial para (x=1, y=0) es 1.
    assert rho_offset(1, 0, 4) == 1
    assert rho_offset(1, 0, 8) == 1


def test_rho_rotates_expected_lane() -> None:
    state = np.zeros((5, 5, 4), dtype=int)

    state[1, 0, :] = np.array([0, 1, 2, 3])

    transformed = rho(state)

    # El lane (1,0) se desplaza una posición.
    expected = np.array([3, 0, 1, 2])

    assert np.array_equal(
        transformed[1, 0, :],
        expected,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_rho_is_permutation(z: int) -> None:
    state = create_labeled_state(z)
    transformed = rho(state)

    assert transformed.shape == state.shape
    assert is_permutation(state, transformed)


# ============================================================
# PRUEBAS DE PI
# ============================================================

def test_pi_destination() -> None:
    # B[y, 2x + 3y mod 5] = A[x,y]

    assert pi_destination(0, 0) == (0, 0)
    assert pi_destination(1, 0) == (0, 2)
    assert pi_destination(0, 1) == (1, 3)
    assert pi_destination(4, 4) == (4, 0)


@pytest.mark.parametrize("z", [4, 8])
def test_pi_is_permutation(z: int) -> None:
    state = create_labeled_state(z)
    transformed = pi(state)

    assert transformed.shape == state.shape
    assert is_permutation(state, transformed)


def test_pi_moves_complete_lane() -> None:
    state = create_labeled_state(4)

    transformed = pi(state)

    x_destination, y_destination = pi_destination(1, 0)

    assert np.array_equal(
        transformed[x_destination, y_destination, :],
        state[1, 0, :],
    )


# ============================================================
# PRUEBAS DE RHO + PI
# ============================================================

@pytest.mark.parametrize("z", [4, 8])
def test_rho_pi_is_permutation(z: int) -> None:
    state = create_labeled_state(z)
    transformed = rho_pi(state)

    assert transformed.shape == state.shape
    assert is_permutation(state, transformed)


def test_rho_pi_coordinate_mapping() -> None:
    z = 4

    state = create_labeled_state(z)
    transformed = rho_pi(state)

    x = 1
    y = 0
    k = 2

    destination = rho_pi_destination(
        x=x,
        y=y,
        k=k,
        z=z,
    )

    assert transformed[destination] == state[x, y, k]