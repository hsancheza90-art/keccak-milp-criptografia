"""Pruebas de la separación dinámica del caso final."""

import numpy as np
import pytest

from keccak_milp.final_case import (
    dynamic_parameter_constant,
    iota_final_case,
)
from keccak_milp.layers import iota


@pytest.mark.parametrize(
    (
        "round_index",
        "security_level",
        "domain_id",
        "z",
        "expected",
    ),
    [
        (0, 0, 0, 4, 4),
        (1, 0, 0, 4, 8),
        (2, 0, 0, 4, 1),
        (1, 1, 2, 4, 5),
        (2, 2, 3, 8, 60),
    ],
)
def test_dynamic_parameter_constant_expected_values(
    round_index: int,
    security_level: int,
    domain_id: int,
    z: int,
    expected: int,
) -> None:
    assert (
        dynamic_parameter_constant(
            round_index,
            security_level,
            domain_id,
            z,
        )
        == expected
    )


@pytest.mark.parametrize("z", [4, 8])
@pytest.mark.parametrize("round_index", [0, 1, 2])
def test_dynamic_parameter_constant_is_unique(
    z: int,
    round_index: int,
) -> None:
    constants = {
        dynamic_parameter_constant(
            round_index,
            security_level,
            domain_id,
            z,
        )
        for security_level in range(3)
        for domain_id in range(4)
    }

    assert len(constants) == 12


@pytest.mark.parametrize("z", [4, 8])
@pytest.mark.parametrize("round_index", [0, 1, 2])
def test_iota_final_case_preserves_standard_iota_lane(
    z: int,
    round_index: int,
) -> None:
    rng = np.random.default_rng(
        1000 + 10 * z + round_index
    )

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5,z),
        dtype=np.int64,
    )

    standard = iota(
        state,
        round_index=round_index,
    )

    modified = iota_final_case(
        state,
        round_index=round_index,
        security_level=1,
        domain_id=2,
    )

    assert np.array_equal(
        modified[0, 0, :],
        standard[0, 0, :],
    )


@pytest.mark.parametrize("z", [4, 8])
def test_iota_final_case_changes_only_designated_lanes(
    z: int,
) -> None:
    round_index = 1
    security_level = 1
    domain_id = 2

    rng = np.random.default_rng(225 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    expected = iota(
        state,
        round_index=round_index,
    )

    constant = dynamic_parameter_constant(
        round_index,
        security_level,
        domain_id,
        z,
    )

    for k in range(z):
        expected[1, 0, k] ^= (
            constant >> k
        ) & 1

    obtained = iota_final_case(
        state,
        round_index=round_index,
        security_level=security_level,
        domain_id=domain_id,
    )

    assert np.array_equal(
        obtained,
        expected,
    )

    for x in range(5):
        for y in range(5):
            if (x, y) in {
                (0, 0),
                (1, 0),
            }:
                continue

            assert np.array_equal(
                obtained[x, y, :],
                state[x, y, :],
            )


@pytest.mark.parametrize("z", [4, 8])
@pytest.mark.parametrize("round_index", [0, 1, 2])
def test_iota_final_case_outputs_are_unique(
    z: int,
    round_index: int,
) -> None:
    state = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    outputs = {
        iota_final_case(
            state,
            round_index=round_index,
            security_level=security_level,
            domain_id=domain_id,
        ).tobytes()
        for security_level in range(3)
        for domain_id in range(4)
    }

    assert len(outputs) == 12


def test_iota_final_case_does_not_modify_input() -> None:
    rng = np.random.default_rng(640)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, 8),
        dtype=np.int64,
    )

    original = state.copy()

    _ = iota_final_case(
        state,
        round_index=2,
        security_level=2,
        domain_id=3,
    )

    assert np.array_equal(
        state,
        original,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_iota_final_case_is_involution(
    z: int,
) -> None:
    rng = np.random.default_rng(701 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    once = iota_final_case(
        state,
        round_index=2,
        security_level=2,
        domain_id=1,
    )

    twice = iota_final_case(
        once,
        round_index=2,
        security_level=2,
        domain_id=1,
    )

    assert np.array_equal(
        twice,
        state,
    )


@pytest.mark.parametrize(
    "invalid_round",
    [
        -1,
        24,
    ],
)
def test_dynamic_parameter_constant_rejects_invalid_round(
    invalid_round: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="índice de ronda",
    ):
        dynamic_parameter_constant(
            invalid_round,
            security_level=0,
            domain_id=0,
            z=4,
        )


@pytest.mark.parametrize(
    "invalid_round",
    [
        True,
        1.0,
        "1",
        None,
    ],
)
def test_dynamic_parameter_constant_rejects_invalid_type(
    invalid_round: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="round_index",
    ):
        dynamic_parameter_constant(
            invalid_round,
            security_level=0,
            domain_id=0,
            z=4,
        )


def test_iota_final_case_rejects_non_binary_state() -> None:
    state = np.zeros(
        (5, 5, 4),
        dtype=np.int64,
    )

    state[2, 3, 1] = 2

    with pytest.raises(
        ValueError,
        match="estado binario",
    ):
        iota_final_case(
            state,
            round_index=0,
            security_level=0,
            domain_id=0,
        )


def test_iota_final_case_rejects_unsupported_z() -> None:
    state = np.zeros(
        (5, 5, 5),
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="4 u 8",
    ):
        iota_final_case(
            state,
            round_index=0,
            security_level=0,
            domain_id=0,
        )
