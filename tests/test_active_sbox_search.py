from __future__ import annotations

import numpy as np
import pytest

from keccak_milp.active_sbox_search import (
    RestrictedThreeRoundSearchResult,
    active_sbox_positions,
    enumerate_two_plus_two_trails,
    integer_to_five_bits,
    search_three_round_two_plus_two,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, [0, 0, 0, 0, 0]),
        (1, [1, 0, 0, 0, 0]),
        (2, [0, 1, 0, 0, 0]),
        (11, [1, 1, 0, 1, 0]),
        (31, [1, 1, 1, 1, 1]),
    ],
)
def test_integer_to_five_bits(
    value: int,
    expected: list[int],
) -> None:
    bits = integer_to_five_bits(
        value
    )

    np.testing.assert_array_equal(
        bits,
        np.asarray(
            expected,
            dtype=np.int64,
        ),
    )

    assert bits.shape == (5,)
    assert bits.dtype == np.int64


@pytest.mark.parametrize(
    "invalid_value",
    [
        -1,
        32,
        100,
    ],
)
def test_integer_to_five_bits_rejects_invalid_values(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="entre 0 y 31",
    ):
        integer_to_five_bits(
            invalid_value
        )


def test_active_sbox_positions_detects_expected_support() -> None:
    state = np.zeros(
        (5, 5, 4),
        dtype=np.int64,
    )

    state[
        :,
        1,
        2,
    ] = np.asarray(
        [1, 0, 1, 0, 0],
        dtype=np.int64,
    )

    state[
        0,
        3,
        0,
    ] = 1

    assert active_sbox_positions(
        state
    ) == (
        (1, 2),
        (3, 0),
    )


def test_active_sbox_positions_returns_empty_support() -> None:
    state = np.zeros(
        (5, 5, 8),
        dtype=np.int64,
    )

    assert active_sbox_positions(
        state
    ) == ()


def test_active_sbox_positions_rejects_invalid_shape() -> None:
    invalid_state = np.zeros(
        (5, 4, 4),
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="forma",
    ):
        active_sbox_positions(
            invalid_state
        )


def test_active_sbox_positions_rejects_nonbinary_state() -> None:
    invalid_state = np.zeros(
        (5, 5, 4),
        dtype=np.int64,
    )

    invalid_state[
        0,
        0,
        0,
    ] = 2

    with pytest.raises(
        ValueError,
        match="binario",
    ):
        active_sbox_positions(
            invalid_state
        )


@pytest.mark.parametrize(
    "invalid_z",
    [
        0,
        1,
        2,
        16,
    ],
)
def test_enumerate_two_plus_two_trails_rejects_invalid_z(
    invalid_z: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="z=4 o z=8",
    ):
        enumerate_two_plus_two_trails(
            invalid_z
        )


@pytest.mark.parametrize(
    "invalid_z",
    [
        0,
        1,
        2,
        16,
    ],
)
def test_three_round_search_rejects_invalid_z(
    invalid_z: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="z=4 o z=8",
    ):
        search_three_round_two_plus_two(
            invalid_z
        )


def test_restricted_search_result_is_json_serializable() -> None:
    result = RestrictedThreeRoundSearchResult(
        z=4,
        trail_count=200,
        realizations_per_trail=1024,
        evaluated_realizations=204_800,
        minimum_round_2_activity=9,
        minimum_total_activity=13,
        best_candidate_count=512,
        support_round_0=(
            (2, 0),
            (4, 0),
        ),
        beta_round_0=(
            1,
            1,
        ),
        support_round_1=(
            (1, 3),
            (2, 2),
        ),
        left_values_round_1=(
            2,
            8,
        ),
        support_round_2=(
            (0, 0),
            (0, 2),
            (1, 1),
            (2, 0),
            (2, 1),
            (3, 2),
            (4, 1),
            (4, 2),
            (4, 3),
        ),
        delta_b2_hamming_weight=12,
        histogram_round_2={
            9: 512,
            10: 2048,
        },
    )

    serialized = result.to_dict()

    assert serialized["z"] == 4
    assert serialized["minimum_total_activity"] == 13

    assert serialized["support_round_0"] == [
        [2, 0],
        [4, 0],
    ]

    assert serialized["beta_round_0"] == [
        1,
        1,
    ]

    assert serialized["histogram_round_2"] == {
        "9": 512,
        "10": 2048,
    }
