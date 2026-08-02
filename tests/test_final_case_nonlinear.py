"""Pruebas de la capa no lineal Chi* del caso final."""

from __future__ import annotations

import numpy as np
import pytest

import keccak_milp.nonlinear as nonlinear
from keccak_milp.nonlinear import (
    CHI_STAR_INVERSE_TABLE,
    CHI_STAR_TABLE,
    chi_star,
    chi_star_cost,
    chi_star_inverse,
    chi_star_inverse_sbox,
    chi_star_sbox,
)


EXPECTED_TABLE = (
    0,
    5,
    10,
    19,
    8,
    21,
    23,
    22,
    28,
    31,
    4,
    27,
    16,
    11,
    29,
    26,
    2,
    15,
    24,
    9,
    1,
    20,
    14,
    7,
    25,
    18,
    17,
    6,
    30,
    13,
    3,
    12,
)


def value_to_bits(
    value: int,
) -> tuple[int, int, int, int, int]:
    return tuple(
        (value >> index) & 1
        for index in range(5)
    )


def bits_to_value(
    bits: np.ndarray,
) -> int:
    return sum(
        int(bits[index]) << index
        for index in range(5)
    )


def parity(
    value: int,
) -> int:
    return value.bit_count() & 1


def build_ddt(
    table: tuple[int, ...],
) -> np.ndarray:
    ddt = np.zeros(
        (32, 32),
        dtype=np.int64,
    )

    for input_difference in range(32):
        for value in range(32):
            output_difference = (
                table[value]
                ^ table[
                    value ^ input_difference
                ]
            )

            ddt[
                input_difference,
                output_difference,
            ] += 1

    return ddt


def build_lat(
    table: tuple[int, ...],
) -> np.ndarray:
    lat = np.zeros(
        (32, 32),
        dtype=np.int64,
    )

    for input_mask in range(32):
        for output_mask in range(32):
            total = 0

            for value in range(32):
                equal_parity = (
                    parity(input_mask & value)
                    == parity(
                        output_mask & table[value]
                    )
                )

                total += (
                    1
                    if equal_parity
                    else -1
                )

            lat[
                input_mask,
                output_mask,
            ] = total

    return lat


def anf_coefficients(
    truth_values: list[int],
) -> list[int]:
    coefficients = list(truth_values)

    for variable in range(5):
        variable_mask = 1 << variable

        for monomial in range(32):
            if monomial & variable_mask:
                coefficients[monomial] ^= (
                    coefficients[
                        monomial ^ variable_mask
                    ]
                )

    return coefficients


def coordinate_nonlinearity(
    truth_values: list[int],
) -> int:
    maximum_walsh = 0

    for input_mask in range(32):
        walsh = sum(
            1
            if (
                parity(input_mask & value)
                == truth_values[value]
            )
            else -1
            for value in range(32)
        )

        maximum_walsh = max(
            maximum_walsh,
            abs(walsh),
        )

    return 16 - maximum_walsh // 2


def test_chi_star_table_is_expected() -> None:
    assert CHI_STAR_TABLE == EXPECTED_TABLE
    assert len(CHI_STAR_TABLE) == 32
    assert set(CHI_STAR_TABLE) == set(range(32))


def test_chi_star_sbox_and_inverse_are_exhaustive() -> None:
    assert len(CHI_STAR_INVERSE_TABLE) == 32
    assert set(CHI_STAR_INVERSE_TABLE) == set(range(32))

    for value in range(32):
        output = chi_star_sbox(value)

        assert output == EXPECTED_TABLE[value]

        assert (
            chi_star_inverse_sbox(output)
            == value
        )

        assert (
            chi_star_sbox(
                chi_star_inverse_sbox(value)
            )
            == value
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        True,
        1.0,
        "1",
        None,
    ],
)
def test_chi_star_sbox_rejects_non_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="value",
    ):
        chi_star_sbox(invalid_value)

    with pytest.raises(
        TypeError,
        match="value",
    ):
        chi_star_inverse_sbox(invalid_value)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -1,
        32,
        100,
    ],
)
def test_chi_star_sbox_rejects_out_of_range(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="entre 0 y 31",
    ):
        chi_star_sbox(invalid_value)

    with pytest.raises(
        ValueError,
        match="entre 0 y 31",
    ):
        chi_star_inverse_sbox(invalid_value)


@pytest.mark.parametrize("z", [4, 8])
def test_chi_star_preserves_shape_and_binary_values(
    z: int,
) -> None:
    rng = np.random.default_rng(500 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    output = chi_star(state)

    assert output.shape == state.shape
    assert output.dtype == np.int64
    assert np.all(np.isin(output, [0, 1]))


@pytest.mark.parametrize("z", [4, 8])
def test_chi_star_matches_table_on_every_slice(
    z: int,
) -> None:
    for value in range(32):
        bits = value_to_bits(value)

        state = np.empty(
            (5, 5, z),
            dtype=np.int64,
        )

        for x in range(5):
            state[x, :, :] = bits[x]

        output = chi_star(state)
        expected = CHI_STAR_TABLE[value]

        for y in range(5):
            for k in range(z):
                assert (
                    bits_to_value(
                        output[:, y, k]
                    )
                    == expected
                )


@pytest.mark.parametrize("z", [4, 8])
def test_chi_star_inverse_round_trip(
    z: int,
) -> None:
    rng = np.random.default_rng(600 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    forward = chi_star(state)
    recovered = chi_star_inverse(forward)

    assert np.array_equal(
        recovered,
        state,
    )

    inverse_first = chi_star_inverse(state)
    recovered_second = chi_star(inverse_first)

    assert np.array_equal(
        recovered_second,
        state,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_chi_star_does_not_modify_input(
    z: int,
) -> None:
    rng = np.random.default_rng(700 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    original = state.copy()

    _ = chi_star(state)
    _ = chi_star_inverse(state)

    assert np.array_equal(
        state,
        original,
    )


def test_chi_star_rejects_non_numpy_state() -> None:
    state = [
        [
            [0 for _ in range(4)]
            for _ in range(5)
        ]
        for _ in range(5)
    ]

    with pytest.raises(
        TypeError,
        match="arreglo NumPy",
    ):
        chi_star(
            state  # type: ignore[arg-type]
        )


def test_chi_star_rejects_non_binary_state() -> None:
    state = np.zeros(
        (5, 5, 4),
        dtype=np.int64,
    )

    state[2, 3, 1] = 2

    with pytest.raises(
        ValueError,
        match="estado binario",
    ):
        chi_star(state)

    with pytest.raises(
        ValueError,
        match="estado binario",
    ):
        chi_star_inverse(state)


def test_chi_star_rejects_unsupported_z() -> None:
    state = np.zeros(
        (5, 5, 5),
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="4 u 8",
    ):
        chi_star(state)

    with pytest.raises(
        ValueError,
        match="4 u 8",
    ):
        chi_star_inverse(state)


@pytest.mark.parametrize("z", [4, 8])
def test_chi_star_slices_are_independent(
    z: int,
) -> None:
    state = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    first_value = 6
    second_value = 25

    first_bits = value_to_bits(first_value)
    second_bits = value_to_bits(second_value)

    for x in range(5):
        state[x, 1, 0] = first_bits[x]
        state[x, 3, z - 1] = second_bits[x]

    output = chi_star(state)

    assert (
        bits_to_value(output[:, 1, 0])
        == CHI_STAR_TABLE[first_value]
    )

    assert (
        bits_to_value(
            output[:, 3, z - 1]
        )
        == CHI_STAR_TABLE[second_value]
    )

    assert (
        bits_to_value(output[:, 0, 0])
        == 0
    )


@pytest.mark.parametrize(
    (
        "z",
        "expected_sboxes",
        "expected_and",
        "expected_xor",
    ),
    [
        (4, 20, 200, 380),
        (8, 40, 400, 760),
    ],
)
def test_chi_star_cost(
    z: int,
    expected_sboxes: int,
    expected_and: int,
    expected_xor: int,
) -> None:
    cost = chi_star_cost(z)

    assert cost.z == z
    assert cost.sboxes_per_round == expected_sboxes

    assert cost.and_per_sbox == 10
    assert cost.xor_per_sbox == 19
    assert cost.not_per_sbox == 0
    assert cost.logical_depth == 4

    assert cost.total_and == expected_and
    assert cost.total_xor == expected_xor
    assert cost.total_not == 0


def test_chi_star_cryptographic_metrics() -> None:
    table = CHI_STAR_TABLE

    ddt = build_ddt(table)
    lat = build_lat(table)

    differential_uniformity = int(
        ddt[1:, :].max()
    )

    maximum_absolute_lat = int(
        np.abs(lat[1:, 1:]).max()
    )

    coordinate_degrees: list[int] = []
    nonlinearities: list[int] = []

    for output_bit in range(5):
        truth_values = [
            (table[value] >> output_bit) & 1
            for value in range(32)
        ]

        coefficients = anf_coefficients(
            truth_values
        )

        active_monomials = [
            monomial
            for monomial, coefficient
            in enumerate(coefficients)
            if coefficient
        ]

        coordinate_degrees.append(
            max(
                monomial.bit_count()
                for monomial in active_monomials
            )
        )

        nonlinearities.append(
            coordinate_nonlinearity(
                truth_values
            )
        )

    fixed_points = tuple(
        value
        for value, output in enumerate(table)
        if value == output
    )

    assert differential_uniformity == 2
    assert maximum_absolute_lat == 8

    assert coordinate_degrees == [
        2,
        2,
        2,
        2,
        2,
    ]

    assert nonlinearities == [
        12,
        12,
        12,
        12,
        12,
    ]

    assert fixed_points == (0,)


def test_nonlinear_public_interface() -> None:
    required = {
        "CHI_STAR_INVERSE_TABLE",
        "CHI_STAR_TABLE",
        "NonlinearCost",
        "chi_star",
        "chi_star_cost",
        "chi_star_inverse",
        "chi_star_inverse_sbox",
        "chi_star_sbox",
    }

    assert required == set(nonlinear.__all__)

    assert len(nonlinear.__all__) == len(
        set(nonlinear.__all__)
    )

    for name in nonlinear.__all__:
        assert hasattr(nonlinear, name)
