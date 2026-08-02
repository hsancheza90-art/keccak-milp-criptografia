"""Pruebas de los parámetros dinámicos del caso final."""

import pytest

from keccak_milp.final_case import (
    encode_dynamic_parameters,
    security_level_to_rounds,
    validate_domain_id,
)


@pytest.mark.parametrize(
    ("security_level", "expected_rounds"),
    [
        (0, 1),
        (1, 2),
        (2, 3),
    ],
)
def test_security_level_to_rounds(
    security_level: int,
    expected_rounds: int,
) -> None:
    assert (
        security_level_to_rounds(security_level)
        == expected_rounds
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
def test_security_level_rejects_non_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError):
        security_level_to_rounds(invalid_value)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -1,
        3,
        10,
    ],
)
def test_security_level_rejects_out_of_range(
    invalid_value: int,
) -> None:
    with pytest.raises(ValueError):
        security_level_to_rounds(invalid_value)


@pytest.mark.parametrize(
    "domain_id",
    [
        0,
        1,
        2,
        3,
    ],
)
def test_validate_domain_id_accepts_supported_values(
    domain_id: int,
) -> None:
    assert validate_domain_id(domain_id) == domain_id


@pytest.mark.parametrize(
    "invalid_value",
    [
        False,
        1.0,
        "1",
        None,
    ],
)
def test_domain_id_rejects_non_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError):
        validate_domain_id(invalid_value)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -1,
        4,
        10,
    ],
)
def test_domain_id_rejects_out_of_range(
    invalid_value: int,
) -> None:
    with pytest.raises(ValueError):
        validate_domain_id(invalid_value)


@pytest.mark.parametrize(
    (
        "security_level",
        "domain_id",
        "z",
        "expected",
    ),
    [
        (0, 0, 4, 4),
        (1, 2, 4, 10),
        (2, 3, 8, 15),
    ],
)
def test_dynamic_encoding_expected_values(
    security_level: int,
    domain_id: int,
    z: int,
    expected: int,
) -> None:
    assert (
        encode_dynamic_parameters(
            security_level,
            domain_id,
            z,
        )
        == expected
    )


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_dynamic_encoding_is_unique_and_nonzero(
    z: int,
) -> None:
    encoded_values = {
        encode_dynamic_parameters(
            security_level,
            domain_id,
            z,
        )
        for security_level in range(3)
        for domain_id in range(4)
    }

    assert len(encoded_values) == 12
    assert min(encoded_values) > 0


@pytest.mark.parametrize(
    "z",
    [
        4,
        8,
    ],
)
def test_dynamic_encoding_fits_in_lane(
    z: int,
) -> None:
    for security_level in range(3):
        for domain_id in range(4):
            encoded = encode_dynamic_parameters(
                security_level,
                domain_id,
                z,
            )

            assert encoded < (1 << z)


@pytest.mark.parametrize(
    "invalid_z",
    [
        True,
        4.0,
        "4",
        None,
    ],
)
def test_dynamic_encoding_rejects_non_integer_z(
    invalid_z: object,
) -> None:
    with pytest.raises(TypeError):
        encode_dynamic_parameters(
            security_level=0,
            domain_id=0,
            z=invalid_z,
        )


@pytest.mark.parametrize(
    "invalid_z",
    [
        0,
        1,
        3,
        16,
    ],
)
def test_dynamic_encoding_rejects_unsupported_z(
    invalid_z: int,
) -> None:
    with pytest.raises(ValueError):
        encode_dynamic_parameters(
            security_level=0,
            domain_id=0,
            z=invalid_z,
        )
