"""Pruebas de integración de la ronda final de Keccak reducido."""

from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

import keccak_milp.final_case as final_case
from keccak_milp.diffusion import (
    linear_layer_final_case,
)
from keccak_milp.final_case import (
    dynamic_parameter_constant,
    iota_final_case,
    keccak_round_final_case,
    keccak_rounds_final_case,
    security_level_to_rounds,
)
from keccak_milp.layers import round_constant
from keccak_milp.nonlinear import chi_star


EXPECTED_DIGESTS = {
    4: {
        (0, 0): "9bf1b431efb613c9",
        (0, 1): "4b2a921af00cdec3",
        (0, 2): "9cb6f5e131f8915c",
        (0, 3): "c15db64691c8b435",
        (1, 0): "3977a560b41ce00f",
        (1, 1): "8fde3804b413e26b",
        (1, 2): "617eebef028584ff",
        (1, 3): "c7b77a439173eef6",
        (2, 0): "e904926038839405",
        (2, 1): "2aa43f23242ea12b",
        (2, 2): "e8f6b6c754339918",
        (2, 3): "84f4c2d4090f1302",
    },
    8: {
        (0, 0): "04252e25c18d207a",
        (0, 1): "5bc8f5aea9fece17",
        (0, 2): "5edf2e2eb6910371",
        (0, 3): "f916d5af985ed42f",
        (1, 0): "a0d4a5b9d6646878",
        (1, 1): "fea39c92cd13b549",
        (1, 2): "50cafde8d1c619f6",
        (1, 3): "cdf652e4a1f2f1e2",
        (2, 0): "b27cfeddb0434130",
        (2, 1): "2bd1d27d562b5254",
        (2, 2): "eba2d686fa89ea2e",
        (2, 3): "7e53ada19db95613",
    },
}


def manual_round(
    state: np.ndarray,
    *,
    round_index: int,
    security_level: int,
    domain_id: int,
) -> np.ndarray:
    """Composición independiente utilizada como oráculo."""

    after_linear = linear_layer_final_case(
        state
    )

    after_nonlinear = chi_star(
        after_linear
    )

    return iota_final_case(
        after_nonlinear,
        round_index=round_index,
        security_level=security_level,
        domain_id=domain_id,
    )


def manual_rounds(
    state: np.ndarray,
    *,
    security_level: int,
    domain_id: int,
) -> np.ndarray:
    """Repite manualmente la composición de referencia."""

    current = state.astype(
        np.int64,
        copy=True,
    )

    for round_index in range(
        security_level_to_rounds(
            security_level
        )
    ):
        current = manual_round(
            current,
            round_index=round_index,
            security_level=security_level,
            domain_id=domain_id,
        )

    return current


def lane_to_integer(
    state: np.ndarray,
    x: int,
    y: int,
) -> int:
    return sum(
        int(state[x, y, k]) << k
        for k in range(state.shape[2])
    )


def state_digest(
    state: np.ndarray,
) -> str:
    packed = np.packbits(
        state.astype(np.uint8).reshape(-1),
        bitorder="little",
    )

    return hashlib.sha256(
        packed.tobytes()
    ).hexdigest()[:16]


def test_round_functions_are_public() -> None:
    assert (
        "keccak_round_final_case"
        in final_case.__all__
    )

    assert (
        "keccak_rounds_final_case"
        in final_case.__all__
    )

    assert len(final_case.__all__) == len(
        set(final_case.__all__)
    )

    for name in final_case.__all__:
        assert hasattr(final_case, name)


def test_round_function_signatures() -> None:
    assert tuple(
        inspect.signature(
            keccak_round_final_case
        ).parameters
    ) == (
        "state",
        "round_index",
        "security_level",
        "domain_id",
    )

    assert tuple(
        inspect.signature(
            keccak_rounds_final_case
        ).parameters
    ) == (
        "state",
        "security_level",
        "domain_id",
    )


@pytest.mark.parametrize("z", [4, 8])
def test_single_round_matches_manual_composition(
    z: int,
) -> None:
    rng = np.random.default_rng(12000 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    for security_level in (0, 1, 2):
        for domain_id in (0, 1, 2, 3):
            for round_index in range(
                security_level_to_rounds(
                    security_level
                )
            ):
                expected = manual_round(
                    state,
                    round_index=round_index,
                    security_level=security_level,
                    domain_id=domain_id,
                )

                obtained = keccak_round_final_case(
                    state,
                    round_index=round_index,
                    security_level=security_level,
                    domain_id=domain_id,
                )

                assert np.array_equal(
                    obtained,
                    expected,
                )


@pytest.mark.parametrize("z", [4, 8])
def test_dynamic_rounds_match_manual_repetition(
    z: int,
) -> None:
    rng = np.random.default_rng(13000 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    for security_level in (0, 1, 2):
        for domain_id in (0, 1, 2, 3):
            expected = manual_rounds(
                state,
                security_level=security_level,
                domain_id=domain_id,
            )

            obtained = keccak_rounds_final_case(
                state,
                security_level=security_level,
                domain_id=domain_id,
            )

            assert np.array_equal(
                obtained,
                expected,
            )


@pytest.mark.parametrize("z", [4, 8])
def test_known_multiround_digests(
    z: int,
) -> None:
    rng = np.random.default_rng(10000 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    for security_level in (0, 1, 2):
        for domain_id in (0, 1, 2, 3):
            output = keccak_rounds_final_case(
                state,
                security_level=security_level,
                domain_id=domain_id,
            )

            assert state_digest(output) == (
                EXPECTED_DIGESTS[z][
                    (
                        security_level,
                        domain_id,
                    )
                ]
            )


@pytest.mark.parametrize("z", [4, 8])
def test_zero_state_first_round_constants(
    z: int,
) -> None:
    state = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    for security_level in (0, 1, 2):
        for domain_id in (0, 1, 2, 3):
            output = keccak_round_final_case(
                state,
                round_index=0,
                security_level=security_level,
                domain_id=domain_id,
            )

            assert lane_to_integer(
                output,
                0,
                0,
            ) == round_constant(0, z)

            assert lane_to_integer(
                output,
                1,
                0,
            ) == dynamic_parameter_constant(
                round_index=0,
                security_level=security_level,
                domain_id=domain_id,
                z=z,
            )

            outside = output.copy()
            outside[0, 0, :] = 0
            outside[1, 0, :] = 0

            assert np.count_nonzero(outside) == 0


@pytest.mark.parametrize("z", [4, 8])
def test_single_round_does_not_modify_input(
    z: int,
) -> None:
    rng = np.random.default_rng(14000 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    original = state.copy()

    _ = keccak_round_final_case(
        state,
        round_index=0,
        security_level=2,
        domain_id=3,
    )

    assert np.array_equal(
        state,
        original,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_multiround_does_not_modify_input(
    z: int,
) -> None:
    rng = np.random.default_rng(15000 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    original = state.copy()

    _ = keccak_rounds_final_case(
        state,
        security_level=2,
        domain_id=3,
    )

    assert np.array_equal(
        state,
        original,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_single_round_is_deterministic(
    z: int,
) -> None:
    rng = np.random.default_rng(16000 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    first = keccak_round_final_case(
        state,
        round_index=0,
        security_level=1,
        domain_id=2,
    )

    second = keccak_round_final_case(
        state,
        round_index=0,
        security_level=1,
        domain_id=2,
    )

    assert np.array_equal(
        first,
        second,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_multiround_is_deterministic(
    z: int,
) -> None:
    rng = np.random.default_rng(17000 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    first = keccak_rounds_final_case(
        state,
        security_level=2,
        domain_id=1,
    )

    second = keccak_rounds_final_case(
        state,
        security_level=2,
        domain_id=1,
    )

    assert np.array_equal(
        first,
        second,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_single_round_separates_domains(
    z: int,
) -> None:
    rng = np.random.default_rng(18000 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    for security_level in (0, 1, 2):
        outputs = {
            keccak_round_final_case(
                state,
                round_index=0,
                security_level=security_level,
                domain_id=domain_id,
            ).tobytes()
            for domain_id in (0, 1, 2, 3)
        }

        assert len(outputs) == 4


@pytest.mark.parametrize("z", [4, 8])
def test_single_round_separates_security_levels(
    z: int,
) -> None:
    rng = np.random.default_rng(19000 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    for domain_id in (0, 1, 2, 3):
        outputs = {
            keccak_round_final_case(
                state,
                round_index=0,
                security_level=security_level,
                domain_id=domain_id,
            ).tobytes()
            for security_level in (0, 1, 2)
        }

        assert len(outputs) == 3


@pytest.mark.parametrize("z", [4, 8])
def test_full_permutation_separates_domains(
    z: int,
) -> None:
    rng = np.random.default_rng(20000 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    for security_level in (0, 1, 2):
        outputs = {
            keccak_rounds_final_case(
                state,
                security_level=security_level,
                domain_id=domain_id,
            ).tobytes()
            for domain_id in (0, 1, 2, 3)
        }

        assert len(outputs) == 4


@pytest.mark.parametrize("z", [4, 8])
def test_full_permutation_separates_security_levels(
    z: int,
) -> None:
    rng = np.random.default_rng(21000 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    for domain_id in (0, 1, 2, 3):
        outputs = {
            keccak_rounds_final_case(
                state,
                security_level=security_level,
                domain_id=domain_id,
            ).tobytes()
            for security_level in (0, 1, 2)
        }

        assert len(outputs) == 3


@pytest.mark.parametrize("z", [4, 8])
def test_output_properties(
    z: int,
) -> None:
    rng = np.random.default_rng(22000 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    for security_level in (0, 1, 2):
        for domain_id in (0, 1, 2, 3):
            output = keccak_rounds_final_case(
                state,
                security_level=security_level,
                domain_id=domain_id,
            )

            assert output.shape == state.shape
            assert output.dtype == np.int64
            assert np.all(
                np.isin(output, [0, 1])
            )


@pytest.mark.parametrize(
    "invalid_round_index",
    [
        True,
        1.0,
        "0",
        None,
    ],
)
def test_single_round_rejects_non_integer_index(
    invalid_round_index: object,
) -> None:
    state = np.zeros(
        (5, 5, 4),
        dtype=np.int64,
    )

    with pytest.raises(TypeError):
        keccak_round_final_case(
            state,
            round_index=invalid_round_index,
            security_level=2,
            domain_id=0,
        )


@pytest.mark.parametrize(
    (
        "round_index",
        "security_level",
    ),
    [
        (-1, 0),
        (1, 0),
        (2, 1),
        (3, 2),
        (24, 2),
    ],
)
def test_single_round_rejects_index_outside_level(
    round_index: int,
    security_level: int,
) -> None:
    state = np.zeros(
        (5, 5, 4),
        dtype=np.int64,
    )

    with pytest.raises(ValueError):
        keccak_round_final_case(
            state,
            round_index=round_index,
            security_level=security_level,
            domain_id=0,
        )


@pytest.mark.parametrize(
    "invalid_security_level",
    [
        True,
        1.0,
        "1",
        None,
    ],
)
def test_round_functions_reject_non_integer_security_level(
    invalid_security_level: object,
) -> None:
    state = np.zeros(
        (5, 5, 4),
        dtype=np.int64,
    )

    with pytest.raises(TypeError):
        keccak_round_final_case(
            state,
            round_index=0,
            security_level=invalid_security_level,
            domain_id=0,
        )

    with pytest.raises(TypeError):
        keccak_rounds_final_case(
            state,
            security_level=invalid_security_level,
            domain_id=0,
        )


@pytest.mark.parametrize(
    "invalid_security_level",
    [
        -1,
        3,
        100,
    ],
)
def test_round_functions_reject_security_level_range(
    invalid_security_level: int,
) -> None:
    state = np.zeros(
        (5, 5, 4),
        dtype=np.int64,
    )

    with pytest.raises(ValueError):
        keccak_round_final_case(
            state,
            round_index=0,
            security_level=invalid_security_level,
            domain_id=0,
        )

    with pytest.raises(ValueError):
        keccak_rounds_final_case(
            state,
            security_level=invalid_security_level,
            domain_id=0,
        )


@pytest.mark.parametrize(
    "invalid_domain_id",
    [
        True,
        1.0,
        "1",
        None,
    ],
)
def test_round_functions_reject_non_integer_domain(
    invalid_domain_id: object,
) -> None:
    state = np.zeros(
        (5, 5, 4),
        dtype=np.int64,
    )

    with pytest.raises(TypeError):
        keccak_round_final_case(
            state,
            round_index=0,
            security_level=0,
            domain_id=invalid_domain_id,
        )

    with pytest.raises(TypeError):
        keccak_rounds_final_case(
            state,
            security_level=0,
            domain_id=invalid_domain_id,
        )


@pytest.mark.parametrize(
    "invalid_domain_id",
    [
        -1,
        4,
        100,
    ],
)
def test_round_functions_reject_domain_range(
    invalid_domain_id: int,
) -> None:
    state = np.zeros(
        (5, 5, 4),
        dtype=np.int64,
    )

    with pytest.raises(ValueError):
        keccak_round_final_case(
            state,
            round_index=0,
            security_level=0,
            domain_id=invalid_domain_id,
        )

    with pytest.raises(ValueError):
        keccak_rounds_final_case(
            state,
            security_level=0,
            domain_id=invalid_domain_id,
        )


def test_round_functions_reject_non_numpy_state() -> None:
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
        keccak_round_final_case(
            state,  # type: ignore[arg-type]
            round_index=0,
            security_level=0,
            domain_id=0,
        )

    with pytest.raises(
        TypeError,
        match="arreglo NumPy",
    ):
        keccak_rounds_final_case(
            state,  # type: ignore[arg-type]
            security_level=0,
            domain_id=0,
        )


def test_round_functions_reject_non_binary_state() -> None:
    state = np.zeros(
        (5, 5, 4),
        dtype=np.int64,
    )

    state[2, 3, 1] = 2

    with pytest.raises(
        ValueError,
        match="estado binario",
    ):
        keccak_round_final_case(
            state,
            round_index=0,
            security_level=0,
            domain_id=0,
        )

    with pytest.raises(
        ValueError,
        match="estado binario",
    ):
        keccak_rounds_final_case(
            state,
            security_level=0,
            domain_id=0,
        )


def test_round_functions_reject_unsupported_z() -> None:
    state = np.zeros(
        (5, 5, 5),
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="4 u 8",
    ):
        keccak_round_final_case(
            state,
            round_index=0,
            security_level=0,
            domain_id=0,
        )

    with pytest.raises(
        ValueError,
        match="4 u 8",
    ):
        keccak_rounds_final_case(
            state,
            security_level=0,
            domain_id=0,
        )
