"""Pruebas de la capa lineal del caso final."""

from collections.abc import Callable

import numpy as np
import pytest

from keccak_milp.diffusion import (
    build_linear_matrix,
    circulant_mix_x,
    compose_dependency_matrix,
    gf2_rank,
    inverse_circulant_mix_x,
    linear_layer_final_case,
    linear_layer_v1,
    summarize_dependency_matrix,
)


LinearLayer = Callable[
    [np.ndarray],
    np.ndarray,
]


@pytest.fixture(scope="module")
def matrices() -> dict[tuple[str, int], np.ndarray]:
    """Matrices calculadas una sola vez por módulo."""

    return {
        ("v1", z): build_linear_matrix(
            linear_layer_v1,
            z,
        )
        for z in (4, 8)
    } | {
        ("final", z): build_linear_matrix(
            linear_layer_final_case,
            z,
        )
        for z in (4, 8)
    }


@pytest.mark.parametrize("z", [4, 8])
def test_circulant_mix_is_invertible(
    z: int,
) -> None:
    rng = np.random.default_rng(100 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    transformed = circulant_mix_x(state)
    recovered = inverse_circulant_mix_x(
        transformed
    )

    assert np.array_equal(
        recovered,
        state,
    )

    inverse_first = inverse_circulant_mix_x(
        state
    )

    recovered_second = circulant_mix_x(
        inverse_first
    )

    assert np.array_equal(
        recovered_second,
        state,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_circulant_mix_is_linear(
    z: int,
) -> None:
    rng = np.random.default_rng(200 + z)

    left_state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    right_state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    obtained = circulant_mix_x(
        left_state ^ right_state
    )

    expected = (
        circulant_mix_x(left_state)
        ^ circulant_mix_x(right_state)
    )

    assert np.array_equal(
        obtained,
        expected,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_circulant_mix_does_not_modify_input(
    z: int,
) -> None:
    rng = np.random.default_rng(300 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    original = state.copy()

    _ = circulant_mix_x(state)

    assert np.array_equal(
        state,
        original,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_final_layer_matches_explicit_composition(
    z: int,
) -> None:
    rng = np.random.default_rng(400 + z)

    state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    obtained = linear_layer_final_case(state)

    expected = circulant_mix_x(
        linear_layer_v1(state)
    )

    assert np.array_equal(
        obtained,
        expected,
    )


@pytest.mark.parametrize("z", [4, 8])
def test_final_layer_is_linear(
    z: int,
) -> None:
    rng = np.random.default_rng(500 + z)

    left_state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    right_state = rng.integers(
        low=0,
        high=2,
        size=(5, 5, z),
        dtype=np.int64,
    )

    obtained = linear_layer_final_case(
        left_state ^ right_state
    )

    expected = (
        linear_layer_final_case(left_state)
        ^ linear_layer_final_case(right_state)
    )

    assert np.array_equal(
        obtained,
        expected,
    )


@pytest.mark.parametrize(
    ("layer_name", "z"),
    [
        ("v1", 4),
        ("v1", 8),
        ("final", 4),
        ("final", 8),
    ],
)
def test_linear_matrices_are_invertible(
    matrices: dict[tuple[str, int], np.ndarray],
    layer_name: str,
    z: int,
) -> None:
    matrix = matrices[(layer_name, z)]

    assert gf2_rank(matrix) == 25 * z


@pytest.mark.parametrize(
    ("z", "expected_minimum"),
    [
        (4, 62),
        (8, 77),
    ],
)
def test_v1_two_step_structural_baseline(
    matrices: dict[tuple[str, int], np.ndarray],
    z: int,
    expected_minimum: int,
) -> None:
    dependencies = compose_dependency_matrix(
        matrices[("v1", z)],
        applications=2,
        mode="structural",
    )

    weights = np.count_nonzero(
        dependencies,
        axis=0,
    )

    assert int(weights.min()) == expected_minimum


@pytest.mark.parametrize(
    ("z", "expected_minimum"),
    [
        (4, 100),
        (8, 194),
    ],
)
def test_final_layer_reaches_structural_target(
    matrices: dict[tuple[str, int], np.ndarray],
    z: int,
    expected_minimum: int,
) -> None:
    dependencies = compose_dependency_matrix(
        matrices[("final", z)],
        applications=2,
        mode="structural",
    )

    weights = np.count_nonzero(
        dependencies,
        axis=0,
    )

    assert int(weights.min()) == expected_minimum
    assert (
        int(weights.min()) / (25 * z)
        >= 0.80
    )


@pytest.mark.parametrize(
    ("z", "expected_minimum"),
    [
        (4, 25),
        (8, 25),
    ],
)
def test_final_layer_one_step_minimum_dependency(
    matrices: dict[tuple[str, int], np.ndarray],
    z: int,
    expected_minimum: int,
) -> None:
    weights = np.count_nonzero(
        matrices[("final", z)],
        axis=0,
    )

    assert int(weights.min()) == expected_minimum


@pytest.mark.parametrize(
    ("z", "expected_minimum"),
    [
        (4, 100),
        (8, 194),
    ],
)
def test_diffusion_metrics_are_consistent(
    matrices: dict[tuple[str, int], np.ndarray],
    z: int,
    expected_minimum: int,
) -> None:
    dependencies = compose_dependency_matrix(
        matrices[("final", z)],
        applications=2,
        mode="structural",
    )

    metrics = summarize_dependency_matrix(
        dependencies,
        z=z,
        applications=2,
        mode="structural",
    )

    assert metrics.minimum == expected_minimum
    assert metrics.minimum <= metrics.mean
    assert metrics.mean <= metrics.maximum

    assert metrics.minimum_coverage == pytest.approx(
        expected_minimum / (25 * z)
    )

    assert metrics.minimum_coverage >= 0.80


@pytest.mark.parametrize(
    "invalid_z",
    [
        True,
        4.0,
        "4",
        None,
    ],
)
def test_build_linear_matrix_rejects_non_integer_z(
    invalid_z: object,
) -> None:
    with pytest.raises(TypeError):
        build_linear_matrix(
            linear_layer_v1,
            invalid_z,
        )


@pytest.mark.parametrize(
    "invalid_z",
    [
        0,
        3,
        16,
    ],
)
def test_build_linear_matrix_rejects_unsupported_z(
    invalid_z: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="4 u 8",
    ):
        build_linear_matrix(
            linear_layer_v1,
            invalid_z,
        )


@pytest.mark.parametrize(
    "invalid_applications",
    [
        True,
        1.0,
        "2",
    ],
)
def test_dependency_composition_rejects_invalid_type(
    matrices: dict[tuple[str, int], np.ndarray],
    invalid_applications: object,
) -> None:
    with pytest.raises(TypeError):
        compose_dependency_matrix(
            matrices[("v1", 4)],
            applications=invalid_applications,
            mode="structural",
        )


@pytest.mark.parametrize(
    "invalid_applications",
    [
        0,
        -1,
    ],
)
def test_dependency_composition_rejects_invalid_range(
    matrices: dict[tuple[str, int], np.ndarray],
    invalid_applications: int,
) -> None:
    with pytest.raises(ValueError):
        compose_dependency_matrix(
            matrices[("v1", 4)],
            applications=invalid_applications,
            mode="structural",
        )
