"""
Capa lineal y métricas de difusión del caso final de Keccak reducido.

La línea base utiliza:

    L_V1 = pi o rho o theta

La variante propuesta añade una mezcla circulante invertible:

    L* = mu o pi o rho o theta

donde:

    mu(B)[x, y, k] =
        B[x, y, k]
        XOR B[x - 1, y, k]
        XOR B[x - 2, y, k]

Todos los índices del eje x se calculan módulo 5.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from keccak_milp.layers import (
    rho_pi,
    theta,
    validate_state_shape,
)


DiffusionMode = Literal[
    "structural",
    "algebraic",
]

LinearLayer = Callable[
    [NDArray[np.integer]],
    NDArray[np.integer],
]


@dataclass(frozen=True)
class DiffusionMetrics:
    """Resumen de dependencias para una capa lineal."""

    z: int
    mode: DiffusionMode
    applications: int
    minimum: int
    mean: float
    maximum: int
    minimum_coverage: float
    mean_coverage: float
    maximum_coverage: float


def _validate_plain_integer(
    value: object,
    parameter_name: str,
) -> int:
    """Valida un entero y rechaza explícitamente booleanos."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"El parámetro {parameter_name} debe ser un entero."
        )

    return value


def _validate_word_size(
    z: object,
) -> int:
    """Valida los tamaños de palabra del caso final."""

    validated = _validate_plain_integer(
        z,
        "z",
    )

    if validated not in {4, 8}:
        raise ValueError(
            "El tamaño de palabra z debe ser 4 u 8. "
            f"Se recibió z={validated}."
        )

    return validated


def _validate_binary_state(
    state: NDArray[np.integer],
) -> int:
    """Valida forma, tamaño de palabra y binariedad."""

    z = validate_state_shape(state)

    if z not in {4, 8}:
        raise ValueError(
            "El tamaño de palabra z debe ser 4 u 8. "
            f"Se recibió z={z}."
        )

    if not np.all(np.isin(state, [0, 1])):
        raise ValueError(
            "La capa lineal requiere un estado binario."
        )

    return z


def linear_layer_v1(
    state: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    Aplica la capa lineal de la V1.

    El orden es:

        theta -> rho -> pi
    """

    _validate_binary_state(state)

    output = theta(
        state.astype(
            np.int64,
            copy=True,
        )
    )

    output = rho_pi(output)

    return output.astype(
        np.int64,
        copy=False,
    )


def circulant_mix_x(
    state: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    Mezcla cada lane con dos vecinos del eje x.

    Formalmente:

        output[x, y, k] =
            state[x, y, k]
            XOR state[x - 1, y, k]
            XOR state[x - 2, y, k]

    Los índices se calculan módulo 5.
    """

    _validate_binary_state(state)

    source = state.astype(
        np.int64,
        copy=False,
    )

    output = np.empty_like(
        source,
        dtype=np.int64,
    )

    for x in range(5):
        output[x, :, :] = (
            source[x, :, :]
            ^ source[(x - 1) % 5, :, :]
            ^ source[(x - 2) % 5, :, :]
        )

    return output


def inverse_circulant_mix_x(
    state: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    Invierte la mezcla circulante del eje x.

    La inversa corresponde al polinomio:

        t + t^2 + t^4

    módulo:

        t^5 + 1

    Por tanto:

        output[x, y, k] =
            state[x - 1, y, k]
            XOR state[x - 2, y, k]
            XOR state[x - 4, y, k]
    """

    _validate_binary_state(state)

    source = state.astype(
        np.int64,
        copy=False,
    )

    output = np.empty_like(
        source,
        dtype=np.int64,
    )

    for x in range(5):
        output[x, :, :] = (
            source[(x - 1) % 5, :, :]
            ^ source[(x - 2) % 5, :, :]
            ^ source[(x - 4) % 5, :, :]
        )

    return output


def linear_layer_final_case(
    state: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    Aplica la capa lineal propuesta para el caso final.

    El orden es:

        theta -> rho -> pi -> mu
    """

    transformed = linear_layer_v1(state)

    return circulant_mix_x(transformed)


def build_linear_matrix(
    layer: LinearLayer,
    z: object,
) -> NDArray[np.uint8]:
    """
    Construye la matriz binaria de una capa lineal.

    Cada columna se obtiene aplicando la capa a un vector de la
    base canónica del espacio de estados.
    """

    validated_z = _validate_word_size(z)
    state_bits = 25 * validated_z

    matrix = np.zeros(
        (state_bits, state_bits),
        dtype=np.uint8,
    )

    for input_index in range(state_bits):
        state = np.zeros(
            (5, 5, validated_z),
            dtype=np.int64,
        )

        state.reshape(-1)[input_index] = 1

        output = layer(state)

        if output.shape != state.shape:
            raise RuntimeError(
                "La capa lineal devolvió una forma inesperada."
            )

        if not np.all(np.isin(output, [0, 1])):
            raise RuntimeError(
                "La capa lineal devolvió valores no binarios."
            )

        matrix[:, input_index] = (
            output.reshape(-1).astype(np.uint8)
        )

    return matrix


def gf2_rank(
    matrix: NDArray[np.integer],
) -> int:
    """Calcula el rango de una matriz binaria sobre GF(2)."""

    if not isinstance(matrix, np.ndarray):
        raise TypeError(
            "La matriz debe ser un arreglo NumPy."
        )

    if matrix.ndim != 2:
        raise ValueError(
            "La matriz debe tener dos dimensiones."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "La matriz debe ser cuadrada."
        )

    if not np.all(np.isin(matrix, [0, 1])):
        raise ValueError(
            "La matriz debe ser binaria."
        )

    reduced = (
        matrix.astype(
            np.uint8,
            copy=True,
        )
        & 1
    )

    row_count, column_count = reduced.shape
    pivot_row = 0

    for column in range(column_count):
        candidates = np.flatnonzero(
            reduced[pivot_row:, column]
        )

        if candidates.size == 0:
            continue

        selected_row = (
            pivot_row + int(candidates[0])
        )

        if selected_row != pivot_row:
            reduced[
                [pivot_row, selected_row],
                :
            ] = reduced[
                [selected_row, pivot_row],
                :
            ]

        rows_to_eliminate = np.flatnonzero(
            reduced[:, column]
        )

        rows_to_eliminate = rows_to_eliminate[
            rows_to_eliminate != pivot_row
        ]

        if rows_to_eliminate.size > 0:
            reduced[rows_to_eliminate, :] ^= (
                reduced[pivot_row, :]
            )

        pivot_row += 1

        if pivot_row == row_count:
            break

    return pivot_row


def compose_dependency_matrix(
    one_step_matrix: NDArray[np.integer],
    applications: object,
    mode: DiffusionMode,
) -> NDArray[np.generic]:
    """
    Compone una matriz lineal varias veces.

    En modo ``structural`` se usa el semianillo booleano: una
    dependencia existe cuando hay al menos un camino.

    En modo ``algebraic`` se opera sobre GF(2), por lo que se
    consideran las cancelaciones XOR.
    """

    if not isinstance(one_step_matrix, np.ndarray):
        raise TypeError(
            "La matriz debe ser un arreglo NumPy."
        )

    if one_step_matrix.ndim != 2:
        raise ValueError(
            "La matriz debe tener dos dimensiones."
        )

    if (
        one_step_matrix.shape[0]
        != one_step_matrix.shape[1]
    ):
        raise ValueError(
            "La matriz debe ser cuadrada."
        )

    if not np.all(
        np.isin(one_step_matrix, [0, 1])
    ):
        raise ValueError(
            "La matriz debe ser binaria."
        )

    validated_applications = _validate_plain_integer(
        applications,
        "applications",
    )

    if validated_applications <= 0:
        raise ValueError(
            "El número de aplicaciones debe ser mayor que cero."
        )

    if mode not in {
        "structural",
        "algebraic",
    }:
        raise ValueError(
            "El modo debe ser 'structural' o 'algebraic'."
        )

    state_bits = one_step_matrix.shape[0]

    if mode == "structural":
        adjacency = one_step_matrix.astype(
            bool,
            copy=False,
        )

        current = np.eye(
            state_bits,
            dtype=bool,
        )

        for _ in range(validated_applications):
            current = (
                adjacency.astype(
                    np.uint16,
                    copy=False,
                )
                @ current.astype(
                    np.uint16,
                    copy=False,
                )
            ) > 0

        return current

    adjacency = one_step_matrix.astype(
        np.uint8,
        copy=False,
    )

    current = np.eye(
        state_bits,
        dtype=np.uint8,
    )

    for _ in range(validated_applications):
        product = (
            adjacency.astype(
                np.uint16,
                copy=False,
            )
            @ current.astype(
                np.uint16,
                copy=False,
            )
        )

        current = (
            product & 1
        ).astype(np.uint8)

    return current


def summarize_dependency_matrix(
    dependency_matrix: NDArray[np.generic],
    z: object,
    applications: object,
    mode: DiffusionMode,
) -> DiffusionMetrics:
    """Resume los pesos de las columnas de dependencias."""

    validated_z = _validate_word_size(z)

    validated_applications = _validate_plain_integer(
        applications,
        "applications",
    )

    if validated_applications <= 0:
        raise ValueError(
            "El número de aplicaciones debe ser mayor que cero."
        )

    if mode not in {
        "structural",
        "algebraic",
    }:
        raise ValueError(
            "El modo debe ser 'structural' o 'algebraic'."
        )

    state_bits = 25 * validated_z

    if dependency_matrix.shape != (
        state_bits,
        state_bits,
    ):
        raise ValueError(
            "La matriz de dependencias tiene dimensiones "
            "incompatibles con z."
        )

    weights = np.count_nonzero(
        dependency_matrix,
        axis=0,
    )

    minimum = int(weights.min())
    mean = float(weights.mean())
    maximum = int(weights.max())

    return DiffusionMetrics(
        z=validated_z,
        mode=mode,
        applications=validated_applications,
        minimum=minimum,
        mean=mean,
        maximum=maximum,
        minimum_coverage=minimum / state_bits,
        mean_coverage=mean / state_bits,
        maximum_coverage=maximum / state_bits,
    )


def measure_diffusion(
    layer: LinearLayer,
    z: object,
    applications: object,
    mode: DiffusionMode = "structural",
) -> DiffusionMetrics:
    """Construye la matriz, compone la capa y resume la difusión."""

    validated_z = _validate_word_size(z)

    one_step = build_linear_matrix(
        layer,
        validated_z,
    )

    dependencies = compose_dependency_matrix(
        one_step,
        applications,
        mode,
    )

    return summarize_dependency_matrix(
        dependencies,
        validated_z,
        applications,
        mode,
    )


__all__ = [
    "DiffusionMetrics",
    "DiffusionMode",
    "build_linear_matrix",
    "circulant_mix_x",
    "compose_dependency_matrix",
    "gf2_rank",
    "inverse_circulant_mix_x",
    "linear_layer_final_case",
    "linear_layer_v1",
    "measure_diffusion",
    "summarize_dependency_matrix",
]
