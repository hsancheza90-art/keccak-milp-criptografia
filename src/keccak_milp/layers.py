"""
Transformaciones estructurales de Keccak.

Convención utilizada
---------------------
El estado se representa como:

    A[x, y, k]

donde:

    x, y ∈ {0, 1, 2, 3, 4}
    k    ∈ {0, ..., z - 1}

La forma del arreglo NumPy es:

    (5, 5, z)

En esta etapa se implementan únicamente las capas rho y pi, que son
permutaciones deterministas. Estas funciones se usarán para validar la
indexación antes de construir las restricciones MILP.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray


T = TypeVar("T")


# ============================================================
# CONSTANTES DE KECCAK
# ============================================================

# Desplazamientos oficiales de la transformación rho.
#
# La entrada RHO_OFFSETS[x][y] indica cuántas posiciones se rota
# el lane A[x, y].
#
# Para versiones reducidas de Keccak se utiliza:
#
#     desplazamiento efectivo = RHO_OFFSETS[x][y] mod z
#
RHO_OFFSETS: tuple[tuple[int, ...], ...] = (
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)


# ============================================================
# VALIDACIÓN DEL ESTADO
# ============================================================

def validate_state_shape(state: NDArray[T]) -> int:
    """
    Valida que el estado tenga forma (5, 5, z).

    Parameters
    ----------
    state:
        Arreglo que representa el estado de Keccak.

    Returns
    -------
    int
        Tamaño de palabra z.

    Raises
    ------
    TypeError
        Si el estado no es un arreglo NumPy.

    ValueError
        Si el estado no tiene tres dimensiones, no tiene una matriz
        de 5 x 5 lanes o si z no es positivo.
    """

    if not isinstance(state, np.ndarray):
        raise TypeError(
            "El estado debe ser un arreglo NumPy."
        )

    if state.ndim != 3:
        raise ValueError(
            "El estado debe tener tres dimensiones: (5, 5, z). "
            f"Forma recibida: {state.shape}."
        )

    if state.shape[0] != 5 or state.shape[1] != 5:
        raise ValueError(
            "Keccak requiere una matriz de 5 x 5 lanes. "
            f"Forma recibida: {state.shape}."
        )

    z = state.shape[2]

    if z <= 0:
        raise ValueError(
            "El tamaño de palabra z debe ser mayor que cero."
        )

    return z


def create_labeled_state(z: int) -> NDArray[np.int64]:
    """
    Crea un estado cuyos elementos tienen identificadores únicos.

    Esta función es útil para verificar visualmente que rho y pi
    únicamente permutan posiciones.

    Parameters
    ----------
    z:
        Tamaño de cada lane.

    Returns
    -------
    numpy.ndarray
        Estado de forma (5, 5, z) con valores de 0 a 25z - 1.
    """

    if not isinstance(z, int):
        raise TypeError("El tamaño z debe ser entero.")

    if z <= 0:
        raise ValueError("El tamaño z debe ser mayor que cero.")

    return np.arange(
        25 * z,
        dtype=np.int64,
    ).reshape(5, 5, z)

# ============================================================
# CAPA THETA
# ============================================================

def column_parities(
    state: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    Calcula las paridades de columna utilizadas por theta.

    Para cada x y k:

        C[x, k] = XOR_{y=0}^{4} A[x, y, k]

    Parameters
    ----------
    state:
        Estado binario de forma (5, 5, z).

    Returns
    -------
    numpy.ndarray
        Matriz binaria C de forma (5, z).
    """

    validate_state_shape(state)

    if not np.all(np.isin(state, [0, 1])):
        raise ValueError(
            "La capa theta requiere un estado binario."
        )

    # XOR de los cinco elementos en la dimensión y.
    return np.bitwise_xor.reduce(
        state,
        axis=1,
    ).astype(np.int64)


def theta_effect(
    state: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    Calcula la matriz D utilizada por theta.

    Para cada x y k:

        D[x, k] =
            C[x-1 mod 5, k]
            XOR
            C[x+1 mod 5, k-1 mod z]

    Parameters
    ----------
    state:
        Estado binario de forma (5, 5, z).

    Returns
    -------
    numpy.ndarray
        Matriz binaria D de forma (5, z).
    """

    z = validate_state_shape(state)
    parities = column_parities(state)

    effect = np.zeros(
        (5, z),
        dtype=np.int64,
    )

    for x in range(5):
        for k in range(z):
            left_column = (x - 1) % 5
            right_column = (x + 1) % 5
            rotated_k = (k - 1) % z

            effect[x, k] = (
                parities[left_column, k]
                ^ parities[right_column, rotated_k]
            )

    return effect


def theta(
    state: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    Aplica la transformación theta a un estado binario.

    Para cada posición:

        A_theta[x, y, k] =
            A[x, y, k] XOR D[x, k]

    Parameters
    ----------
    state:
        Estado binario de forma (5, 5, z).

    Returns
    -------
    numpy.ndarray
        Estado binario transformado.
    """

    validate_state_shape(state)

    if not np.all(np.isin(state, [0, 1])):
        raise ValueError(
            "La capa theta requiere un estado binario."
        )

    effect = theta_effect(state)
    output = np.empty_like(
        state,
        dtype=np.int64,
    )

    for x in range(5):
        for y in range(5):
            for k in range(state.shape[2]):
                output[x, y, k] = (
                    int(state[x, y, k])
                    ^ int(effect[x, k])
                )

    return output


def hamming_weight(
    state: NDArray[np.integer],
) -> int:
    """
    Calcula el número de posiciones activas de un estado binario.
    """

    validate_state_shape(state)

    if not np.all(np.isin(state, [0, 1])):
        raise ValueError(
            "El peso de Hamming requiere un estado binario."
        )

    return int(np.sum(state))


def create_single_active_bit_state(
    z: int,
    x: int,
    y: int,
    k: int,
) -> NDArray[np.int64]:
    """
    Crea un estado binario con una sola posición activa.
    """

    if z <= 0:
        raise ValueError(
            "El tamaño de palabra z debe ser positivo."
        )

    if x not in range(5) or y not in range(5):
        raise ValueError(
            "Los índices x e y deben estar entre 0 y 4."
        )

    if k not in range(z):
        raise ValueError(
            f"El índice k debe estar entre 0 y {z - 1}."
        )

    state = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    state[x, y, k] = 1

    return state

# ============================================================
# CAPA RHO
# ============================================================

def rho_offset(x: int, y: int, z: int) -> int:
    """
    Devuelve el desplazamiento efectivo de rho.

    El desplazamiento oficial se reduce módulo z para las versiones
    reducidas de Keccak.
    """

    if x not in range(5) or y not in range(5):
        raise ValueError(
            "Los índices x e y deben estar entre 0 y 4."
        )

    if z <= 0:
        raise ValueError(
            "El tamaño de palabra z debe ser mayor que cero."
        )

    return RHO_OFFSETS[x][y] % z


def rho(state: NDArray[T]) -> NDArray[T]:
    """
    Aplica la transformación rho al estado.

    Para cada lane A[x, y], los bits se rotan según:

        A_rho[x, y, (k + r[x,y]) mod z] = A[x, y, k]

    Parameters
    ----------
    state:
        Estado de forma (5, 5, z).

    Returns
    -------
    numpy.ndarray
        Nuevo estado después de rho.
    """

    z = validate_state_shape(state)

    output = np.empty_like(state)

    for x in range(5):
        for y in range(5):
            shift = rho_offset(x, y, z)

            # np.roll con shift positivo implementa:
            #
            # output[(k + shift) mod z] = input[k]
            output[x, y, :] = np.roll(
                state[x, y, :],
                shift=shift,
            )

    return output


# ============================================================
# CAPA PI
# ============================================================

def pi_destination(x: int, y: int) -> tuple[int, int]:
    """
    Calcula la posición destino de un lane en la capa pi.

    Se utiliza la convención:

        B[y, (2x + 3y) mod 5] = A[x, y]

    Returns
    -------
    tuple[int, int]
        Coordenadas destino (x_destino, y_destino).
    """

    if x not in range(5) or y not in range(5):
        raise ValueError(
            "Los índices x e y deben estar entre 0 y 4."
        )

    x_destination = y
    y_destination = (2 * x + 3 * y) % 5

    return x_destination, y_destination


def pi(state: NDArray[T]) -> NDArray[T]:
    """
    Aplica la transformación pi al estado.

    La transformación cambia la posición de los lanes, pero no modifica
    el orden interno de los bits de cada lane.

    Parameters
    ----------
    state:
        Estado de forma (5, 5, z).

    Returns
    -------
    numpy.ndarray
        Nuevo estado después de pi.
    """

    validate_state_shape(state)

    output = np.empty_like(state)

    for x in range(5):
        for y in range(5):
            x_destination, y_destination = pi_destination(x, y)

            output[x_destination, y_destination, :] = state[x, y, :]

    return output


# ============================================================
# COMPOSICIÓN RHO + PI
# ============================================================

def rho_pi(state: NDArray[T]) -> NDArray[T]:
    """
    Aplica rho y luego pi.

    Esta es la secuencia utilizada dentro de una ronda de Keccak:

        A --rho--> A_rho --pi--> B
    """

    return pi(rho(state))


def rho_pi_destination(
    x: int,
    y: int,
    k: int,
    z: int,
) -> tuple[int, int, int]:
    """
    Calcula directamente el destino de un bit después de rho y pi.

    Esta función será útil posteriormente para construir las igualdades
    del modelo MILP sin necesidad de crear operaciones NumPy.

    Returns
    -------
    tuple[int, int, int]
        Coordenadas destino después de rho y pi.
    """

    if k not in range(z):
        raise ValueError(
            f"El índice k debe estar entre 0 y {z - 1}."
        )

    shift = rho_offset(x, y, z)
    k_destination = (k + shift) % z

    x_destination, y_destination = pi_destination(x, y)

    return (
        x_destination,
        y_destination,
        k_destination,
    )

def chi(
    state: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    Aplica la transformación chi a un estado binario.

    Para cada posición del estado:

        output[x, y, k] =
            state[x, y, k]
            XOR
            (
                NOT state[(x + 1) mod 5, y, k]
                AND state[(x + 2) mod 5, y, k]
            )

    Parameters
    ----------
    state:
        Estado binario con forma ``(5, 5, z)``.

    Returns
    -------
    NDArray[np.int64]
        Estado binario después de aplicar chi.

    Raises
    ------
    TypeError
        Si el estado no es un arreglo NumPy.
    ValueError
        Si el estado no tiene forma válida o contiene valores
        distintos de cero y uno.
    """
    validate_state_shape(state)

    if not np.all(np.isin(state, [0, 1])):
        raise ValueError(
            "La capa chi requiere un estado binario."
        )

    z = state.shape[2]

    output = np.empty(
        (5, 5, z),
        dtype=np.int64,
    )

    input_state = state.astype(
        np.int64,
        copy=False,
    )

    for x in range(5):
        next_x = (x + 1) % 5
        next_next_x = (x + 2) % 5

        for y in range(5):
            current = input_state[x, y, :]
            adjacent = input_state[next_x, y, :]
            second_adjacent = input_state[
                next_next_x,
                y,
                :,
            ]

            nonlinear_term = (
                (1 - adjacent)
                & second_adjacent
            )

            output[x, y, :] = (
                current
                ^ nonlinear_term
            )

    return output

# ============================================================
# UTILIDADES DE VALIDACIÓN
# ============================================================

def is_permutation(
    original: NDArray[T],
    transformed: NDArray[T],
) -> bool:
    """
    Verifica que dos estados contengan exactamente los mismos elementos.

    Esta comprobación es válida para estados etiquetados con valores únicos.
    """

    validate_state_shape(original)
    validate_state_shape(transformed)

    if original.shape != transformed.shape:
        return False

    return bool(
        np.array_equal(
            np.sort(original.reshape(-1)),
            np.sort(transformed.reshape(-1)),
        )
    )