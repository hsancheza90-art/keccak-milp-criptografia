"""
Capa no lineal del caso final de Keccak reducido.

La S-box propuesta se define sobre GF(2^5) mediante:

    Chi*(x) = (x XOR 0x06)^20 XOR 0x17

con polinomio irreducible:

    u^5 + u^3 + 1

La implementación funcional utiliza una red ANF compartida con:

    10 AND
    19 XOR
    0 NOT
    profundidad lógica 4

No se afirma que esta red sea un circuito mínimo global.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from keccak_milp.layers import validate_state_shape


CHI_STAR_TABLE: tuple[int, ...] = (
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


def _build_inverse_table(
    table: tuple[int, ...],
) -> tuple[int, ...]:
    """Construye la tabla inversa de una permutación."""

    if len(table) != 32:
        raise ValueError(
            "La tabla de la S-box debe contener 32 entradas."
        )

    if set(table) != set(range(32)):
        raise ValueError(
            "La tabla de la S-box debe ser una permutación "
            "de los valores entre 0 y 31."
        )

    inverse = [0] * 32

    for input_value, output_value in enumerate(table):
        inverse[output_value] = input_value

    return tuple(inverse)


CHI_STAR_INVERSE_TABLE: tuple[int, ...] = (
    _build_inverse_table(CHI_STAR_TABLE)
)


@dataclass(frozen=True)
class NonlinearCost:
    """Coste lógico de Chi* para una ronda."""

    z: int
    sboxes_per_round: int
    and_per_sbox: int
    xor_per_sbox: int
    not_per_sbox: int
    logical_depth: int
    total_and: int
    total_xor: int
    total_not: int


def _validate_plain_integer(
    value: object,
    parameter_name: str,
) -> int:
    """Valida un entero y rechaza booleanos."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"El parámetro {parameter_name} debe ser un entero."
        )

    return value


def _validate_sbox_value(
    value: object,
) -> int:
    """Valida una entrada o salida de cinco bits."""

    validated = _validate_plain_integer(
        value,
        "value",
    )

    if validated not in range(32):
        raise ValueError(
            "El valor de la S-box debe encontrarse "
            "entre 0 y 31."
        )

    return validated


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
            "La capa Chi* requiere un estado binario."
        )

    return z


def chi_star_sbox(
    value: object,
) -> int:
    """Evalúa la S-box Chi* sobre un valor de cinco bits."""

    validated = _validate_sbox_value(value)

    return CHI_STAR_TABLE[validated]


def chi_star_inverse_sbox(
    value: object,
) -> int:
    """Evalúa la S-box inversa de Chi*."""

    validated = _validate_sbox_value(value)

    return CHI_STAR_INVERSE_TABLE[validated]


def chi_star(
    state: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    Aplica Chi* a cada slice de cinco bits del estado.

    Cada S-box se identifica mediante ``(y, k)`` y procesa:

        state[0:5, y, k]

    La red utiliza diez monomios cuadráticos y subexpresiones
    XOR compartidas. El estado de entrada no se modifica.
    """

    _validate_binary_state(state)

    source = state.astype(
        np.int64,
        copy=False,
    )

    x0 = source[0, :, :]
    x1 = source[1, :, :]
    x2 = source[2, :, :]
    x3 = source[3, :, :]
    x4 = source[4, :, :]

    # Diez monomios cuadráticos compartidos.
    q01 = x0 & x1
    q02 = x0 & x2
    q03 = x0 & x3
    q04 = x0 & x4

    q12 = x1 & x2
    q13 = x1 & x3
    q14 = x1 & x4

    q23 = x2 & x3
    q24 = x2 & x4
    q34 = x3 & x4

    # Cinco subexpresiones XOR compartidas.
    t0 = x3 ^ q01
    t1 = x0 ^ q34
    t2 = q12 ^ t1
    t3 = q02 ^ t0
    t4 = x1 ^ q24

    # Coordenadas de salida.
    y0 = q24 ^ t2

    y1_left = t4 ^ x4
    y1_right_1 = q03 ^ q13
    y1_right_2 = y1_right_1 ^ q34
    y1 = y1_left ^ y1_right_2

    y2_left = q03 ^ q23
    y2_middle = y2_left ^ t0
    y2 = y2_middle ^ t2

    y3_left = x2 ^ q04
    y3_middle = y3_left ^ t4
    y3 = y3_middle ^ t3

    y4_left = q13 ^ q14
    y4_middle = y4_left ^ q12
    y4 = y4_middle ^ t3

    output = np.empty_like(
        source,
        dtype=np.int64,
    )

    output[0, :, :] = y0
    output[1, :, :] = y1
    output[2, :, :] = y2
    output[3, :, :] = y3
    output[4, :, :] = y4

    return output


def chi_star_inverse(
    state: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    Aplica la permutación inversa de Chi* a cada S-box.

    La inversa se evalúa mediante la tabla exhaustiva auditada.
    """

    _validate_binary_state(state)

    source = state.astype(
        np.int64,
        copy=False,
    )

    packed = np.zeros(
        source.shape[1:],
        dtype=np.int64,
    )

    for x in range(5):
        packed |= source[x, :, :] << x

    inverse_table = np.asarray(
        CHI_STAR_INVERSE_TABLE,
        dtype=np.int64,
    )

    unpacked_value = inverse_table[packed]

    output = np.empty_like(
        source,
        dtype=np.int64,
    )

    for x in range(5):
        output[x, :, :] = (
            unpacked_value >> x
        ) & 1

    return output


def chi_star_cost(
    z: object,
) -> NonlinearCost:
    """Devuelve el coste lógico conservador por ronda."""

    validated_z = _validate_word_size(z)

    sboxes_per_round = 5 * validated_z

    and_per_sbox = 10
    xor_per_sbox = 19
    not_per_sbox = 0
    logical_depth = 4

    return NonlinearCost(
        z=validated_z,
        sboxes_per_round=sboxes_per_round,
        and_per_sbox=and_per_sbox,
        xor_per_sbox=xor_per_sbox,
        not_per_sbox=not_per_sbox,
        logical_depth=logical_depth,
        total_and=(
            and_per_sbox
            * sboxes_per_round
        ),
        total_xor=(
            xor_per_sbox
            * sboxes_per_round
        ),
        total_not=(
            not_per_sbox
            * sboxes_per_round
        ),
    )


__all__ = [
    "CHI_STAR_INVERSE_TABLE",
    "CHI_STAR_TABLE",
    "NonlinearCost",
    "chi_star",
    "chi_star_cost",
    "chi_star_inverse",
    "chi_star_inverse_sbox",
    "chi_star_sbox",
]
