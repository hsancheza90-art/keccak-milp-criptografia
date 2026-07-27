from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np

from keccak_milp.layers import chi, rho_pi, theta


@dataclass(frozen=True)
class TwoPlusTwoTrail:
    """
    Trayectoria diferencial con dos S-boxes activas
    en las dos primeras rondas.
    """

    support_round_0: tuple[
        tuple[int, int],
        tuple[int, int],
    ]
    beta_round_0: tuple[int, int]
    delta_b1: np.ndarray
    support_round_1: tuple[
        tuple[int, int],
        tuple[int, int],
    ]


@dataclass(frozen=True)
class RestrictedThreeRoundSearchResult:
    """
    Resultado de la búsqueda exhaustiva restringida:

        2 + 2 + c

    para tres rondas de Keccak reducido.
    """

    z: int
    trail_count: int
    realizations_per_trail: int
    evaluated_realizations: int
    minimum_round_2_activity: int
    minimum_total_activity: int
    best_candidate_count: int

    support_round_0: tuple[
        tuple[int, int],
        tuple[int, int],
    ]
    beta_round_0: tuple[int, int]
    support_round_1: tuple[
        tuple[int, int],
        tuple[int, int],
    ]
    left_values_round_1: tuple[int, int]
    support_round_2: tuple[
        tuple[int, int],
        ...,
    ]
    delta_b2_hamming_weight: int

    histogram_round_2: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        """
        Convierte el resultado a una estructura serializable como JSON.
        """
        result = asdict(self)

        result["support_round_0"] = [
            list(position)
            for position in self.support_round_0
        ]

        result["beta_round_0"] = list(
            self.beta_round_0
        )

        result["support_round_1"] = [
            list(position)
            for position in self.support_round_1
        ]

        result["left_values_round_1"] = list(
            self.left_values_round_1
        )

        result["support_round_2"] = [
            list(position)
            for position in self.support_round_2
        ]

        result["histogram_round_2"] = {
            str(activity): count
            for activity, count in sorted(
                self.histogram_round_2.items()
            )
        }

        return result


def integer_to_five_bits(
    value: int,
) -> np.ndarray:
    """
    Convierte un entero entre 0 y 31 en cinco bits.

    El bit de índice x corresponde a la coordenada x
    de una aplicación local de chi.
    """
    if value not in range(32):
        raise ValueError(
            "El valor debe encontrarse entre 0 y 31."
        )

    return np.asarray(
        [
            (value >> x) & 1
            for x in range(5)
        ],
        dtype=np.int64,
    )


def active_sbox_positions(
    difference_state: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """
    Devuelve las posiciones (y, k) donde la diferencia
    de entrada a chi contiene al menos un bit activo.
    """
    state = np.asarray(
        difference_state,
        dtype=np.int64,
    )

    if (
        state.ndim != 3
        or state.shape[0:2] != (5, 5)
    ):
        raise ValueError(
            "El estado debe tener forma (5, 5, z)."
        )

    if not np.all(
        np.isin(
            state,
            [0, 1],
        )
    ):
        raise ValueError(
            "El estado diferencial debe ser binario."
        )

    z = state.shape[2]

    return tuple(
        (y, k)
        for y in range(5)
        for k in range(z)
        if np.any(
            state[:, y, k]
        )
    )


def _validate_word_size(
    z: int,
) -> None:
    if z not in (4, 8):
        raise ValueError(
            "La búsqueda académica admite únicamente z=4 o z=8."
        )


def _build_linear_images(
    z: int,
) -> dict[
    tuple[int, int, int],
    np.ndarray,
]:
    """
    Precalcula:

        L(beta) = rho_pi(theta(beta))

    para cada posición de S-box y cada diferencia
    de salida no nula de chi.
    """
    positions = [
        (y, k)
        for y in range(5)
        for k in range(z)
    ]

    linear_images: dict[
        tuple[int, int, int],
        np.ndarray,
    ] = {}

    for y, k in positions:
        for beta_integer in range(
            1,
            32,
        ):
            delta_after_chi = np.zeros(
                (5, 5, z),
                dtype=np.int64,
            )

            delta_after_chi[
                :,
                y,
                k,
            ] = integer_to_five_bits(
                beta_integer
            )

            linear_images[
                y,
                k,
                beta_integer,
            ] = rho_pi(
                theta(
                    delta_after_chi
                )
            )

    return linear_images


def enumerate_two_plus_two_trails(
    z: int,
) -> tuple[TwoPlusTwoTrail, ...]:
    """
    Enumera exhaustivamente todas las trayectorias diferenciales
    cuyo número de S-boxes activas es:

        ronda 0: 2
        ronda 1: 2

    La enumeración considera todos los soportes de tamaño dos
    y todas las diferencias beta no nulas de cinco bits.
    """
    _validate_word_size(
        z
    )

    positions = [
        (y, k)
        for y in range(5)
        for k in range(z)
    ]

    linear_images = _build_linear_images(
        z
    )

    trails: list[
        TwoPlusTwoTrail
    ] = []

    for (
        first_position,
        second_position,
    ) in combinations(
        positions,
        2,
    ):
        first_y, first_k = (
            first_position
        )

        second_y, second_k = (
            second_position
        )

        for first_beta in range(
            1,
            32,
        ):
            first_image = linear_images[
                first_y,
                first_k,
                first_beta,
            ]

            for second_beta in range(
                1,
                32,
            ):
                second_image = linear_images[
                    second_y,
                    second_k,
                    second_beta,
                ]

                delta_b1 = np.bitwise_xor(
                    first_image,
                    second_image,
                )

                support_round_1 = (
                    active_sbox_positions(
                        delta_b1
                    )
                )

                if len(
                    support_round_1
                ) != 2:
                    continue

                trails.append(
                    TwoPlusTwoTrail(
                        support_round_0=(
                            first_position,
                            second_position,
                        ),
                        beta_round_0=(
                            first_beta,
                            second_beta,
                        ),
                        delta_b1=(
                            delta_b1.copy()
                        ),
                        support_round_1=(
                            support_round_1
                        ),
                    )
                )

    return tuple(
        trails
    )


def search_three_round_two_plus_two(
    z: int,
) -> RestrictedThreeRoundSearchResult:
    """
    Ejecuta la búsqueda exhaustiva restringida:

        2 + 2 + c

    Para cada trayectoria 2+2 se enumeran los 32^2 valores
    absolutos posibles en las dos S-boxes activas de la
    segunda ronda.

    Las posiciones con diferencia cero pueden fijarse a cero
    porque sus estados izquierdo y derecho son iguales y no
    producen diferencia de salida en chi.
    """
    _validate_word_size(
        z
    )

    trails = enumerate_two_plus_two_trails(
        z
    )

    if not trails:
        raise RuntimeError(
            "No se encontraron trayectorias 2+2."
        )

    realizations_per_trail = (
        32 * 32
    )

    evaluated_realizations = 0
    minimum_round_2_activity = 5 * z
    best_candidate_count = 0

    best_trail: TwoPlusTwoTrail | None = None
    best_left_values: tuple[int, int] | None = None
    best_support_round_2: tuple[
        tuple[int, int],
        ...,
    ] | None = None
    best_delta_b2_hamming_weight: int | None = None

    histogram: Counter[int] = Counter()

    for trail in trails:
        (
            first_position,
            second_position,
        ) = trail.support_round_1

        first_y, first_k = (
            first_position
        )

        second_y, second_k = (
            second_position
        )

        for first_left_integer in range(
            32
        ):
            first_left_bits = (
                integer_to_five_bits(
                    first_left_integer
                )
            )

            for second_left_integer in range(
                32
            ):
                second_left_bits = (
                    integer_to_five_bits(
                        second_left_integer
                    )
                )

                evaluated_realizations += 1

                left_b1 = np.zeros(
                    (5, 5, z),
                    dtype=np.int64,
                )

                left_b1[
                    :,
                    first_y,
                    first_k,
                ] = first_left_bits

                left_b1[
                    :,
                    second_y,
                    second_k,
                ] = second_left_bits

                right_b1 = np.bitwise_xor(
                    left_b1,
                    trail.delta_b1,
                )

                delta_after_chi_round_1 = (
                    np.bitwise_xor(
                        chi(left_b1),
                        chi(right_b1),
                    )
                )

                delta_b2 = rho_pi(
                    theta(
                        delta_after_chi_round_1
                    )
                )

                support_round_2 = (
                    active_sbox_positions(
                        delta_b2
                    )
                )

                round_2_activity = len(
                    support_round_2
                )

                histogram[
                    round_2_activity
                ] += 1

                if (
                    round_2_activity
                    < minimum_round_2_activity
                ):
                    minimum_round_2_activity = (
                        round_2_activity
                    )

                    best_candidate_count = 1
                    best_trail = trail

                    best_left_values = (
                        first_left_integer,
                        second_left_integer,
                    )

                    best_support_round_2 = (
                        support_round_2
                    )

                    best_delta_b2_hamming_weight = int(
                        delta_b2.sum()
                    )

                elif (
                    round_2_activity
                    == minimum_round_2_activity
                ):
                    best_candidate_count += 1

    if (
        best_trail is None
        or best_left_values is None
        or best_support_round_2 is None
        or best_delta_b2_hamming_weight is None
    ):
        raise RuntimeError(
            "La búsqueda terminó sin un mejor candidato."
        )

    expected_realizations = (
        len(trails)
        * realizations_per_trail
    )

    if (
        evaluated_realizations
        != expected_realizations
    ):
        raise AssertionError(
            "El número de realizaciones evaluadas "
            "no coincide con el esperado."
        )

    return RestrictedThreeRoundSearchResult(
        z=z,
        trail_count=len(
            trails
        ),
        realizations_per_trail=(
            realizations_per_trail
        ),
        evaluated_realizations=(
            evaluated_realizations
        ),
        minimum_round_2_activity=(
            minimum_round_2_activity
        ),
        minimum_total_activity=(
            2
            + 2
            + minimum_round_2_activity
        ),
        best_candidate_count=(
            best_candidate_count
        ),
        support_round_0=(
            best_trail.support_round_0
        ),
        beta_round_0=(
            best_trail.beta_round_0
        ),
        support_round_1=(
            best_trail.support_round_1
        ),
        left_values_round_1=(
            best_left_values
        ),
        support_round_2=(
            best_support_round_2
        ),
        delta_b2_hamming_weight=(
            best_delta_b2_hamming_weight
        ),
        histogram_round_2=dict(
            sorted(
                histogram.items()
            )
        ),
    )
