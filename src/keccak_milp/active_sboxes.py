"""
Extensión diferencial para contar S-boxes activas en la capa chi.

Este módulo amplía el modelo emparejado exacto de Keccak con:

1. diferencias internas a la entrada de chi;
2. variables binarias de actividad por S-box.

Para cada ronda r y cada bit (x, y, k), se define:

    delta_chi_input[r, x, y, k]
        =
    rho_pi_left[r, x, y, k]
        XOR
    rho_pi_right[r, x, y, k]

La relación XOR se representa mediante:

    left + right = delta + 2 q

Cada S-box de chi se identifica mediante (r, y, k) y procesa los
cinco bits correspondientes a x = 0, ..., 4.

En esta etapa las variables active_chi ya se crean, pero todavía no
se agregan sus restricciones de enlace con delta_chi_input.
"""

from __future__ import annotations

from typing import TypeAlias

import pulp

from keccak_milp.config import ExperimentConfig
from keccak_milp.differential import (
    PairedKeccakMILPModel,
)


ChiInputDifferenceIndex: TypeAlias = tuple[
    int,
    int,
    int,
    int,
]

ActiveChiIndex: TypeAlias = tuple[
    int,
    int,
    int,
]


class ActiveSBoxPairedKeccakMILPModel(
    PairedKeccakMILPModel
):
    """
    Modelo emparejado con diferencias internas y actividad de chi.

    La entrada de chi corresponde a la salida de rho-pi de cada ronda.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        name: str | None = None,
    ) -> None:
        """Inicializa el modelo y las estructuras de la extensión."""
        super().__init__(
            config=config,
            name=name,
        )

        self.delta_chi_input: dict[
            ChiInputDifferenceIndex,
            pulp.LpVariable,
        ] = {}

        self.delta_chi_input_q: dict[
            ChiInputDifferenceIndex,
            pulp.LpVariable,
        ] = {}

        self.active_chi: dict[
            ActiveChiIndex,
            pulp.LpVariable,
        ] = {}

        self._chi_input_differences_built = False
        self._active_chi_variables_built = False

        self._active_sbox_upper_bound: int | None = None

        self._round_active_sbox_counts: dict[int, int] = {}

        self._round_active_sbox_supports: dict[int, frozenset[tuple[int, int]]] = {}

    def _create_binary_variable(
        self,
        name: str,
    ) -> pulp.LpVariable:
        """
        Crea una variable binaria asociada al problema combinado.

        Se utiliza add_variable cuando está disponible para mantener
        compatibilidad con las versiones recientes de PuLP.
        """
        if hasattr(self.problem, "add_variable"):
            return self.problem.add_variable(
                name=name,
                lowBound=0,
                upBound=1,
                cat=pulp.LpBinary,
            )

        return pulp.LpVariable(
            name=name,
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary,
        )

    def _create_chi_input_difference_variables(
        self,
    ) -> None:
        """
        Crea delta y la variable auxiliar q para cada bit de entrada
        de chi.
        """
        for round_index in range(
            self.config.rounds
        ):
            for x in range(5):
                for y in range(5):
                    for k in range(self.config.z):
                        index = (
                            round_index,
                            x,
                            y,
                            k,
                        )

                        self.delta_chi_input[index] = (
                            self._create_binary_variable(
                                name=(
                                    f"delta_chi_input"
                                    f"_r{round_index}"
                                    f"_x{x}_y{y}_k{k}"
                                )
                            )
                        )

                        self.delta_chi_input_q[index] = (
                            self._create_binary_variable(
                                name=(
                                    f"delta_chi_input_q"
                                    f"_r{round_index}"
                                    f"_x{x}_y{y}_k{k}"
                                )
                            )
                        )

    def _add_chi_input_difference_constraints(
        self,
    ) -> None:
        """
        Conecta las salidas rho-pi izquierda y derecha mediante XOR.

        Para cada bit se agrega:

            left + right = delta + 2 q
        """
        for round_index in range(
            self.config.rounds
        ):
            for x in range(5):
                for y in range(5):
                    for k in range(self.config.z):
                        index = (
                            round_index,
                            x,
                            y,
                            k,
                        )

                        left_variable = (
                            self.left.rho_pi_output_variable(
                                round_index=round_index,
                                x=x,
                                y=y,
                                k=k,
                            )
                        )

                        right_variable = (
                            self.right.rho_pi_output_variable(
                                round_index=round_index,
                                x=x,
                                y=y,
                                k=k,
                            )
                        )

                        delta_variable = (
                            self.delta_chi_input[index]
                        )

                        parity_variable = (
                            self.delta_chi_input_q[index]
                        )

                        self.problem += (
                            left_variable + right_variable
                            == (
                                delta_variable
                                + 2 * parity_variable
                            ),
                            (
                                f"delta_chi_input_xor"
                                f"_r{round_index}"
                                f"_x{x}_y{y}_k{k}"
                            ),
                        )

    def _create_active_chi_variables(
        self,
    ) -> None:
        """
        Crea una variable binaria por cada S-box de chi.

        Una S-box queda identificada mediante:

            round_index, y, k

        porque procesa simultáneamente los cinco bits asociados a
        x = 0, ..., 4.
        """
        for round_index in range(
            self.config.rounds
        ):
            for y in range(5):
                for k in range(self.config.z):
                    index = (
                        round_index,
                        y,
                        k,
                    )

                    self.active_chi[index] = (
                        self._create_binary_variable(
                            name=(
                                f"active_chi"
                                f"_r{round_index}"
                                f"_y{y}_k{k}"
                            )
                        )
                    )

    def _add_active_chi_constraints(
        self,
    ) -> None:
        """
        Enlaza cada variable active_chi con los cinco bits de
        diferencia que forman la entrada de una S-box de chi.

        Para cada S-box (r, y, k):

            delta_chi_input[r, x, y, k] <= active_chi[r, y, k]

        para x = 0, ..., 4, y:

            active_chi[r, y, k]
                <=
            sum_x delta_chi_input[r, x, y, k]
        """
        for round_index in range(
            self.config.rounds
        ):
            for y in range(5):
                for k in range(self.config.z):
                    active_variable = (
                        self.active_chi_variable(
                            round_index=round_index,
                            y=y,
                            k=k,
                        )
                    )

                    difference_variables = [
                        self.chi_input_difference_variable(
                            round_index=round_index,
                            x=x,
                            y=y,
                            k=k,
                        )
                        for x in range(5)
                    ]

                    # Si alg?n bit est? activo, la S-box debe estar activa.
                    for x, difference_variable in enumerate(
                        difference_variables
                    ):
                        self.problem += (
                            difference_variable
                            <= active_variable,
                            (
                                f"active_chi_lower"
                                f"_r{round_index}"
                                f"_x{x}_y{y}_k{k}"
                            ),
                        )

                    # Si todos los bits son cero, la S-box debe ser cero.
                    self.problem += (
                        active_variable
                        <= pulp.lpSum(
                            difference_variables
                        ),
                        (
                            f"active_chi_upper"
                            f"_r{round_index}"
                            f"_y{y}_k{k}"
                        ),
                    )

    def build_paired_model(self) -> None:
        """
        Construye el modelo emparejado, las diferencias internas y
        las variables de actividad de chi.

        La operación es idempotente.
        """
        super().build_paired_model()

        if not self._chi_input_differences_built:
            self._create_chi_input_difference_variables()
            self._add_chi_input_difference_constraints()

            self._chi_input_differences_built = True

        if not self._active_chi_variables_built:
            self._create_active_chi_variables()

            self._active_chi_variables_built = True

            self._add_active_chi_constraints()

    def chi_input_difference_variable(
        self,
        round_index: int,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """
        Devuelve una variable de diferencia a la entrada de chi.
        """
        if not self._chi_input_differences_built:
            raise RuntimeError(
                "Debe ejecutarse build_paired_model() antes de "
                "acceder a las diferencias de entrada de chi."
            )

        index = (
            round_index,
            x,
            y,
            k,
        )

        if index not in self.delta_chi_input:
            raise KeyError(
                "La diferencia de entrada de chi solicitada "
                "no existe."
            )

        return self.delta_chi_input[index]

    def active_chi_variable(
        self,
        round_index: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """
        Devuelve la variable binaria de actividad de una S-box.

        En esta etapa la variable todavía no está enlazada mediante
        restricciones OR con los cinco bits delta_chi_input.
        """
        if not self._active_chi_variables_built:
            raise RuntimeError(
                "Debe ejecutarse build_paired_model() antes de "
                "acceder a las variables de actividad de chi."
            )

        index = (
            round_index,
            y,
            k,
        )

        if index not in self.active_chi:
            raise KeyError(
                "La variable de actividad de chi solicitada "
                "no existe."
            )

        return self.active_chi[index]

    def set_active_sbox_objective(
        self,
    ) -> None:
        """
        Minimiza el n?mero total de S-boxes activas de chi.

        La funci?n objetivo es:

            sum active_chi[r, y, k]

        para todas las rondas, filas y posiciones de palabra.
        """
        if not self._active_chi_variables_built:
            raise RuntimeError(
                "Debe ejecutarse build_paired_model() antes de "
                "definir el objetivo de S-boxes activas."
            )

        if self._objective_added:
            raise RuntimeError(
                "El modelo ya tiene una funci?n objetivo."
            )

        objective = pulp.lpSum(
            self.active_chi[
                round_index,
                y,
                k,
            ]
            for round_index in range(
                self.config.rounds
            )
            for y in range(5)
            for k in range(self.config.z)
        )

        self.problem.sense = pulp.LpMinimize
        self.problem.setObjective(objective)

        self._objective_added = True

    def add_active_sbox_upper_bound(
        self,
        max_active_sboxes: int,
    ) -> None:
        """
        Agrega una cota superior al n?mero total de S-boxes activas.

        La restricci?n es:

            sum active_chi[r, y, k] <= max_active_sboxes

        La misma cota puede solicitarse repetidamente sin duplicar
        restricciones. Una cota distinta se rechaza para evitar
        modificar silenciosamente el experimento.
        """
        if not self._active_chi_variables_built:
            raise RuntimeError(
                "Debe ejecutarse build_paired_model() antes de "
                "agregar una cota superior de S-boxes activas."
            )

        if (
            isinstance(max_active_sboxes, bool)
            or not isinstance(max_active_sboxes, int)
        ):
            raise TypeError(
                "La cota superior debe ser un n?mero entero."
            )

        total_available_sboxes = (
            self.config.rounds
            * 5
            * self.config.z
        )

        if max_active_sboxes < 0:
            raise ValueError(
                "La cota superior no puede ser negativa."
            )

        if max_active_sboxes > total_available_sboxes:
            raise ValueError(
                "La cota superior no puede exceder el n?mero "
                "total de S-boxes disponibles."
            )

        if self._active_sbox_upper_bound is not None:
            if (
                self._active_sbox_upper_bound
                == max_active_sboxes
            ):
                return

            raise RuntimeError(
                "El modelo ya tiene una cota superior de "
                "S-boxes activas distinta."
            )

        total_activity = pulp.lpSum(
            self.active_chi[
                round_index,
                y,
                k,
            ]
            for round_index in range(
                self.config.rounds
            )
            for y in range(5)
            for k in range(self.config.z)
        )

        self.problem += (
            total_activity
            <= max_active_sboxes,
            "active_sbox_upper_bound",
        )

        self._active_sbox_upper_bound = (
            max_active_sboxes
        )

    def add_round_active_sbox_count(
        self,
        round_index: int,
        active_sboxes: int,
    ) -> None:
        """
        Fija exactamente el n?mero de S-boxes activas de una ronda.

        Para la ronda indicada se agrega:

            sum_{y,k} active_chi[round_index, y, k]
                =
            active_sboxes

        La misma configuraci?n es idempotente. Si la ronda ya tiene
        un valor diferente, el m?todo rechaza la modificaci?n.
        """
        if not self._active_chi_variables_built:
            raise RuntimeError(
                "Debe ejecutarse build_paired_model() antes de "
                "fijar la actividad de una ronda."
            )

        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
        ):
            raise TypeError(
                "El ?ndice de ronda debe ser un n?mero entero."
            )

        if round_index not in range(
            self.config.rounds
        ):
            raise ValueError(
                "El ?ndice de ronda debe encontrarse entre 0 y "
                f"{self.config.rounds - 1}."
            )

        if (
            isinstance(active_sboxes, bool)
            or not isinstance(active_sboxes, int)
        ):
            raise TypeError(
                "El n?mero de S-boxes activas debe ser entero."
            )

        available_in_round = (
            5 * self.config.z
        )

        if active_sboxes < 0:
            raise ValueError(
                "El n?mero de S-boxes activas no puede ser "
                "negativo."
            )

        if active_sboxes > available_in_round:
            raise ValueError(
                "El n?mero de S-boxes activas no puede exceder "
                f"{available_in_round} en una ronda."
            )

        if round_index in self._round_active_sbox_counts:
            current_value = (
                self._round_active_sbox_counts[
                    round_index
                ]
            )

            if current_value == active_sboxes:
                return

            raise RuntimeError(
                f"La ronda {round_index} ya tiene fijado un "
                "n?mero diferente de S-boxes activas."
            )

        round_activity = pulp.lpSum(
            self.active_chi[
                round_index,
                y,
                k,
            ]
            for y in range(5)
            for k in range(self.config.z)
        )

        self.problem += (
            round_activity == active_sboxes,
            (
                f"round_active_sbox_count"
                f"_r{round_index}"
            ),
        )

        self._round_active_sbox_counts[
            round_index
        ] = active_sboxes

    def fix_round_active_sbox_support(
        self,
        round_index: int,
        active_positions: object,
    ) -> None:
        """
        Fija exactamente el soporte de S-boxes activas de una ronda.

        Cada posici?n (y, k) incluida en active_positions se fija a 1.
        Todas las dem?s posiciones de la ronda se fijan a 0.

        La misma configuraci?n es idempotente. Un soporte diferente
        para una ronda ya configurada se rechaza.
        """
        if not self._active_chi_variables_built:
            raise RuntimeError(
                "Debe ejecutarse build_paired_model() antes de "
                "fijar el soporte activo de una ronda."
            )

        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
        ):
            raise TypeError(
                "El ?ndice de ronda debe ser un n?mero entero."
            )

        if round_index not in range(
            self.config.rounds
        ):
            raise ValueError(
                "El ?ndice de ronda debe encontrarse entre 0 y "
                f"{self.config.rounds - 1}."
            )

        if isinstance(
            active_positions,
            (str, bytes),
        ):
            raise TypeError(
                "Las posiciones activas deben proporcionarse como "
                "una colecci?n de pares (y, k)."
            )

        try:
            raw_positions = list(
                active_positions
            )
        except TypeError as error:
            raise TypeError(
                "Las posiciones activas deben proporcionarse como "
                "una colecci?n iterable de pares (y, k)."
            ) from error

        normalized_positions: set[
            tuple[int, int]
        ] = set()

        for position in raw_positions:
            if (
                not isinstance(
                    position,
                    (tuple, list),
                )
                or len(position) != 2
            ):
                raise TypeError(
                    "Cada posici?n activa debe ser un par (y, k)."
                )

            y, k = position

            if (
                isinstance(y, bool)
                or not isinstance(y, int)
                or isinstance(k, bool)
                or not isinstance(k, int)
            ):
                raise TypeError(
                    "Las coordenadas y y k deben ser n?meros enteros."
                )

            if y not in range(5):
                raise ValueError(
                    "La coordenada y debe encontrarse entre 0 y 4."
                )

            if k not in range(
                self.config.z
            ):
                raise ValueError(
                    "La coordenada k debe encontrarse entre 0 y "
                    f"{self.config.z - 1}."
                )

            normalized_positions.add(
                (
                    y,
                    k,
                )
            )

        normalized_support = frozenset(
            normalized_positions
        )

        if round_index in self._round_active_sbox_supports:
            current_support = (
                self._round_active_sbox_supports[
                    round_index
                ]
            )

            if current_support == normalized_support:
                return

            raise RuntimeError(
                f"La ronda {round_index} ya tiene fijado un "
                "soporte activo diferente."
            )

        if round_index in self._round_active_sbox_counts:
            expected_count = (
                self._round_active_sbox_counts[
                    round_index
                ]
            )

            if len(normalized_support) != expected_count:
                raise RuntimeError(
                    f"El soporte de la ronda {round_index} contiene "
                    f"{len(normalized_support)} posiciones, pero el "
                    f"conteo fijado previamente es {expected_count}."
                )

        for y in range(5):
            for k in range(self.config.z):
                expected_value = int(
                    (
                        y,
                        k,
                    )
                    in normalized_support
                )

                self.problem += (
                    self.active_chi[
                        round_index,
                        y,
                        k,
                    ]
                    == expected_value,
                    (
                        f"round_active_support"
                        f"_r{round_index}"
                        f"_y{y}"
                        f"_k{k}"
                    ),
                )

        self._round_active_sbox_supports[
            round_index
        ] = normalized_support

