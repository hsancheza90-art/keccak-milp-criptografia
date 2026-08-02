"""
Modelo MILP permanente de una ronda del caso final de Keccak.

La ronda representada es:

    L* -> Chi* -> Iota*

donde L* es la capa lineal reforzada, Chi* es la S-box propuesta
e Iota* aplica las constantes públicas dependientes de los parámetros
dinámicos.

Este módulo es independiente de los modelos funcionales V1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pulp
from numpy.typing import NDArray

from keccak_milp.diffusion import (
    linear_layer_final_case,
)
from keccak_milp.final_case import (
    encode_dynamic_parameters,
    iota_final_case,
    security_level_to_rounds,
)
from keccak_milp.final_case_milp import (
    ChiStarMILPVariables,
    add_chi_star_sbox_constraints,
    create_binary_variable,
    create_integer_variable,
)


StatePosition = tuple[int, int, int]


@dataclass(frozen=True)
class FinalCaseRoundMILPStatistics:
    """Resumen estructural del modelo de una ronda."""

    z: int
    state_bits: int
    sboxes: int
    variables: int
    constraints: int
    linear_parity_variables: int
    chi_auxiliary_variables: int


def _validate_integer(
    value: object,
    *,
    parameter_name: str,
) -> int:
    """Valida un entero excluyendo valores booleanos."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{parameter_name} debe ser un número entero."
        )

    return value


def _validate_word_size(
    z: object,
) -> int:
    """Valida los tamaños de palabra permitidos."""

    validated = _validate_integer(
        z,
        parameter_name="z",
    )

    if validated not in (4, 8):
        raise ValueError(
            "z debe ser 4 u 8."
        )

    return validated


def _validate_round_parameters(
    *,
    z: object,
    security_level: object,
    domain_id: object,
    round_index: object,
) -> tuple[int, int, int, int, int]:
    """Valida los parámetros dinámicos de la ronda."""

    validated_z = _validate_word_size(z)

    validated_security = _validate_integer(
        security_level,
        parameter_name="security_level",
    )

    validated_domain = _validate_integer(
        domain_id,
        parameter_name="domain_id",
    )

    validated_round = _validate_integer(
        round_index,
        parameter_name="round_index",
    )

    number_of_rounds = security_level_to_rounds(
        validated_security
    )

    encode_dynamic_parameters(
        validated_security,
        validated_domain,
        validated_z,
    )

    if validated_round not in range(
        number_of_rounds
    ):
        raise ValueError(
            "round_index debe encontrarse entre 0 y "
            f"{number_of_rounds - 1} para el nivel de seguridad "
            f"{validated_security}."
        )

    return (
        validated_z,
        validated_security,
        validated_domain,
        validated_round,
        number_of_rounds,
    )


@lru_cache(maxsize=2)
def _cached_state_positions(
    z: int,
) -> tuple[StatePosition, ...]:
    """Devuelve el orden canónico de bits del estado."""

    return tuple(
        (
            x,
            y,
            k,
        )
        for x in range(5)
        for y in range(5)
        for k in range(z)
    )


def final_case_state_positions(
    z: object,
) -> tuple[StatePosition, ...]:
    """Devuelve las posiciones del estado para un tamaño válido."""

    validated_z = _validate_word_size(z)

    return _cached_state_positions(
        validated_z
    )


@lru_cache(maxsize=2)
def _cached_final_case_linear_matrix(
    z: int,
) -> NDArray[np.uint8]:
    """Construye y conserva la matriz binaria exacta de L*."""

    positions = _cached_state_positions(z)
    state_bits = len(positions)

    matrix = np.zeros(
        (
            state_bits,
            state_bits,
        ),
        dtype=np.uint8,
    )

    for input_index, position in enumerate(
        positions
    ):
        basis = np.zeros(
            (
                5,
                5,
                z,
            ),
            dtype=np.int64,
        )

        basis[position] = 1

        transformed = linear_layer_final_case(
            basis
        )

        matrix[
            :,
            input_index,
        ] = transformed.reshape(
            -1
        ).astype(
            np.uint8
        )

    matrix.setflags(
        write=False
    )

    return matrix


def build_final_case_linear_matrix(
    z: object,
) -> NDArray[np.uint8]:
    """
    Devuelve una copia de la matriz binaria exacta de L*.

    La copia impide que una modificación externa altere la matriz
    conservada internamente.
    """

    validated_z = _validate_word_size(z)

    return _cached_final_case_linear_matrix(
        validated_z
    ).copy()


@lru_cache(maxsize=2)
def _cached_linear_supports(
    z: int,
) -> tuple[tuple[int, ...], ...]:
    """Devuelve los soportes de las ecuaciones lineales."""

    matrix = _cached_final_case_linear_matrix(
        z
    )

    return tuple(
        tuple(
            int(index)
            for index in np.flatnonzero(
                matrix[
                    output_index,
                    :,
                ]
            )
        )
        for output_index in range(
            matrix.shape[0]
        )
    )


def _validate_state_values(
    values: object,
    *,
    z: int,
) -> NDArray[np.int64]:
    """Valida y copia un estado binario."""

    if not isinstance(
        values,
        np.ndarray,
    ):
        raise TypeError(
            "values debe ser un arreglo de NumPy."
        )

    if values.shape != (
        5,
        5,
        z,
    ):
        raise ValueError(
            "values debe tener forma "
            f"(5, 5, {z})."
        )

    if not np.issubdtype(
        values.dtype,
        np.integer,
    ):
        raise TypeError(
            "values debe contener valores enteros."
        )

    if not np.all(
        np.isin(
            values,
            (
                0,
                1,
            ),
        )
    ):
        raise ValueError(
            "values debe ser un estado binario."
        )

    return values.astype(
        np.int64,
        copy=True,
    )


def _add_constraint(
    problem: pulp.LpProblem,
    constraint: pulp.LpConstraint,
    name: str,
) -> None:
    """Agrega una restricción con nombre explícito."""

    problem.addConstraint(
        constraint,
        name=name,
    )


class FinalCaseRoundMILPModel:
    """
    Modelo MILP exacto de una ronda del caso final.

    El modelo se construye explícitamente mediante build_model().
    La función objetivo de factibilidad se configura con
    set_feasibility_objective().
    """

    def __init__(
        self,
        *,
        z: int,
        security_level: int,
        domain_id: int,
        round_index: int,
        name: str | None = None,
    ) -> None:
        (
            validated_z,
            validated_security,
            validated_domain,
            validated_round,
            number_of_rounds,
        ) = _validate_round_parameters(
            z=z,
            security_level=security_level,
            domain_id=domain_id,
            round_index=round_index,
        )

        if name is not None:
            if not isinstance(name, str):
                raise TypeError(
                    "name debe ser una cadena de texto o None."
                )

            normalized_name = name.strip()

            if not normalized_name:
                raise ValueError(
                    "name no puede estar vacío."
                )
        else:
            normalized_name = (
                f"final_case_round"
                f"_z{validated_z}"
                f"_s{validated_security}"
                f"_d{validated_domain}"
                f"_r{validated_round}"
            )

        self.z = validated_z
        self.security_level = validated_security
        self.domain_id = validated_domain
        self.round_index = validated_round
        self.number_of_rounds = number_of_rounds
        self.name = normalized_name

        self.positions = _cached_state_positions(
            self.z
        )

        self._index_by_position = {
            position: index
            for index, position
            in enumerate(self.positions)
        }

        self.problem = pulp.LpProblem(
            self.name,
            pulp.LpMinimize,
        )

        self.input_state: dict[
            StatePosition,
            pulp.LpVariable,
        ] = {}

        self.linear_output: dict[
            StatePosition,
            pulp.LpVariable,
        ] = {}

        self.chi_output: dict[
            StatePosition,
            pulp.LpVariable,
        ] = {}

        self.round_output: dict[
            StatePosition,
            pulp.LpVariable,
        ] = {}

        self.linear_parity: dict[
            StatePosition,
            pulp.LpVariable,
        ] = {}

        self.chi_auxiliary: dict[
            tuple[int, int],
            ChiStarMILPVariables,
        ] = {}

        self._model_built = False
        self._objective_added = False

        self._fixed_input: tuple[int, ...] | None = None
        self._fixed_output: tuple[int, ...] | None = None

    def _require_built(
        self,
    ) -> None:
        """Exige que el modelo haya sido construido."""

        if not self._model_built:
            raise RuntimeError(
                "Debe ejecutarse build_model() primero."
            )

    def _create_state_variables(
        self,
    ) -> None:
        """Crea las cuatro fronteras de variables binarias."""

        for position in self.positions:
            x, y, k = position

            self.input_state[position] = (
                create_binary_variable(
                    self.problem,
                    (
                        f"{self.name}_input"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )
            )

            self.linear_output[position] = (
                create_binary_variable(
                    self.problem,
                    (
                        f"{self.name}_linear"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )
            )

            self.chi_output[position] = (
                create_binary_variable(
                    self.problem,
                    (
                        f"{self.name}_chi"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )
            )

            self.round_output[position] = (
                create_binary_variable(
                    self.problem,
                    (
                        f"{self.name}_output"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )
            )

    def _add_linear_constraints(
        self,
    ) -> None:
        """Agrega la matriz exacta de L* mediante paridades."""

        supports = _cached_linear_supports(
            self.z
        )

        input_variables = tuple(
            self.input_state[position]
            for position in self.positions
        )

        for output_index, position in enumerate(
            self.positions
        ):
            x, y, k = position
            support = supports[
                output_index
            ]

            if not support:
                raise RuntimeError(
                    "Se encontró una salida lineal sin soporte."
                )

            parity = create_integer_variable(
                self.problem,
                (
                    f"{self.name}_linear_parity"
                    f"_x{x}_y{y}_k{k}"
                ),
                low_bound=0,
                upper_bound=len(support) // 2,
            )

            self.linear_parity[position] = parity

            _add_constraint(
                self.problem,
                (
                    pulp.lpSum(
                        input_variables[input_index]
                        for input_index in support
                    )
                    == (
                        self.linear_output[position]
                        + 2 * parity
                    )
                ),
                (
                    f"{self.name}_linear_xor"
                    f"_x{x}_y{y}_k{k}"
                ),
            )

    def _add_chi_star_constraints(
        self,
    ) -> None:
        """Agrega una S-box Chi* por cada par (y, k)."""

        for y in range(5):
            for k in range(self.z):
                inputs = tuple(
                    self.linear_output[
                        (
                            x,
                            y,
                            k,
                        )
                    ]
                    for x in range(5)
                )

                outputs = tuple(
                    self.chi_output[
                        (
                            x,
                            y,
                            k,
                        )
                    ]
                    for x in range(5)
                )

                self.chi_auxiliary[
                    (
                        y,
                        k,
                    )
                ] = add_chi_star_sbox_constraints(
                    self.problem,
                    inputs,
                    outputs,
                    prefix=(
                        f"{self.name}_chi_star"
                        f"_y{y}_k{k}"
                    ),
                )

    def _add_iota_constraints(
        self,
    ) -> None:
        """Agrega la constante pública de Iota*."""

        zero_state = np.zeros(
            (
                5,
                5,
                self.z,
            ),
            dtype=np.int64,
        )

        constant = iota_final_case(
            zero_state,
            round_index=self.round_index,
            security_level=self.security_level,
            domain_id=self.domain_id,
        )

        for position in self.positions:
            x, y, k = position

            chi_variable = self.chi_output[
                position
            ]

            output_variable = self.round_output[
                position
            ]

            constant_bit = int(
                constant[position]
            )

            if constant_bit == 0:
                constraint = (
                    output_variable
                    == chi_variable
                )
            else:
                constraint = (
                    output_variable
                    + chi_variable
                    == 1
                )

            _add_constraint(
                self.problem,
                constraint,
                (
                    f"{self.name}_iota"
                    f"_x{x}_y{y}_k{k}"
                ),
            )

    def build_model(
        self,
    ) -> None:
        """Construye la ronda completa de forma idempotente."""

        if self._model_built:
            return

        self._create_state_variables()
        self._add_linear_constraints()
        self._add_chi_star_constraints()
        self._add_iota_constraints()

        self._model_built = True

    def set_feasibility_objective(
        self,
    ) -> None:
        """Define una función objetivo nula."""

        self._require_built()

        if self._objective_added:
            return

        self.problem.setObjective(
            pulp.LpAffineExpression()
        )

        self._objective_added = True

    def _fix_values(
        self,
        *,
        values: NDArray[np.int64],
        variables: dict[
            StatePosition,
            pulp.LpVariable,
        ],
        prefix: str,
    ) -> None:
        """Fija una frontera completa del estado."""

        for position in self.positions:
            x, y, k = position

            _add_constraint(
                self.problem,
                variables[position]
                == int(values[position]),
                (
                    f"{self.name}_{prefix}"
                    f"_x{x}_y{y}_k{k}"
                ),
            )

    def fix_input_values(
        self,
        values: object,
    ) -> None:
        """Fija el estado de entrada de forma idempotente."""

        self._require_built()

        normalized = _validate_state_values(
            values,
            z=self.z,
        )

        signature = tuple(
            int(value)
            for value in normalized.reshape(-1)
        )

        if self._fixed_input is not None:
            if self._fixed_input == signature:
                return

            raise RuntimeError(
                "El estado de entrada ya fue fijado "
                "con valores diferentes."
            )

        self._fix_values(
            values=normalized,
            variables=self.input_state,
            prefix="fix_input",
        )

        self._fixed_input = signature

    def fix_output_values(
        self,
        values: object,
    ) -> None:
        """Fija el estado de salida de forma idempotente."""

        self._require_built()

        normalized = _validate_state_values(
            values,
            z=self.z,
        )

        signature = tuple(
            int(value)
            for value in normalized.reshape(-1)
        )

        if self._fixed_output is not None:
            if self._fixed_output == signature:
                return

            raise RuntimeError(
                "El estado de salida ya fue fijado "
                "con valores diferentes."
            )

        self._fix_values(
            values=normalized,
            variables=self.round_output,
            prefix="fix_output",
        )

        self._fixed_output = signature

    def solve(
        self,
        solver: pulp.LpSolver,
    ) -> str:
        """Resuelve el modelo utilizando el solver proporcionado."""

        self._require_built()

        if not self._objective_added:
            raise RuntimeError(
                "El modelo no tiene función objetivo."
            )

        if not isinstance(
            solver,
            pulp.LpSolver,
        ):
            raise TypeError(
                "solver debe ser una instancia de pulp.LpSolver."
            )

        self.problem.solve(
            solver
        )

        return pulp.LpStatus[
            self.problem.status
        ]

    def _variable_at(
        self,
        mapping: dict[
            StatePosition,
            pulp.LpVariable,
        ],
        *,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Recupera una variable validando su posición."""

        self._require_built()

        validated_x = _validate_integer(
            x,
            parameter_name="x",
        )

        validated_y = _validate_integer(
            y,
            parameter_name="y",
        )

        validated_k = _validate_integer(
            k,
            parameter_name="k",
        )

        if validated_x not in range(5):
            raise ValueError(
                "x debe encontrarse entre 0 y 4."
            )

        if validated_y not in range(5):
            raise ValueError(
                "y debe encontrarse entre 0 y 4."
            )

        if validated_k not in range(self.z):
            raise ValueError(
                "k debe encontrarse entre 0 y "
                f"{self.z - 1}."
            )

        return mapping[
            (
                validated_x,
                validated_y,
                validated_k,
            )
        ]

    def input_variable(
        self,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve una variable de entrada."""

        return self._variable_at(
            self.input_state,
            x=x,
            y=y,
            k=k,
        )

    def linear_output_variable(
        self,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve una variable posterior a L*."""

        return self._variable_at(
            self.linear_output,
            x=x,
            y=y,
            k=k,
        )

    def chi_output_variable(
        self,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve una variable posterior a Chi*."""

        return self._variable_at(
            self.chi_output,
            x=x,
            y=y,
            k=k,
        )

    def output_variable(
        self,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve una variable de salida de ronda."""

        return self._variable_at(
            self.round_output,
            x=x,
            y=y,
            k=k,
        )

    def output_values(
        self,
        *,
        tolerance: float = 0.5,
    ) -> NDArray[np.int64]:
        """Recupera el estado de salida después de resolver."""

        self._require_built()

        if not isinstance(
            tolerance,
            (int, float),
        ) or isinstance(tolerance, bool):
            raise TypeError(
                "tolerance debe ser numérico."
            )

        if not 0.0 < float(tolerance) < 1.0:
            raise ValueError(
                "tolerance debe encontrarse entre 0 y 1."
            )

        result = np.zeros(
            (
                5,
                5,
                self.z,
            ),
            dtype=np.int64,
        )

        for position in self.positions:
            value = self.round_output[
                position
            ].value()

            if value is None:
                raise RuntimeError(
                    "El modelo todavía no tiene una solución."
                )

            result[position] = int(
                value >= float(tolerance)
            )

        return result

    def statistics(
        self,
    ) -> FinalCaseRoundMILPStatistics:
        """Devuelve las dimensiones actuales del modelo."""

        self._require_built()

        chi_auxiliary_count = sum(
            len(auxiliary.products)
            + len(auxiliary.parity)
            for auxiliary in self.chi_auxiliary.values()
        )

        return FinalCaseRoundMILPStatistics(
            z=self.z,
            state_bits=25 * self.z,
            sboxes=5 * self.z,
            variables=self.problem.numVariables(),
            constraints=self.problem.numConstraints(),
            linear_parity_variables=len(
                self.linear_parity
            ),
            chi_auxiliary_variables=(
                chi_auxiliary_count
            ),
        )


__all__ = [
    "FinalCaseRoundMILPModel",
    "FinalCaseRoundMILPStatistics",
    "StatePosition",
    "build_final_case_linear_matrix",
    "final_case_state_positions",
]
