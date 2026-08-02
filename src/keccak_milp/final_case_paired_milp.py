"""
Modelo MILP diferencial emparejado del caso final de Keccak.

El modelo incorpora dos ramas funcionales con los mismos parámetros
dinámicos y define exactamente:

    delta_input = input_left XOR input_right

    delta_chi_input =
        linear_output_left XOR linear_output_right

    active_sbox =
        OR(delta_chi_input[0], ..., delta_chi_input[4])

La función objetivo minimiza la cantidad total de S-boxes activas.

Este módulo no modifica los modelos V1 ni los modelos funcionales
permanentes ya validados.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np
import pulp
from numpy.typing import NDArray

from keccak_milp.final_case import (
    encode_dynamic_parameters,
    security_level_to_rounds,
)
from keccak_milp.final_case_milp import (
    create_binary_variable,
)
from keccak_milp.final_case_round_milp import (
    StatePosition,
)
from keccak_milp.final_case_rounds_milp import (
    FinalCaseRoundsMILPModel,
)


ChiInputDifferenceKey = tuple[
    int,
    int,
    int,
    int,
]

ActiveSBoxKey = tuple[
    int,
    int,
    int,
]


@dataclass(frozen=True)
class FinalCasePairedMILPStatistics:
    """Resumen estructural del modelo diferencial emparejado."""

    z: int
    security_level: int
    domain_id: int
    rounds: int
    state_bits: int

    variables: int
    constraints: int

    left_variables: int
    right_variables: int

    left_constraints: int
    right_constraints: int

    initial_difference_variables: int
    chi_input_difference_variables: int
    active_sbox_variables: int


def _validate_integer(
    value: object,
    *,
    parameter_name: str,
) -> int:
    """Valida un entero y excluye valores booleanos."""

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
    """Valida los tamaños de palabra admitidos."""

    validated = _validate_integer(
        z,
        parameter_name="z",
    )

    if validated not in (
        4,
        8,
    ):
        raise ValueError(
            "z debe ser 4 u 8."
        )

    return validated


def _validate_parameters(
    *,
    z: object,
    security_level: object,
    domain_id: object,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    """Valida los parámetros dinámicos."""

    validated_z = _validate_word_size(
        z
    )

    validated_security = _validate_integer(
        security_level,
        parameter_name="security_level",
    )

    validated_domain = _validate_integer(
        domain_id,
        parameter_name="domain_id",
    )

    rounds = security_level_to_rounds(
        validated_security
    )

    encode_dynamic_parameters(
        validated_security,
        validated_domain,
        validated_z,
    )

    return (
        validated_z,
        validated_security,
        validated_domain,
        rounds,
    )


def _validate_name(
    name: object,
    *,
    z: int,
    security_level: int,
    domain_id: int,
) -> str:
    """Valida o genera el nombre del modelo."""

    if name is None:
        return (
            "final_case_paired"
            f"_z{z}"
            f"_s{security_level}"
            f"_d{domain_id}"
        )

    if not isinstance(
        name,
        str,
    ):
        raise TypeError(
            "name debe ser una cadena de texto o None."
        )

    normalized = name.strip()

    if not normalized:
        raise ValueError(
            "name no puede estar vacío."
        )

    return normalized


def _validate_branch(
    branch: object,
) -> str:
    """Valida el identificador de una rama."""

    if not isinstance(
        branch,
        str,
    ):
        raise TypeError(
            "branch debe ser una cadena de texto."
        )

    normalized = branch.strip().lower()

    if normalized not in (
        "left",
        "right",
    ):
        raise ValueError(
            "branch debe ser left o right."
        )

    return normalized


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


def _validate_tolerance(
    tolerance: object,
) -> float:
    """Valida el umbral usado para recuperar bits."""

    if (
        isinstance(tolerance, bool)
        or not isinstance(
            tolerance,
            (
                int,
                float,
            ),
        )
    ):
        raise TypeError(
            "tolerance debe ser numérico."
        )

    normalized = float(
        tolerance
    )

    if not (
        0.0
        < normalized
        < 1.0
    ):
        raise ValueError(
            "tolerance debe encontrarse entre 0 y 1."
        )

    return normalized


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


def _add_xor2_constraints(
    problem: pulp.LpProblem,
    left: pulp.LpVariable,
    right: pulp.LpVariable,
    difference: pulp.LpVariable,
    *,
    prefix: str,
) -> None:
    """Modela exactamente difference = left XOR right."""

    _add_constraint(
        problem,
        difference >= left - right,
        f"{prefix}_lower_left",
    )

    _add_constraint(
        problem,
        difference >= right - left,
        f"{prefix}_lower_right",
    )

    _add_constraint(
        problem,
        difference <= left + right,
        f"{prefix}_upper_sum",
    )

    _add_constraint(
        problem,
        difference <= 2 - left - right,
        f"{prefix}_upper_complement",
    )


def _add_or5_constraints(
    problem: pulp.LpProblem,
    inputs: tuple[
        pulp.LpVariable,
        ...,
    ],
    output: pulp.LpVariable,
    *,
    prefix: str,
) -> None:
    """Modela exactamente el OR de cinco variables binarias."""

    if len(inputs) != 5:
        raise ValueError(
            "inputs debe contener exactamente cinco variables."
        )

    for index, variable in enumerate(
        inputs
    ):
        _add_constraint(
            problem,
            output >= variable,
            f"{prefix}_lower_{index}",
        )

    _add_constraint(
        problem,
        output <= pulp.lpSum(inputs),
        f"{prefix}_upper",
    )


def _merge_problem_constraints(
    parent: pulp.LpProblem,
    child: pulp.LpProblem,
    *,
    prefix: str,
) -> None:
    """Copia las restricciones de un problema hijo."""

    constraint_items = tuple(
        child.constraints.items()
    )

    for (
        original_name,
        original_constraint,
    ) in constraint_items:
        _add_constraint(
            parent,
            copy(original_constraint),
            f"{prefix}_{original_name}",
        )


class FinalCasePairedMILPModel:
    """
    Modelo MILP emparejado del caso final.

    El modelo se construye mediante build_model() y la función
    objetivo se establece mediante set_active_sbox_objective().
    """

    def __init__(
        self,
        *,
        z: int,
        security_level: int,
        domain_id: int,
        name: str | None = None,
    ) -> None:
        (
            validated_z,
            validated_security,
            validated_domain,
            rounds,
        ) = _validate_parameters(
            z=z,
            security_level=security_level,
            domain_id=domain_id,
        )

        normalized_name = _validate_name(
            name,
            z=validated_z,
            security_level=validated_security,
            domain_id=validated_domain,
        )

        self.z = validated_z
        self.security_level = validated_security
        self.domain_id = validated_domain
        self.number_of_rounds = rounds
        self.name = normalized_name

        self.left = FinalCaseRoundsMILPModel(
            z=self.z,
            security_level=self.security_level,
            domain_id=self.domain_id,
            name=f"{self.name}_left",
        )

        self.right = FinalCaseRoundsMILPModel(
            z=self.z,
            security_level=self.security_level,
            domain_id=self.domain_id,
            name=f"{self.name}_right",
        )

        self.positions = self.left.positions

        if self.positions != self.right.positions:
            raise RuntimeError(
                "Las ramas no comparten el mismo orden de estado."
            )

        self.problem = pulp.LpProblem(
            self.name,
            pulp.LpMinimize,
        )

        self.initial_difference: dict[
            StatePosition,
            pulp.LpVariable,
        ] = {}

        self.chi_input_difference: dict[
            ChiInputDifferenceKey,
            pulp.LpVariable,
        ] = {}

        self.active_sboxes: dict[
            ActiveSBoxKey,
            pulp.LpVariable,
        ] = {}

        self._model_built = False
        self._objective_added = False

        self._fixed_left_input: (
            tuple[int, ...] | None
        ) = None

        self._fixed_right_input: (
            tuple[int, ...] | None
        ) = None

    def _require_built(
        self,
    ) -> None:
        """Exige que el modelo haya sido construido."""

        if not self._model_built:
            raise RuntimeError(
                "Debe ejecutarse build_model() primero."
            )

    def _build_branches(
        self,
    ) -> None:
        """Construye las dos ramas funcionales."""

        self.left.build_model()
        self.right.build_model()

    def _merge_branches(
        self,
    ) -> None:
        """Incorpora las restricciones de ambas ramas."""

        _merge_problem_constraints(
            self.problem,
            self.left.problem,
            prefix=f"{self.name}_merge_left",
        )

        _merge_problem_constraints(
            self.problem,
            self.right.problem,
            prefix=f"{self.name}_merge_right",
        )

    def _add_initial_difference_constraints(
        self,
    ) -> None:
        """Crea la diferencia inicial y exige que sea no nula."""

        for position in self.positions:
            x, y, k = position

            difference = create_binary_variable(
                self.problem,
                (
                    f"{self.name}_delta_input"
                    f"_x{x}_y{y}_k{k}"
                ),
            )

            self.initial_difference[
                position
            ] = difference

            _add_xor2_constraints(
                self.problem,
                self.left.initial_state_variable(
                    x,
                    y,
                    k,
                ),
                self.right.initial_state_variable(
                    x,
                    y,
                    k,
                ),
                difference,
                prefix=(
                    f"{self.name}_xor_input"
                    f"_x{x}_y{y}_k{k}"
                ),
            )

        _add_constraint(
            self.problem,
            pulp.lpSum(
                self.initial_difference.values()
            ) >= 1,
            (
                f"{self.name}"
                "_nonzero_initial_difference"
            ),
        )

    def _add_chi_input_difference_constraints(
        self,
    ) -> None:
        """Crea las diferencias de entrada a cada Chi*."""

        for round_index in range(
            self.number_of_rounds
        ):
            left_round = self.left.round_model(
                round_index
            )

            right_round = self.right.round_model(
                round_index
            )

            for position in self.positions:
                x, y, k = position

                difference = create_binary_variable(
                    self.problem,
                    (
                        f"{self.name}_delta_chi_input"
                        f"_r{round_index}"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

                key = (
                    round_index,
                    x,
                    y,
                    k,
                )

                self.chi_input_difference[
                    key
                ] = difference

                _add_xor2_constraints(
                    self.problem,
                    left_round.linear_output_variable(
                        x,
                        y,
                        k,
                    ),
                    right_round.linear_output_variable(
                        x,
                        y,
                        k,
                    ),
                    difference,
                    prefix=(
                        f"{self.name}_xor_chi_input"
                        f"_r{round_index}"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

    def _add_active_sbox_constraints(
        self,
    ) -> None:
        """Crea el indicador activo de cada S-box."""

        for round_index in range(
            self.number_of_rounds
        ):
            for y in range(5):
                for k in range(
                    self.z
                ):
                    active = create_binary_variable(
                        self.problem,
                        (
                            f"{self.name}_active"
                            f"_r{round_index}"
                            f"_y{y}_k{k}"
                        ),
                    )

                    key = (
                        round_index,
                        y,
                        k,
                    )

                    self.active_sboxes[
                        key
                    ] = active

                    difference_slice = tuple(
                        self.chi_input_difference[
                            (
                                round_index,
                                x,
                                y,
                                k,
                            )
                        ]
                        for x in range(5)
                    )

                    _add_or5_constraints(
                        self.problem,
                        difference_slice,
                        active,
                        prefix=(
                            f"{self.name}_active_or"
                            f"_r{round_index}"
                            f"_y{y}_k{k}"
                        ),
                    )

    def _validate_unique_names(
        self,
    ) -> None:
        """Comprueba que no existan nombres duplicados."""

        variable_names = [
            variable.name
            for variable
            in self.problem.variables()
        ]

        if len(variable_names) != len(
            set(variable_names)
        ):
            raise RuntimeError(
                "El modelo contiene nombres de variable duplicados."
            )

        constraint_names = list(
            self.problem.constraints.keys()
        )

        if len(constraint_names) != len(
            set(constraint_names)
        ):
            raise RuntimeError(
                "El modelo contiene nombres de restricción duplicados."
            )

    def build_model(
        self,
    ) -> None:
        """Construye el modelo completo de forma idempotente."""

        if self._model_built:
            return

        self._build_branches()
        self._merge_branches()

        self._add_initial_difference_constraints()
        self._add_chi_input_difference_constraints()
        self._add_active_sbox_constraints()

        self._validate_unique_names()

        self._model_built = True

    def set_active_sbox_objective(
        self,
    ) -> None:
        """Minimiza la suma total de S-boxes activas."""

        self._require_built()

        if self._objective_added:
            return

        self.problem.setObjective(
            pulp.lpSum(
                self.active_sboxes.values()
            )
        )

        self._objective_added = True

    def branch_model(
        self,
        branch: str,
    ) -> FinalCaseRoundsMILPModel:
        """Devuelve la rama izquierda o derecha."""

        self._require_built()

        normalized = _validate_branch(
            branch
        )

        return (
            self.left
            if normalized == "left"
            else self.right
        )

    def _fix_branch_input(
        self,
        *,
        branch: str,
        values: object,
    ) -> None:
        """Fija la entrada de una rama de forma idempotente."""

        self._require_built()

        normalized_branch = _validate_branch(
            branch
        )

        normalized_values = _validate_state_values(
            values,
            z=self.z,
        )

        signature = tuple(
            int(value)
            for value
            in normalized_values.reshape(
                -1
            )
        )

        if normalized_branch == "left":
            previous_signature = (
                self._fixed_left_input
            )

            selected_branch = self.left
            prefix = "fix_left_input"
        else:
            previous_signature = (
                self._fixed_right_input
            )

            selected_branch = self.right
            prefix = "fix_right_input"

        if previous_signature is not None:
            if previous_signature == signature:
                return

            raise RuntimeError(
                f"La entrada {normalized_branch} "
                "ya fue fijada con valores diferentes."
            )

        for position in self.positions:
            x, y, k = position

            _add_constraint(
                self.problem,
                (
                    selected_branch.initial_state_variable(
                        x,
                        y,
                        k,
                    )
                    == int(
                        normalized_values[
                            position
                        ]
                    )
                ),
                (
                    f"{self.name}_{prefix}"
                    f"_x{x}_y{y}_k{k}"
                ),
            )

        if normalized_branch == "left":
            self._fixed_left_input = signature
        else:
            self._fixed_right_input = signature

    def fix_left_input_values(
        self,
        values: object,
    ) -> None:
        """Fija el estado de entrada de la rama izquierda."""

        self._fix_branch_input(
            branch="left",
            values=values,
        )

    def fix_right_input_values(
        self,
        values: object,
    ) -> None:
        """Fija el estado de entrada de la rama derecha."""

        self._fix_branch_input(
            branch="right",
            values=values,
        )

    def solve(
        self,
        solver: pulp.LpSolver,
    ) -> str:
        """Resuelve el modelo con el solver suministrado."""

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

    def _validate_round_index(
        self,
        round_index: object,
    ) -> int:
        """Valida el índice de ronda."""

        validated = _validate_integer(
            round_index,
            parameter_name="round_index",
        )

        if validated not in range(
            self.number_of_rounds
        ):
            raise ValueError(
                "round_index debe encontrarse entre 0 y "
                f"{self.number_of_rounds - 1}."
            )

        return validated

    def _validate_position(
        self,
        *,
        x: object,
        y: object,
        k: object,
    ) -> StatePosition:
        """Valida una posición del estado."""

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

        if validated_k not in range(
            self.z
        ):
            raise ValueError(
                "k debe encontrarse entre 0 y "
                f"{self.z - 1}."
            )

        return (
            validated_x,
            validated_y,
            validated_k,
        )

    def initial_difference_variable(
        self,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve una variable de diferencia inicial."""

        self._require_built()

        position = self._validate_position(
            x=x,
            y=y,
            k=k,
        )

        return self.initial_difference[
            position
        ]

    def chi_input_difference_variable(
        self,
        round_index: int,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve una diferencia de entrada a Chi*."""

        self._require_built()

        validated_round = (
            self._validate_round_index(
                round_index
            )
        )

        position = self._validate_position(
            x=x,
            y=y,
            k=k,
        )

        return self.chi_input_difference[
            (
                validated_round,
                *position,
            )
        ]

    def active_sbox_variable(
        self,
        round_index: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve el indicador de una S-box activa."""

        self._require_built()

        validated_round = (
            self._validate_round_index(
                round_index
            )
        )

        validated_y = _validate_integer(
            y,
            parameter_name="y",
        )

        validated_k = _validate_integer(
            k,
            parameter_name="k",
        )

        if validated_y not in range(5):
            raise ValueError(
                "y debe encontrarse entre 0 y 4."
            )

        if validated_k not in range(
            self.z
        ):
            raise ValueError(
                "k debe encontrarse entre 0 y "
                f"{self.z - 1}."
            )

        return self.active_sboxes[
            (
                validated_round,
                validated_y,
                validated_k,
            )
        ]

    @staticmethod
    def _variable_bit(
        variable: pulp.LpVariable,
        *,
        tolerance: float,
    ) -> int:
        """Recupera el valor binario de una variable."""

        value = variable.value()

        if value is None:
            raise RuntimeError(
                "El modelo todavía no tiene una solución."
            )

        return int(
            value >= tolerance
        )

    def _branch_state_values(
        self,
        *,
        branch: str,
        final: bool,
        tolerance: object,
    ) -> NDArray[np.int64]:
        """Recupera la frontera inicial o final de una rama."""

        self._require_built()

        normalized_branch = _validate_branch(
            branch
        )

        normalized_tolerance = (
            _validate_tolerance(
                tolerance
            )
        )

        selected = (
            self.left
            if normalized_branch == "left"
            else self.right
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
            if final:
                variable = (
                    selected.final_state_variable(
                        *position
                    )
                )
            else:
                variable = (
                    selected.initial_state_variable(
                        *position
                    )
                )

            result[position] = (
                self._variable_bit(
                    variable,
                    tolerance=normalized_tolerance,
                )
            )

        return result

    def branch_initial_state_values(
        self,
        branch: str,
        *,
        tolerance: float = 0.5,
    ) -> NDArray[np.int64]:
        """Recupera la entrada resuelta de una rama."""

        return self._branch_state_values(
            branch=branch,
            final=False,
            tolerance=tolerance,
        )

    def branch_final_state_values(
        self,
        branch: str,
        *,
        tolerance: float = 0.5,
    ) -> NDArray[np.int64]:
        """Recupera la salida resuelta de una rama."""

        return self._branch_state_values(
            branch=branch,
            final=True,
            tolerance=tolerance,
        )

    def initial_difference_values(
        self,
        *,
        tolerance: float = 0.5,
    ) -> NDArray[np.int64]:
        """Recupera la diferencia inicial."""

        self._require_built()

        normalized_tolerance = (
            _validate_tolerance(
                tolerance
            )
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
            result[position] = (
                self._variable_bit(
                    self.initial_difference[
                        position
                    ],
                    tolerance=normalized_tolerance,
                )
            )

        return result

    def chi_input_difference_values(
        self,
        round_index: int,
        *,
        tolerance: float = 0.5,
    ) -> NDArray[np.int64]:
        """Recupera la diferencia de entrada a Chi*."""

        self._require_built()

        validated_round = (
            self._validate_round_index(
                round_index
            )
        )

        normalized_tolerance = (
            _validate_tolerance(
                tolerance
            )
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
            x, y, k = position

            result[position] = (
                self._variable_bit(
                    self.chi_input_difference[
                        (
                            validated_round,
                            x,
                            y,
                            k,
                        )
                    ],
                    tolerance=normalized_tolerance,
                )
            )

        return result

    def active_sbox_values(
        self,
        round_index: int,
        *,
        tolerance: float = 0.5,
    ) -> NDArray[np.int64]:
        """Recupera el patrón activo de una ronda."""

        self._require_built()

        validated_round = (
            self._validate_round_index(
                round_index
            )
        )

        normalized_tolerance = (
            _validate_tolerance(
                tolerance
            )
        )

        result = np.zeros(
            (
                5,
                self.z,
            ),
            dtype=np.int64,
        )

        for y in range(5):
            for k in range(
                self.z
            ):
                result[
                    y,
                    k,
                ] = self._variable_bit(
                    self.active_sboxes[
                        (
                            validated_round,
                            y,
                            k,
                        )
                    ],
                    tolerance=normalized_tolerance,
                )

        return result

    def active_sbox_counts(
        self,
        *,
        tolerance: float = 0.5,
    ) -> tuple[int, ...]:
        """Devuelve la cantidad activa por ronda."""

        return tuple(
            int(
                self.active_sbox_values(
                    round_index,
                    tolerance=tolerance,
                ).sum()
            )
            for round_index in range(
                self.number_of_rounds
            )
        )

    def objective_value(
        self,
    ) -> int:
        """Devuelve el valor entero de la función objetivo."""

        self._require_built()

        if not self._objective_added:
            raise RuntimeError(
                "El modelo no tiene función objetivo."
            )

        value = pulp.value(
            self.problem.objective
        )

        if value is None:
            raise RuntimeError(
                "El modelo todavía no tiene una solución."
            )

        return int(
            round(
                float(value)
            )
        )

    def statistics(
        self,
    ) -> FinalCasePairedMILPStatistics:
        """Devuelve las dimensiones del modelo."""

        self._require_built()

        left_statistics = (
            self.left.statistics()
        )

        right_statistics = (
            self.right.statistics()
        )

        return FinalCasePairedMILPStatistics(
            z=self.z,
            security_level=self.security_level,
            domain_id=self.domain_id,
            rounds=self.number_of_rounds,
            state_bits=25 * self.z,
            variables=self.problem.numVariables(),
            constraints=self.problem.numConstraints(),
            left_variables=left_statistics.variables,
            right_variables=right_statistics.variables,
            left_constraints=left_statistics.constraints,
            right_constraints=right_statistics.constraints,
            initial_difference_variables=len(
                self.initial_difference
            ),
            chi_input_difference_variables=len(
                self.chi_input_difference
            ),
            active_sbox_variables=len(
                self.active_sboxes
            ),
        )


__all__ = [
    "FinalCasePairedMILPModel",
    "FinalCasePairedMILPStatistics",
]
