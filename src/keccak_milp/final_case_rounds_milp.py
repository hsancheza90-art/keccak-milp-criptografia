"""
Modelo MILP permanente multirronda del caso final de Keccak.

El modelo compone una, dos o tres instancias de:

    L* -> Chi* -> Iota*

según security_level, y conecta cada salida de ronda con la entrada
de la ronda siguiente.

Los modelos V1 permanecen separados y no son modificados.
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
from keccak_milp.final_case_round_milp import (
    FinalCaseRoundMILPModel,
    StatePosition,
    final_case_state_positions,
)


@dataclass(frozen=True)
class FinalCaseRoundsMILPStatistics:
    """Resumen estructural del modelo multirronda."""

    z: int
    security_level: int
    domain_id: int
    rounds: int
    state_bits: int
    variables: int
    constraints: int
    boundary_constraints: int
    round_variables: tuple[int, ...]
    round_constraints: tuple[int, ...]


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
    """Valida el tamaño de palabra."""

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
) -> tuple[int, int, int, int]:
    """Valida los parámetros del modelo."""

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
    """Valida o genera el nombre del problema."""

    if name is None:
        return (
            "final_case_rounds"
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


class FinalCaseRoundsMILPModel:
    """
    Modelo MILP exacto de una, dos o tres rondas.

    La cantidad de rondas queda determinada por security_level:

        0 -> 1 ronda;
        1 -> 2 rondas;
        2 -> 3 rondas.
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

        self.positions = final_case_state_positions(
            self.z
        )

        self.problem = pulp.LpProblem(
            self.name,
            pulp.LpMinimize,
        )

        self.round_models: list[
            FinalCaseRoundMILPModel
        ] = []

        self.boundary_constraint_names: list[
            str
        ] = []

        self._model_built = False
        self._objective_added = False

        self._fixed_initial_state: (
            tuple[int, ...] | None
        ) = None

        self._fixed_final_state: (
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

    def _build_round_models(
        self,
    ) -> None:
        """Construye los modelos individuales de cada ronda."""

        for round_index in range(
            self.number_of_rounds
        ):
            round_model = FinalCaseRoundMILPModel(
                z=self.z,
                security_level=self.security_level,
                domain_id=self.domain_id,
                round_index=round_index,
                name=(
                    f"{self.name}"
                    f"_round_{round_index}"
                ),
            )

            round_model.build_model()

            self.round_models.append(
                round_model
            )

    def _merge_round_constraints(
        self,
    ) -> None:
        """
        Incorpora las restricciones de cada ronda.

        Se utiliza una copia superficial de cada restricción para
        conservar intacto el modelo individual de ronda.
        """

        for round_index, round_model in enumerate(
            self.round_models
        ):
            constraint_items = tuple(
                round_model.problem.constraints.items()
            )

            for (
                original_name,
                original_constraint,
            ) in constraint_items:
                merged_name = (
                    f"{self.name}"
                    f"_merged_round_{round_index}"
                    f"_{original_name}"
                )

                merged_constraint = copy(
                    original_constraint
                )

                _add_constraint(
                    self.problem,
                    merged_constraint,
                    merged_name,
                )

    def _add_boundary_constraints(
        self,
    ) -> None:
        """Conecta la salida de cada ronda con la entrada siguiente."""

        for round_index in range(
            self.number_of_rounds - 1
        ):
            current_round = self.round_models[
                round_index
            ]

            next_round = self.round_models[
                round_index + 1
            ]

            for position in self.positions:
                x, y, k = position

                name = (
                    f"{self.name}_boundary"
                    f"_r{round_index}"
                    f"_to_r{round_index + 1}"
                    f"_x{x}_y{y}_k{k}"
                )

                _add_constraint(
                    self.problem,
                    (
                        current_round.output_variable(
                            x,
                            y,
                            k,
                        )
                        == next_round.input_variable(
                            x,
                            y,
                            k,
                        )
                    ),
                    name,
                )

                self.boundary_constraint_names.append(
                    name
                )

    def _validate_unique_names(
        self,
    ) -> None:
        """Comprueba que las variables y restricciones sean únicas."""

        variable_names = [
            variable.name
            for variable in self.problem.variables()
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

        self._build_round_models()
        self._merge_round_constraints()
        self._add_boundary_constraints()
        self._validate_unique_names()

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

    def round_model(
        self,
        round_index: int,
    ) -> FinalCaseRoundMILPModel:
        """Devuelve el modelo correspondiente a una ronda."""

        self._require_built()

        validated_index = _validate_integer(
            round_index,
            parameter_name="round_index",
        )

        if validated_index not in range(
            self.number_of_rounds
        ):
            raise ValueError(
                "round_index debe encontrarse entre 0 y "
                f"{self.number_of_rounds - 1}."
            )

        return self.round_models[
            validated_index
        ]

    def _state_variable(
        self,
        *,
        final: bool,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Recupera una variable de la frontera inicial o final."""

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

        if validated_k not in range(
            self.z
        ):
            raise ValueError(
                "k debe encontrarse entre 0 y "
                f"{self.z - 1}."
            )

        selected_round = (
            self.round_models[-1]
            if final
            else self.round_models[0]
        )

        if final:
            return selected_round.output_variable(
                validated_x,
                validated_y,
                validated_k,
            )

        return selected_round.input_variable(
            validated_x,
            validated_y,
            validated_k,
        )

    def initial_state_variable(
        self,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve una variable del estado inicial."""

        return self._state_variable(
            final=False,
            x=x,
            y=y,
            k=k,
        )

    def final_state_variable(
        self,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve una variable del estado final."""

        return self._state_variable(
            final=True,
            x=x,
            y=y,
            k=k,
        )

    def _fix_state(
        self,
        *,
        values: NDArray[np.int64],
        final: bool,
        prefix: str,
    ) -> None:
        """Fija una frontera completa."""

        for position in self.positions:
            x, y, k = position

            variable = self._state_variable(
                final=final,
                x=x,
                y=y,
                k=k,
            )

            _add_constraint(
                self.problem,
                variable == int(
                    values[position]
                ),
                (
                    f"{self.name}_{prefix}"
                    f"_x{x}_y{y}_k{k}"
                ),
            )

    def fix_initial_state_values(
        self,
        values: object,
    ) -> None:
        """Fija el estado inicial de forma idempotente."""

        self._require_built()

        normalized = _validate_state_values(
            values,
            z=self.z,
        )

        signature = tuple(
            int(value)
            for value in normalized.reshape(
                -1
            )
        )

        if self._fixed_initial_state is not None:
            if (
                self._fixed_initial_state
                == signature
            ):
                return

            raise RuntimeError(
                "El estado inicial ya fue fijado "
                "con valores diferentes."
            )

        self._fix_state(
            values=normalized,
            final=False,
            prefix="fix_initial",
        )

        self._fixed_initial_state = signature

    def fix_final_state_values(
        self,
        values: object,
    ) -> None:
        """Fija el estado final de forma idempotente."""

        self._require_built()

        normalized = _validate_state_values(
            values,
            z=self.z,
        )

        signature = tuple(
            int(value)
            for value in normalized.reshape(
                -1
            )
        )

        if self._fixed_final_state is not None:
            if (
                self._fixed_final_state
                == signature
            ):
                return

            raise RuntimeError(
                "El estado final ya fue fijado "
                "con valores diferentes."
            )

        self._fix_state(
            values=normalized,
            final=True,
            prefix="fix_final",
        )

        self._fixed_final_state = signature

    def solve(
        self,
        solver: pulp.LpSolver,
    ) -> str:
        """Resuelve el problema con el solver suministrado."""

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

    def final_state_values(
        self,
        *,
        tolerance: float = 0.5,
    ) -> NDArray[np.int64]:
        """Recupera el estado final después de resolver."""

        self._require_built()

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

        normalized_tolerance = float(
            tolerance
        )

        if not (
            0.0
            < normalized_tolerance
            < 1.0
        ):
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
            variable = self.final_state_variable(
                *position
            )

            value = variable.value()

            if value is None:
                raise RuntimeError(
                    "El modelo todavía no tiene una solución."
                )

            result[position] = int(
                value
                >= normalized_tolerance
            )

        return result

    def statistics(
        self,
    ) -> FinalCaseRoundsMILPStatistics:
        """Devuelve las dimensiones estructurales."""

        self._require_built()

        round_statistics = tuple(
            round_model.statistics()
            for round_model in self.round_models
        )

        return FinalCaseRoundsMILPStatistics(
            z=self.z,
            security_level=self.security_level,
            domain_id=self.domain_id,
            rounds=self.number_of_rounds,
            state_bits=25 * self.z,
            variables=self.problem.numVariables(),
            constraints=self.problem.numConstraints(),
            boundary_constraints=len(
                self.boundary_constraint_names
            ),
            round_variables=tuple(
                statistics.variables
                for statistics
                in round_statistics
            ),
            round_constraints=tuple(
                statistics.constraints
                for statistics
                in round_statistics
            ),
        )


__all__ = [
    "FinalCaseRoundsMILPModel",
    "FinalCaseRoundsMILPStatistics",
]
