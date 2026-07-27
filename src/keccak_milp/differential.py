"""
Modelo diferencial exacto basado en dos ejecuciones de Keccak.

Se construyen dos instancias independientes de KeccakMILPModel:

- ejecución izquierda;
- ejecución derecha.

La diferencia XOR entre sus estados de frontera se representa mediante:

    L[r,x,y,k] + R[r,x,y,k]
    =
    Delta[r,x,y,k] + 2 Q[r,x,y,k]
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
import pulp

from keccak_milp.config import ExperimentConfig
from keccak_milp.model import KeccakMILPModel
from keccak_milp.solver import build_solver


DifferenceBitIndex: TypeAlias = tuple[int, int, int, int]
ExecutionSide: TypeAlias = Literal["left", "right"]


class PairedKeccakMILPModel:
    """
    Modelo MILP de dos ejecuciones exactas de Keccak.

    Las diferencias XOR se definen sobre todos los estados de
    frontera:

        Delta_r = Left_r XOR Right_r
    """

    def __init__(
        self,
        config: ExperimentConfig,
        name: str | None = None,
    ) -> None:
        """Inicializa las dos ejecuciones y el problema combinado."""
        self.config = config

        problem_name = name or (
            f"paired_keccak_milp"
            f"_z{config.z}_r{config.rounds}"
        )

        self.problem = pulp.LpProblem(
            name=problem_name,
            sense=pulp.LpMinimize,
        )

        self.left = KeccakMILPModel(
            config,
            name=f"{problem_name}_left",
        )

        self.right = KeccakMILPModel(
            config,
            name=f"{problem_name}_right",
        )

        self.delta_state: dict[
            DifferenceBitIndex,
            pulp.LpVariable,
        ] = {}

        self.delta_q: dict[
            DifferenceBitIndex,
            pulp.LpVariable,
        ] = {}

        self._paired_model_built = False
        self._nonzero_input_difference_added = False
        self._objective_added = False

    @staticmethod
    def _prefix_model_variables(
        model: KeccakMILPModel,
        prefix: str,
    ) -> None:
        """Agrega un prefijo a las variables de una ejecución."""
        for variable in model.problem.variables():
            if not variable.name.startswith(prefix):
                variable.name = (
                    f"{prefix}{variable.name}"
                )

    def _merge_model_constraints(
        self,
        model: KeccakMILPModel,
        prefix: str,
    ) -> None:
        """Incorpora las restricciones de una ejecución."""
        self._prefix_model_variables(
            model=model,
            prefix=prefix,
        )

        for constraint_name, constraint in (
            model.problem.constraints.items()
        ):
            self.problem += (
                constraint,
                f"{prefix}{constraint_name}",
            )

    def _create_difference_variables(self) -> None:
        """Crea variables de diferencia para cada frontera."""
        for boundary_index in range(
            self.config.rounds + 1
        ):
            for x in range(5):
                for y in range(5):
                    for k in range(self.config.z):
                        index = (
                            boundary_index,
                            x,
                            y,
                            k,
                        )

                        self.delta_state[index] = (
                            pulp.LpVariable(
                                name=(
                                    f"delta_state"
                                    f"_r{boundary_index}"
                                    f"_x{x}_y{y}_k{k}"
                                ),
                                lowBound=0,
                                upBound=1,
                                cat=pulp.LpBinary,
                            )
                        )

                        self.delta_q[index] = (
                            pulp.LpVariable(
                                name=(
                                    f"delta_q"
                                    f"_r{boundary_index}"
                                    f"_x{x}_y{y}_k{k}"
                                ),
                                lowBound=0,
                                upBound=1,
                                cat=pulp.LpBinary,
                            )
                        )

    def _add_difference_constraints(self) -> None:
        """
        Agrega las ecuaciones XOR entre ambas ejecuciones.

        Para cada bit:

            left + right = delta + 2 q
        """
        for boundary_index in range(
            self.config.rounds + 1
        ):
            for x in range(5):
                for y in range(5):
                    for k in range(self.config.z):
                        index = (
                            boundary_index,
                            x,
                            y,
                            k,
                        )

                        left_variable = (
                            self.left.state_variable(
                                round_index=boundary_index,
                                x=x,
                                y=y,
                                k=k,
                            )
                        )

                        right_variable = (
                            self.right.state_variable(
                                round_index=boundary_index,
                                x=x,
                                y=y,
                                k=k,
                            )
                        )

                        delta_variable = (
                            self.delta_state[index]
                        )

                        parity_variable = (
                            self.delta_q[index]
                        )

                        self.problem += (
                            left_variable + right_variable
                            == (
                                delta_variable
                                + 2 * parity_variable
                            ),
                            (
                                f"delta_state_xor"
                                f"_r{boundary_index}"
                                f"_x{x}_y{y}_k{k}"
                            ),
                        )

    def build_paired_model(self) -> None:
        """
        Construye las dos ejecuciones y sus diferencias.

        La operación es idempotente.
        """
        if self._paired_model_built:
            return

        self.left.add_all_rounds()
        self.right.add_all_rounds()

        self._merge_model_constraints(
            model=self.left,
            prefix="left_",
        )

        self._merge_model_constraints(
            model=self.right,
            prefix="right_",
        )

        self._create_difference_variables()
        self._add_difference_constraints()

        self._paired_model_built = True

    def _require_built(self) -> None:
        """Comprueba que el modelo emparejado esté construido."""
        if not self._paired_model_built:
            raise RuntimeError(
                "Debe ejecutarse build_paired_model() primero."
            )

    def difference_variable(
        self,
        boundary_index: int,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve una variable de diferencia de frontera."""
        self._require_built()

        index = (
            boundary_index,
            x,
            y,
            k,
        )

        if index not in self.delta_state:
            raise KeyError(
                "La variable de diferencia solicitada "
                "no existe."
            )

        return self.delta_state[index]

    def difference_state_values(
        self,
        boundary_index: int,
        tolerance: float = 0.5,
    ) -> np.ndarray:
        """Recupera un estado de diferencias resuelto."""
        self._require_built()

        if boundary_index not in range(
            self.config.rounds + 1
        ):
            raise ValueError(
                "El estado de frontera debe encontrarse "
                f"entre 0 y {self.config.rounds}."
            )

        output = np.zeros(
            (5, 5, self.config.z),
            dtype=np.int64,
        )

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    variable = self.difference_variable(
                        boundary_index,
                        x,
                        y,
                        k,
                    )

                    value = variable.value()

                    if value is None:
                        raise RuntimeError(
                            "El modelo debe resolverse antes "
                            "de recuperar las diferencias."
                        )

                    output[x, y, k] = int(
                        value > tolerance
                    )

        return output

    def concrete_state_values(
        self,
        side: ExecutionSide,
        boundary_index: int,
        tolerance: float = 0.5,
    ) -> np.ndarray:
        """Recupera un estado concreto de una ejecución."""
        self._require_built()

        if side == "left":
            model = self.left
        elif side == "right":
            model = self.right
        else:
            raise ValueError(
                "El lado debe ser 'left' o 'right'."
            )

        if boundary_index not in range(
            self.config.rounds + 1
        ):
            raise ValueError(
                "Índice de frontera inválido."
            )

        output = np.zeros(
            (5, 5, self.config.z),
            dtype=np.int64,
        )

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    variable = model.state_variable(
                        boundary_index,
                        x,
                        y,
                        k,
                    )

                    value = variable.value()

                    if value is None:
                        raise RuntimeError(
                            "El modelo debe resolverse antes "
                            "de recuperar el estado."
                        )

                    output[x, y, k] = int(
                        value > tolerance
                    )

        return output

    def add_nonzero_input_difference_constraint(
        self,
    ) -> None:
        """Impide que la diferencia inicial sea nula."""
        self._require_built()

        if self._nonzero_input_difference_added:
            return

        self.problem += (
            pulp.lpSum(
                self.delta_state[
                    0,
                    x,
                    y,
                    k,
                ]
                for x in range(5)
                for y in range(5)
                for k in range(self.config.z)
            )
            >= 1,
            "nonzero_input_difference",
        )

        self._nonzero_input_difference_added = True

    def set_boundary_difference_weight_objective(
        self,
        boundary_index: int,
    ) -> None:
        """Minimiza el peso de una diferencia de frontera."""
        self._require_built()

        if boundary_index not in range(
            self.config.rounds + 1
        ):
            raise ValueError(
                "El estado de frontera debe encontrarse "
                f"entre 0 y {self.config.rounds}."
            )

        objective = pulp.lpSum(
            self.delta_state[
                boundary_index,
                x,
                y,
                k,
            ]
            for x in range(5)
            for y in range(5)
            for k in range(self.config.z)
        )

        self.problem.setObjective(objective)
        self._objective_added = True

    def set_input_output_difference_objective(
        self,
    ) -> None:
        """Minimiza HW(Delta A_0) + HW(Delta A_R)."""
        self._require_built()

        final_boundary = self.config.rounds

        objective = pulp.lpSum(
            [
                self.delta_state[
                    0,
                    x,
                    y,
                    k,
                ]
                for x in range(5)
                for y in range(5)
                for k in range(self.config.z)
            ]
            + [
                self.delta_state[
                    final_boundary,
                    x,
                    y,
                    k,
                ]
                for x in range(5)
                for y in range(5)
                for k in range(self.config.z)
            ]
        )

        self.problem.setObjective(objective)
        self._objective_added = True

    def solve(self) -> str:
        """Resuelve el problema emparejado."""
        if not self._objective_added:
            raise RuntimeError(
                "El modelo no tiene función objetivo."
            )

        solver = build_solver(self.config)
        self.problem.solve(solver)

        return pulp.LpStatus[
            self.problem.status
        ]

    def objective_value(self) -> float:
        """Recupera el objetivo después de resolver."""
        if (
            not self._objective_added
            or self.problem.objective is None
        ):
            raise RuntimeError(
                "El modelo no tiene función objetivo."
            )

        if self.problem.status == pulp.LpStatusNotSolved:
            raise RuntimeError(
                "El modelo debe resolverse antes de "
                "recuperar el objetivo."
            )

        result = pulp.value(
            self.problem.objective
        )

        if result is None:
            raise RuntimeError(
                "No fue posible recuperar el objetivo."
            )

        return float(result)

    def declared_variable_count(self) -> int:
        """Cuenta variables declaradas por los tres bloques."""
        return (
            self.left.declared_variable_count()
            + self.right.declared_variable_count()
            + len(self.delta_state)
            + len(self.delta_q)
        )

    def attached_variable_count(self) -> int:
        """Cuenta las variables conectadas al problema combinado."""
        return len(
            self.problem.variables()
        )

    def constraint_count(self) -> int:
        """Cuenta las restricciones del problema combinado."""
        return len(
            self.problem.constraints
        )