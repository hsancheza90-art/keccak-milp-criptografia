"""
Esqueleto del modelo MILP para Keccak reducido.

En esta etapa se construyen únicamente:

- el problema de optimización;
- las variables binarias de los estados de frontera;
- una restricción que impide la diferencia inicial nula;
- una función objetivo provisional para validar el modelo.

Las capas theta, rho, pi y chi se incorporarán posteriormente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import pulp

from keccak_milp.config import ExperimentConfig
from keccak_milp.layers import (
    rho_pi_destination,
    round_constant,
)
from keccak_milp.solver import build_solver


# ============================================================
# TIPOS
# ============================================================

StateIndex: TypeAlias = tuple[int, int, int, int]

ThetaColumnIndex: TypeAlias = tuple[int, int, int]
ThetaBitIndex: TypeAlias = tuple[int, int, int, int]

RhoPiBitIndex: TypeAlias = tuple[int, int, int, int]
ChiBitIndex: TypeAlias = tuple[int, int, int, int]

@dataclass(frozen=True)
class ModelStatistics:
    """
    Resumen estructural del modelo MILP.

    declared_variables:
        Variables creadas y almacenadas por la implementación.

    attached_variables:
        Variables que PuLP ya reconoce porque participan en el
        objetivo o en alguna restricción.

    total_constraints:
        Restricciones actualmente agregadas al problema.
    """

    z: int
    rounds: int
    state_bits: int
    boundary_states: int
    declared_variables: int
    attached_variables: int
    total_constraints: int


# ============================================================
# MODELO PRINCIPAL
# ============================================================

class KeccakMILPModel:
    """
    Modelo MILP progresivo para versiones reducidas de Keccak.

    Convención de las variables de estado
    --------------------------------------
    state[(r, x, y, k)]

    donde:

        r = estado de frontera entre rondas
        x, y ∈ {0, 1, 2, 3, 4}
        k ∈ {0, ..., z - 1}

    Para R rondas se crean R + 1 estados de frontera:

        estado 0: entrada al modelo;
        estado 1: salida de la primera ronda;
        ...
        estado R: salida de la última ronda.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        name: str | None = None,
    ) -> None:
        """
        Inicializa el problema y crea las variables de estado.

        Parameters
        ----------
        config:
            Configuración del experimento.

        name:
            Nombre opcional del problema MILP.
        """

        self.config = config

        problem_name = name or (
            f"keccak_milp_z{config.z}_r{config.rounds}"
        )

        self.problem = pulp.LpProblem(
            name=problem_name,
            sense=pulp.LpMinimize,
        )

        self.state: dict[StateIndex, pulp.LpVariable] = {}

                # Variables de la capa theta.
        self.theta_c: dict[
            ThetaColumnIndex,
            pulp.LpVariable,
        ] = {}

        self.theta_qc: dict[
            ThetaColumnIndex,
            pulp.LpVariable,
        ] = {}

        self.theta_d: dict[
            ThetaColumnIndex,
            pulp.LpVariable,
        ] = {}

        self.theta_qd: dict[
            ThetaColumnIndex,
            pulp.LpVariable,
        ] = {}

        self.theta_output: dict[
            ThetaBitIndex,
            pulp.LpVariable,
        ] = {}

        self.theta_qt: dict[
            ThetaBitIndex,
            pulp.LpVariable,
        ] = {}

        # Variables de salida después de rho y pi.
        self.rho_pi_output: dict[
            RhoPiBitIndex,
            pulp.LpVariable,
        ] = {}

        # Variables de la capa chi.
        self.chi_and: dict[
            ChiBitIndex,
            pulp.LpVariable,
        ] = {}

        self.chi_output: dict[
            ChiBitIndex,
            pulp.LpVariable,
        ] = {}

        self.chi_q: dict[
            ChiBitIndex,
            pulp.LpVariable,
        ] = {}

        # Control de las rondas rho-pi ya agregadas.
        self._rho_pi_rounds_added: set[int] = set()

        # Control de las rondas chi ya agregadas.
        self._chi_rounds_added: set[int] = set()

        # Control de las rondas iota ya agregadas.
        self._iota_rounds_added: set[int] = set()

        # Control de las rondas theta ya agregadas.
        self._theta_rounds_added: set[int] = set()

        # Control de rondas completas ya agregadas.
        self._rounds_added: set[int] = set()

        self._nonzero_input_added = False
        self._objective_added = False

        self._create_state_variables()

    # ========================================================
    # CREACIÓN DE VARIABLES
    # ========================================================
    # ========================================================
    # VARIABLES DE RHO + PI
    # ========================================================

    def _create_rho_pi_variables(
        self,
        round_index: int,
    ) -> None:
        """
        Crea las variables binarias de salida de rho y pi.

        Por cada ronda se crean:

            25 × z

        variables binarias.
        """

        if round_index not in range(self.config.rounds):
            raise ValueError(
                "La ronda rho-pi debe encontrarse entre 0 y "
                f"{self.config.rounds - 1}."
            )

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    index = (
                        round_index,
                        x,
                        y,
                        k,
                    )

                    self.rho_pi_output[index] = (
                        self._create_binary_variable(
                            name=(
                                f"rho_pi_r{round_index}"
                                f"_x{x}_y{y}_k{k}"
                            )
                        )
                    )

    def _create_binary_variable(
        self,
        name: str,
    ) -> pulp.LpVariable:
        """
        Crea y adjunta una variable binaria al problema.

        Se utiliza `add_variable`, disponible en las versiones recientes
        de PuLP, para evitar la creación de variables desconectadas.
        """

        if hasattr(self.problem, "add_variable"):
            return self.problem.add_variable(
                name=name,
                lowBound=0,
                upBound=1,
                cat=pulp.LpBinary,
            )

        # Compatibilidad con versiones anteriores de PuLP.
        variable = pulp.LpVariable(
            name=name,
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary,
        )

        # Restricción redundante que garantiza que la variable quede
        # asociada al modelo en versiones antiguas.
        self.problem += variable >= 0, f"attach_{name}"

        return variable

    def _create_integer_variable(
        self,
        name: str,
        low_bound: int,
        up_bound: int,
    ) -> pulp.LpVariable:
        """
        Crea una variable entera acotada y la registra en el problema.
        """

        if low_bound > up_bound:
            raise ValueError(
                "El límite inferior no puede superar al superior."
            )

        if hasattr(self.problem, "add_variable"):
            return self.problem.add_variable(
                name=name,
                lowBound=low_bound,
                upBound=up_bound,
                cat=pulp.LpInteger,
            )

        return pulp.LpVariable(
            name=name,
            lowBound=low_bound,
            upBound=up_bound,
            cat=pulp.LpInteger,
        )

    def _create_state_variables(self) -> None:
        """
        Crea las variables binarias de todos los estados de frontera.

        Cantidad esperada:

            (rounds + 1) × 5 × 5 × z
        """

        for round_index in range(self.config.rounds + 1):
            for x in range(5):
                for y in range(5):
                    for k in range(self.config.z):
                        variable_name = (
                            f"a_r{round_index}"
                            f"_x{x}"
                            f"_y{y}"
                            f"_k{k}"
                        )

                        self.state[
                            round_index,
                            x,
                            y,
                            k,
                        ] = self._create_binary_variable(
                            variable_name
                        )

    def _create_chi_variables(
        self,
        round_index: int,
    ) -> None:
        """
        Crea las variables de la capa chi para una ronda.

        Por cada bit se crean:

        - una variable binaria para el término AND;
        - una variable binaria de salida;
        - una variable binaria de paridad para el XOR.

        En total se crean:

            3 × 25 × z

        variables binarias por ronda.
        """

        if round_index not in range(self.config.rounds):
            raise ValueError(
                "La ronda chi debe encontrarse entre 0 y "
                f"{self.config.rounds - 1}."
            )

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    index = (
                        round_index,
                        x,
                        y,
                        k,
                    )

                    self.chi_and[index] = (
                        self._create_binary_variable(
                            name=(
                                f"chi_and_r{round_index}"
                                f"_x{x}_y{y}_k{k}"
                            )
                        )
                    )

                    self.chi_output[index] = (
                        self._create_binary_variable(
                            name=(
                                f"chi_output_r{round_index}"
                                f"_x{x}_y{y}_k{k}"
                            )
                        )
                    )

                    self.chi_q[index] = (
                        self._create_binary_variable(
                            name=(
                                f"chi_q_r{round_index}"
                                f"_x{x}_y{y}_k{k}"
                            )
                        )
                    )

    def _add_iota_constraints(
        self,
        round_index: int,
    ) -> None:
        """
        Conecta la salida de chi con el siguiente estado de frontera.

        Para todos los lanes distintos de (0, 0):

            A[r + 1, x, y, k] = Chi[r, x, y, k]

        Para el lane (0, 0):

            A[r + 1, 0, 0, k]
            =
            Chi[r, 0, 0, k] XOR RC[r, k]

        Como RC[r, k] es una constante:

        - si RC[r, k] = 0, se agrega una igualdad directa;
        - si RC[r, k] = 1, se agrega:

            A[r + 1, 0, 0, k]
            =
            1 - Chi[r, 0, 0, k]
        """
        if round_index not in range(self.config.rounds):
            raise ValueError(
                "La ronda iota debe encontrarse entre 0 y "
                f"{self.config.rounds - 1}."
            )

        constant = round_constant(
            round_index=round_index,
            z=self.config.z,
        )

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    chi_variable = self.chi_output_variable(
                        round_index=round_index,
                        x=x,
                        y=y,
                        k=k,
                    )

                    next_state_variable = self.state_variable(
                        round_index=round_index + 1,
                        x=x,
                        y=y,
                        k=k,
                    )

                    constant_bit = 0

                    if x == 0 and y == 0:
                        constant_bit = (
                            constant >> k
                        ) & 1

                    if constant_bit == 0:
                        self.problem += (
                            next_state_variable
                            == chi_variable,
                            (
                                f"iota_equal_r{round_index}"
                                f"_x{x}_y{y}_k{k}"
                            ),
                        )
                    else:
                        self.problem += (
                            next_state_variable
                            == 1 - chi_variable,
                            (
                                f"iota_toggle_r{round_index}"
                                f"_x{x}_y{y}_k{k}"
                            ),
                        )


    def add_iota_layer(
        self,
        round_index: int,
    ) -> None:
        """
        Agrega la capa iota después de chi.

        La salida de iota se conecta directamente con el estado de
        frontera de la ronda siguiente.

        Requiere que chi de la misma ronda ya exista.
        La operación es idempotente.
        """
        if round_index not in range(self.config.rounds):
            raise ValueError(
                "La ronda iota debe encontrarse entre 0 y "
                f"{self.config.rounds - 1}."
            )

        if round_index in self._iota_rounds_added:
            return

        if round_index not in self._chi_rounds_added:
            raise RuntimeError(
                "Debe agregarse chi antes de iota para la "
                f"ronda {round_index}."
            )

        self._add_iota_constraints(round_index)

        self._iota_rounds_added.add(round_index)

    def iota_output_variable(
        self,
        round_index: int,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """
        Devuelve una variable de salida de iota.

        La salida de iota de la ronda r corresponde al estado de
        frontera r + 1.
        """
        if round_index not in self._iota_rounds_added:
            raise KeyError(
                "La variable iota solicitada no existe. "
                "Ejecuta add_iota_layer() primero."
            )

        return self.state_variable(
            round_index=round_index + 1,
            x=x,
            y=y,
            k=k,
        )

    def iota_output_values(
        self,
        round_index: int,
        tolerance: float = 0.5,
    ) -> list[list[list[int]]]:
        """
        Recupera la salida de iota como una estructura 5 × 5 × z.

        La salida corresponde al estado de frontera r + 1.
        """
        output = [
            [
                [0 for _ in range(self.config.z)]
                for _ in range(5)
            ]
            for _ in range(5)
        ]

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    variable = self.iota_output_variable(
                        round_index=round_index,
                        x=x,
                        y=y,
                        k=k,
                    )

                    value = variable.value()

                    if value is None:
                        raise RuntimeError(
                            "El modelo debe resolverse antes de "
                            "recuperar la salida iota."
                        )

                    output[x][y][k] = int(
                        value > tolerance
                    )

        return output

    def add_round(
        self,
        round_index: int,
    ) -> None:
        """
        Agrega una ronda completa de Keccak al modelo.

        El orden de construcción es:

            theta -> rho-pi -> chi -> iota

        La salida de la ronda queda conectada con el estado de
        frontera ``round_index + 1``.

        La operación es idempotente.
        """
        if round_index not in range(self.config.rounds):
            raise ValueError(
                "La ronda debe encontrarse entre 0 y "
                f"{self.config.rounds - 1}."
            )

        if round_index in self._rounds_added:
            return

        self.add_theta_layer(round_index)
        self.add_rho_pi_layers(round_index)
        self.add_chi_layer(round_index)
        self.add_iota_layer(round_index)

        self._rounds_added.add(round_index)

    def add_all_rounds(self) -> None:
        """
        Agrega todas las rondas configuradas en orden consecutivo.

        Para ``rounds = R`` se construyen las rondas:

            0, 1, ..., R - 1
        """
        for round_index in range(
            self.config.rounds
        ):
            self.add_round(round_index)


    # ========================================================
    # ACCESO A VARIABLES
    # ========================================================

    def state_variable(
        self,
        round_index: int,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """
        Devuelve una variable específica del estado.

        Raises
        ------
        KeyError
            Si los índices no pertenecen al modelo.
        """

        index = (round_index, x, y, k)

        if index not in self.state:
            raise KeyError(
                "Índice de estado inválido: "
                f"r={round_index}, x={x}, y={y}, k={k}."
            )

        return self.state[index]

    def initial_state_variables(
        self,
    ) -> list[pulp.LpVariable]:
        """Devuelve las variables del estado inicial."""

        return [
            self.state[0, x, y, k]
            for x in range(5)
            for y in range(5)
            for k in range(self.config.z)
        ]

    def final_state_variables(
        self,
    ) -> list[pulp.LpVariable]:
        """Devuelve las variables del último estado de frontera."""

        final_round = self.config.rounds

        return [
            self.state[final_round, x, y, k]
            for x in range(5)
            for y in range(5)
            for k in range(self.config.z)
        ]

    # ========================================================
    # RESTRICCIÓN INICIAL
    # ========================================================

    def _add_chi_constraints(
        self,
        round_index: int,
    ) -> None:
        """
        Agrega la formulación MILP de chi.

        Para cada posición se define:

            a = B[x, y, k]
            b = B[(x + 1) mod 5, y, k]
            c = B[(x + 2) mod 5, y, k]

            t = (NOT b) AND c
            d = a XOR t

        El término AND se linealiza mediante:

            t <= 1 - b
            t <= c
            t >= c - b

        El XOR se representa mediante:

            a + t = d + 2 q
        """

        for x in range(5):
            next_x = (x + 1) % 5
            second_next_x = (x + 2) % 5

            for y in range(5):
                for k in range(self.config.z):
                    index = (
                        round_index,
                        x,
                        y,
                        k,
                    )

                    a_variable = self.rho_pi_output_variable(
                        round_index=round_index,
                        x=x,
                        y=y,
                        k=k,
                    )

                    b_variable = self.rho_pi_output_variable(
                        round_index=round_index,
                        x=next_x,
                        y=y,
                        k=k,
                    )

                    c_variable = self.rho_pi_output_variable(
                        round_index=round_index,
                        x=second_next_x,
                        y=y,
                        k=k,
                    )

                    and_variable = self.chi_and[index]
                    output_variable = self.chi_output[index]
                    parity_variable = self.chi_q[index]

                    # t <= 1 - b
                    self.problem += (
                        and_variable <= 1 - b_variable,
                        (
                            f"chi_and_not_b_r{round_index}"
                            f"_x{x}_y{y}_k{k}"
                        ),
                    )

                    # t <= c
                    self.problem += (
                        and_variable <= c_variable,
                        (
                            f"chi_and_c_r{round_index}"
                            f"_x{x}_y{y}_k{k}"
                        ),
                    )

                    # t >= c - b
                    self.problem += (
                        and_variable >= c_variable - b_variable,
                        (
                            f"chi_and_lower_r{round_index}"
                            f"_x{x}_y{y}_k{k}"
                        ),
                    )

                    # a XOR t = d
                    self.problem += (
                        a_variable + and_variable
                        == output_variable + 2 * parity_variable,
                        (
                            f"chi_xor_r{round_index}"
                            f"_x{x}_y{y}_k{k}"
                        ),
                    )


    def add_chi_layer(
    self,
    round_index: int,
    ) -> None:
        """
        Agrega la capa chi después de rho y pi.

        Requiere que las capas rho y pi de la misma ronda ya existan.
        La operación es idempotente.
        """

        if round_index in self._chi_rounds_added:
            return

        if round_index not in self._rho_pi_rounds_added:
            raise RuntimeError(
                "Deben agregarse rho y pi antes de chi para la "
                f"ronda {round_index}."
            )

        self._create_chi_variables(round_index)
        self._add_chi_constraints(round_index)

        self._chi_rounds_added.add(round_index)
        
    def chi_output_variable(
        self,
        round_index: int,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve una variable de salida de chi."""

        index = (
            round_index,
            x,
            y,
            k,
        )

        if index not in self.chi_output:
            raise KeyError(
                "La variable chi solicitada no existe. "
                "Ejecuta add_chi_layer() primero."
            )

        return self.chi_output[index]
                    
    def _add_rho_pi_constraints(
        self,
        round_index: int,
    ) -> None:
        """
        Conecta la salida de theta con la salida conjunta de rho y pi.

        Para cada posición de origen:

            P[x_dest, y_dest, k_dest] = T[x, y, k]
        """

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    destination = rho_pi_destination(
                        x=x,
                        y=y,
                        k=k,
                        z=self.config.z,
                    )

                    x_destination, y_destination, k_destination = (
                        destination
                    )

                    theta_variable = self.theta_output_variable(
                        round_index=round_index,
                        x=x,
                        y=y,
                        k=k,
                    )

                    rho_pi_variable = self.rho_pi_output[
                        round_index,
                        x_destination,
                        y_destination,
                        k_destination,
                    ]

                    self.problem += (
                        rho_pi_variable == theta_variable,
                        (
                            f"rho_pi_mapping_r{round_index}"
                            f"_x{x}_y{y}_k{k}"
                        ),
                    )

    def chi_output_values(
        self,
        round_index: int,
        tolerance: float = 0.5,
    ) -> list[list[list[int]]]:
        """
        Recupera la salida de chi como una estructura 5 × 5 × z.
        """

        output = [
            [
                [0 for _ in range(self.config.z)]
                for _ in range(5)
            ]
            for _ in range(5)
        ]

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    variable = self.chi_output_variable(
                        round_index=round_index,
                        x=x,
                        y=y,
                        k=k,
                    )

                    value = variable.value()

                    if value is None:
                        raise RuntimeError(
                            "El modelo debe resolverse antes de "
                            "recuperar la salida chi."
                        )

                    output[x][y][k] = int(
                        value > tolerance
                    )

        return output

    def add_nonzero_input_constraint(self) -> None:
        """
        Impide la solución diferencial completamente nula.

        Se impone:

            sum A[0, x, y, k] >= 1

        Esta restricción será necesaria también en el modelo final.
        """

        if self._nonzero_input_added:
            return

        self.problem += (
            pulp.lpSum(self.initial_state_variables()) >= 1,
            "entrada_diferencial_no_nula",
        )

        self._nonzero_input_added = True

    # ========================================================
    # OBJETIVO PROVISIONAL
    # ========================================================

    def set_smoke_test_objective(self) -> None:
        """
        Define una función objetivo provisional.

        Se minimiza el número de bits activos en el estado inicial:

            min sum A[0, x, y, k]

        Con la restricción de entrada no nula, el óptimo esperado es 1.

        Esta no es todavía la función objetivo criptográfica final.
        """

        if self._objective_added:
            return

        self.problem += (
            pulp.lpSum(self.initial_state_variables()),
            "objetivo_provisional_bits_iniciales",
        )

        self._objective_added = True

    def set_boundary_hamming_weight_objective(
        self,
        boundary_index: int,
    ) -> None:
        """
        Minimiza el peso de Hamming de un estado de frontera.

        El objetivo se define como:

            sum_{x,y,k} state[boundary_index, x, y, k]

        Parameters
        ----------
        boundary_index:
            Índice del estado de frontera. Debe encontrarse entre
            0 y config.rounds, ambos inclusive.

        Raises
        ------
        TypeError
            Si el índice no es entero.

        ValueError
            Si el índice no corresponde a un estado de frontera.
        """
        if not isinstance(boundary_index, int):
            raise TypeError(
                "El índice del estado de frontera debe ser un entero."
            )

        if boundary_index not in range(
            self.config.rounds + 1
        ):
            raise ValueError(
                "El estado de frontera debe encontrarse entre 0 y "
                f"{self.config.rounds}."
            )

        objective = pulp.lpSum(
            self.state[
                boundary_index,
                x,
                y,
                k,
            ]
            for x in range(5)
            for y in range(5)
            for k in range(self.config.z)
        )

        self.problem.setObjective(
            objective
        )

        self._objective_added = True

    def set_input_output_hamming_weight_objective(
        self,
    ) -> None:
        """
        Minimiza la suma de los pesos de Hamming de la entrada y
        del estado de frontera final.

        El objetivo es:

            HW(A_0) + HW(A_R)

        donde R es el número de rondas configurado.
        """
        final_boundary = self.config.rounds

        input_weight = pulp.lpSum(
            self.state[
                0,
                x,
                y,
                k,
            ]
            for x in range(5)
            for y in range(5)
            for k in range(self.config.z)
        )

        output_weight = pulp.lpSum(
            self.state[
                final_boundary,
                x,
                y,
                k,
            ]
            for x in range(5)
            for y in range(5)
            for k in range(self.config.z)
        )

        self.problem.setObjective(
            input_weight + output_weight
        )

        self._objective_added = True
        

    def objective_value(self) -> float:
        """
        Devuelve el valor de la función objetivo después de resolver.

        Raises
        ------
        RuntimeError
            Si el modelo no tiene función objetivo o todavía no ha
            sido resuelto.
        """
        if (
            not self._objective_added
            or self.problem.objective is None
        ):
            raise RuntimeError(
                "El modelo no tiene una función objetivo."
            )

        if self.problem.status == pulp.LpStatusNotSolved:
            raise RuntimeError(
                "El modelo debe resolverse antes de recuperar "
                "el valor objetivo."
            )

        result = pulp.value(
            self.problem.objective
        )

        if result is None:
            raise RuntimeError(
                "No fue posible recuperar el valor de la "
                "función objetivo."
            )

        return float(result)

        
    # ========================================================
    # VARIABLES DE THETA
    # ========================================================

    def _create_theta_variables(
        self,
        round_index: int,
    ) -> None:
        """
        Crea las variables auxiliares de theta para una ronda.

        Variables por ronda:

        - C y QC: 5 × z cada una;
        - D y QD: 5 × z cada una;
        - T y QT: 25 × z cada una.
        """

        if round_index not in range(self.config.rounds):
            raise ValueError(
                "La ronda theta debe encontrarse entre 0 y "
                f"{self.config.rounds - 1}."
            )

        for x in range(5):
            for k in range(self.config.z):
                column_index = (
                    round_index,
                    x,
                    k,
                )

                self.theta_c[column_index] = (
                    self._create_binary_variable(
                        name=(
                            f"theta_c_r{round_index}"
                            f"_x{x}_k{k}"
                        )
                    )
                )

                self.theta_qc[column_index] = (
                    self._create_integer_variable(
                        name=(
                            f"theta_qc_r{round_index}"
                            f"_x{x}_k{k}"
                        ),
                        low_bound=0,
                        up_bound=2,
                    )
                )

                self.theta_d[column_index] = (
                    self._create_binary_variable(
                        name=(
                            f"theta_d_r{round_index}"
                            f"_x{x}_k{k}"
                        )
                    )
                )

                self.theta_qd[column_index] = (
                    self._create_binary_variable(
                        name=(
                            f"theta_qd_r{round_index}"
                            f"_x{x}_k{k}"
                        )
                    )
                )

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    bit_index = (
                        round_index,
                        x,
                        y,
                        k,
                    )

                    self.theta_output[bit_index] = (
                        self._create_binary_variable(
                            name=(
                                f"theta_t_r{round_index}"
                                f"_x{x}_y{y}_k{k}"
                            )
                        )
                    )

                    self.theta_qt[bit_index] = (
                        self._create_binary_variable(
                            name=(
                                f"theta_qt_r{round_index}"
                                f"_x{x}_y{y}_k{k}"
                            )
                        )
                    )


    def _add_theta_c_constraints(
        self,
        round_index: int,
    ) -> None:
        """
        Agrega:

            sum_y A[r,x,y,k] = C[r,x,k] + 2 QC[r,x,k]
        """

        for x in range(5):
            for k in range(self.config.z):
                c_variable = self.theta_c[
                    round_index,
                    x,
                    k,
                ]

                qc_variable = self.theta_qc[
                    round_index,
                    x,
                    k,
                ]

                input_sum = pulp.lpSum(
                    self.state[
                        round_index,
                        x,
                        y,
                        k,
                    ]
                    for y in range(5)
                )

                self.problem += (
                    input_sum
                    == c_variable + 2 * qc_variable,
                    (
                        f"theta_c_parity_r{round_index}"
                        f"_x{x}_k{k}"
                    ),
                )
    def _add_theta_d_constraints(
        self,
        round_index: int,
    ) -> None:
        """
        Agrega:

            C[x-1,k] + C[x+1,k-1]
            =
            D[x,k] + 2 QD[x,k]
        """

        z = self.config.z

        for x in range(5):
            for k in range(z):
                left_x = (x - 1) % 5
                right_x = (x + 1) % 5
                rotated_k = (k - 1) % z

                left_c = self.theta_c[
                    round_index,
                    left_x,
                    k,
                ]

                right_c = self.theta_c[
                    round_index,
                    right_x,
                    rotated_k,
                ]

                d_variable = self.theta_d[
                    round_index,
                    x,
                    k,
                ]

                qd_variable = self.theta_qd[
                    round_index,
                    x,
                    k,
                ]

                self.problem += (
                    left_c + right_c
                    == d_variable + 2 * qd_variable,
                    (
                        f"theta_d_parity_r{round_index}"
                        f"_x{x}_k{k}"
                    ),
                )
    
    def _add_theta_output_constraints(
        self,
        round_index: int,
    ) -> None:
        """
        Agrega:

            A[r,x,y,k] + D[r,x,k]
            =
            T[r,x,y,k] + 2 QT[r,x,y,k]
        """

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    input_variable = self.state[
                        round_index,
                        x,
                        y,
                        k,
                    ]

                    d_variable = self.theta_d[
                        round_index,
                        x,
                        k,
                    ]

                    output_variable = self.theta_output[
                        round_index,
                        x,
                        y,
                        k,
                    ]

                    qt_variable = self.theta_qt[
                        round_index,
                        x,
                        y,
                        k,
                    ]

                    self.problem += (
                        input_variable + d_variable
                        == output_variable + 2 * qt_variable,
                        (
                            f"theta_output_r{round_index}"
                            f"_x{x}_y{y}_k{k}"
                        ),
                    )

    def add_theta_layer(
        self,
        round_index: int,
    ) -> None:
        """
        Crea y conecta la capa theta de una ronda.

        La operación es idempotente.
        """

        if round_index in self._theta_rounds_added:
            return

        self._create_theta_variables(round_index)
        self._add_theta_c_constraints(round_index)
        self._add_theta_d_constraints(round_index)
        self._add_theta_output_constraints(round_index)

        self._theta_rounds_added.add(round_index)

    def theta_output_variable(
        self,
        round_index: int,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve una variable de salida de theta."""

        index = (
            round_index,
            x,
            y,
            k,
        )

        if index not in self.theta_output:
            raise KeyError(
                "La variable theta solicitada no existe. "
                "Ejecuta add_theta_layer() primero."
            )

        return self.theta_output[index]

    def theta_output_values(
        self,
        round_index: int,
        tolerance: float = 0.5,
    ) -> list[list[list[int]]]:
        """
        Recupera la salida theta como una estructura 5 × 5 × z.
        """

        output = [
            [
                [0 for _ in range(self.config.z)]
                for _ in range(5)
            ]
            for _ in range(5)
        ]

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    variable = self.theta_output_variable(
                        round_index,
                        x,
                        y,
                        k,
                    )

                    value = variable.value()

                    if value is None:
                        raise RuntimeError(
                            "El modelo debe resolverse antes de "
                            "recuperar la salida theta."
                        )

                    output[x][y][k] = int(
                        value > tolerance
                    )

        return output

    def fix_state_values(
        self,
        round_index: int,
        values: list[list[list[int]]],
    ) -> None:
        """
        Fija un estado completo a valores binarios conocidos.
        """

        if len(values) != 5:
            raise ValueError(
                "El estado debe contener cinco posiciones en x."
            )

        for x in range(5):
            if len(values[x]) != 5:
                raise ValueError(
                    "El estado debe contener cinco posiciones en y."
                )

            for y in range(5):
                if len(values[x][y]) != self.config.z:
                    raise ValueError(
                        "Cada lane debe contener exactamente z bits."
                    )

                for k in range(self.config.z):
                    value = int(values[x][y][k])

                    if value not in {0, 1}:
                        raise ValueError(
                            "Solo pueden fijarse valores binarios."
                        )

                    variable = self.state_variable(
                        round_index,
                        x,
                        y,
                        k,
                    )

                    self.problem += (
                        variable == value,
                        (
                            f"fix_state_r{round_index}"
                            f"_x{x}_y{y}_k{k}"
                        ),
                    )

    def set_feasibility_objective(self) -> None:
        """
        Define un objetivo constante para problemas de factibilidad.

        La solución queda determinada por las restricciones.
        """

        if self._objective_added:
            return

        self.problem += 0, "objetivo_factibilidad"
        self._objective_added = True

    def add_rho_pi_layers(
        self,
        round_index: int,
    ) -> None:
        """
        Agrega las capas rho y pi después de theta.

        Requiere que la capa theta de la misma ronda ya exista.
        La operación es idempotente.
        """

        if round_index in self._rho_pi_rounds_added:
            return

        if round_index not in self._theta_rounds_added:
            raise RuntimeError(
                "Debe agregarse theta antes de rho y pi para la "
                f"ronda {round_index}."
            )

        self._create_rho_pi_variables(round_index)
        self._add_rho_pi_constraints(round_index)

        self._rho_pi_rounds_added.add(round_index)

    def rho_pi_output_variable(
        self,
        round_index: int,
        x: int,
        y: int,
        k: int,
    ) -> pulp.LpVariable:
        """Devuelve una variable de salida de rho y pi."""

        index = (
            round_index,
            x,
            y,
            k,
        )

        if index not in self.rho_pi_output:
            raise KeyError(
                "La variable rho-pi solicitada no existe. "
                "Ejecuta add_rho_pi_layers() primero."
            )

        return self.rho_pi_output[index]

    def rho_pi_output_values(
        self,
        round_index: int,
        tolerance: float = 0.5,
    ) -> list[list[list[int]]]:
        """
        Recupera la salida de rho y pi como una estructura 5 × 5 × z.
        """

        output = [
            [
                [0 for _ in range(self.config.z)]
                for _ in range(5)
            ]
            for _ in range(5)
        ]

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    variable = self.rho_pi_output_variable(
                        round_index=round_index,
                        x=x,
                        y=y,
                        k=k,
                    )

                    value = variable.value()

                    if value is None:
                        raise RuntimeError(
                            "El modelo debe resolverse antes de "
                            "recuperar la salida rho-pi."
                        )

                    output[x][y][k] = int(
                        value > tolerance
                    )

        return output
    
            
   

    # ========================================================
    # CONSTRUCCIÓN Y RESOLUCIÓN
    # ========================================================

    def build_skeleton(self) -> None:
        """
        Construye el modelo mínimo de validación.

        Incluye:

        - entrada no nula;
        - objetivo provisional.
        """

        self.add_nonzero_input_constraint()
        self.set_smoke_test_objective()

    def solve(self) -> str:
        """
        Resuelve el modelo con el solver definido en la configuración.

        Returns
        -------
        str
            Estado textual de PuLP.
        """

        if not self._objective_added:
            raise RuntimeError(
                "El modelo no tiene función objetivo. "
                "Ejecuta build_skeleton() o define un objetivo."
            )

        solver = build_solver(self.config)
        self.problem.solve(solver)

        return pulp.LpStatus[self.problem.status]

                
    # ========================================================
    # RESULTADOS
    # ========================================================

    def active_initial_positions(
        self,
        tolerance: float = 0.5,
    ) -> list[tuple[int, int, int]]:
        """
        Devuelve las posiciones activas del estado inicial.

        Parameters
        ----------
        tolerance:
            Umbral usado para interpretar los valores binarios.
        """

        active_positions: list[tuple[int, int, int]] = []

        for x in range(5):
            for y in range(5):
                for k in range(self.config.z):
                    variable = self.state[0, x, y, k]

                    if variable.value() is not None:
                        if variable.value() > tolerance:
                            active_positions.append((x, y, k))

        return active_positions
    
    
    def add_linear_layers(
        self,
        round_index: int,
    ) -> None:
        """
        Agrega las capas lineales theta, rho y pi de una ronda.
        """

        self.add_theta_layer(round_index)
        self.add_rho_pi_layers(round_index)


    # ========================================================
    # CONTEO E INSPECCIÓN DEL MODELO
    # ========================================================

    def declared_variable_count(self) -> int:
        """
        Devuelve todas las variables declaradas por el modelo.
        """

        return (
            len(self.state)
            + len(self.theta_c)
            + len(self.theta_qc)
            + len(self.theta_d)
            + len(self.theta_qd)
            + len(self.theta_output)
            + len(self.theta_qt)
            + len(self.rho_pi_output)
            + len(self.chi_and)
            + len(self.chi_output)
            + len(self.chi_q)
        )
    

    def attached_variable_count(self) -> int:
        """
        Devuelve el número de variables actualmente reconocidas por PuLP.

        PuLP incorpora una variable a problem.variables() cuando esta
        participa en la función objetivo o en alguna restricción.
        """

        return len(self.problem.variables())

    def constraint_count(self) -> int:
        """Devuelve el número de restricciones del problema."""

        constraints_method = getattr(
            self.problem,
            "constraints",
            None,
        )

        # PuLP 4: constraints() devuelve una lista.
        if callable(constraints_method):
            return len(constraints_method())

        # Compatibilidad defensiva con versiones anteriores.
        return len(constraints_method)

    def has_constraint(self, name: str) -> bool:
        """
        Indica si existe una restricción con el nombre recibido.
        """

        get_by_name = getattr(
            self.problem,
            "get_constraint_by_name",
            None,
        )

        if callable(get_by_name):
            return get_by_name(name) is not None

        constraints = self.problem.constraints

        if callable(constraints):
            return any(
                constraint.name == name
                for constraint in constraints()
            )

        return name in constraints
    
    
    def statistics(self) -> ModelStatistics:
        """
        Calcula las estadísticas estructurales del modelo.

        Se diferencia entre variables declaradas y variables que PuLP
        ya tiene conectadas al objetivo o a las restricciones.
        """

        return ModelStatistics(
            z=self.config.z,
            rounds=self.config.rounds,
            state_bits=self.config.state_bits,
            boundary_states=self.config.rounds + 1,
            declared_variables=self.declared_variable_count(),
            attached_variables=self.attached_variable_count(),
            total_constraints=self.constraint_count(),
        )