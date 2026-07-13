"""
Validación inicial del entorno y del solver MILP.

Este script no implementa todavía Keccak.
Solo verifica que Python, PuLP y CBC funcionen correctamente.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import pulp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from keccak_milp.config import ExperimentConfig
from keccak_milp.solver import available_solvers, build_solver


def solve_smoke_test() -> None:
    """
    Resuelve un MILP pequeño:

        minimizar x + y

        sujeto a:
            x + 2y >= 1
            x, y binarias

    La solución óptima esperada es:
        x = 0
        y = 1
        objetivo = 1
    """

    config = ExperimentConfig(
        z=4,
        rounds=1,
        solver="cbc",
        time_limit_seconds=60,
        mip_gap=0.0,
        verbose=False,
    )

    problem = pulp.LpProblem(
        name="validacion_solver",
        sense=pulp.LpMinimize,
    )

    x = pulp.LpVariable("x", cat=pulp.LpBinary)
    y = pulp.LpVariable("y", cat=pulp.LpBinary)

    problem += x + y, "funcion_objetivo"
    problem += x + 2 * y >= 1, "restriccion_validacion"

    solver = build_solver(config)
    problem.solve(solver)

    status = pulp.LpStatus[problem.status]
    objective = pulp.value(problem.objective)

    print("=" * 60)
    print("VALIDACIÓN DEL ENTORNO")
    print("=" * 60)
    print(f"Python             : {sys.version.split()[0]}")
    print(f"Sistema operativo  : {platform.platform()}")
    print(f"PuLP               : {pulp.__version__}")
    print(f"Solvers disponibles: {available_solvers()}")
    print("-" * 60)
    print(f"Estado             : {status}")
    print(f"x                  : {x.value()}")
    print(f"y                  : {y.value()}")
    print(f"Objetivo            : {objective}")
    print("=" * 60)

    if status != "Optimal":
        raise RuntimeError(
            f"El solver no encontró una solución óptima. Estado: {status}"
        )

    if objective != 1.0:
        raise RuntimeError(
            f"Resultado inesperado. Objetivo obtenido: {objective}"
        )

    print("Validación completada correctamente.")


if __name__ == "__main__":
    solve_smoke_test()