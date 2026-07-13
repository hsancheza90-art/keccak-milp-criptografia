"""
Construcción controlada de los solvers MILP.

CBC es el solver predeterminado.
Gurobi se utiliza solamente cuando está instalado y dispone de licencia.
"""

from __future__ import annotations

import importlib.util

import pulp

from keccak_milp.config import ExperimentConfig


def is_gurobi_installed() -> bool:
    """Indica si el paquete gurobipy está instalado."""

    return importlib.util.find_spec("gurobipy") is not None


def build_solver(config: ExperimentConfig) -> pulp.LpSolver:
    """
    Construye el solver solicitado.

    Parameters
    ----------
    config:
        Configuración del experimento.

    Returns
    -------
    pulp.LpSolver
        Instancia de CBC o Gurobi compatible con PuLP.
    """

    if config.solver == "cbc":
        return pulp.COIN_CMD(
            msg=config.verbose,
            timeLimit=config.time_limit_seconds,
            gapRel=config.mip_gap,
        )

    if config.solver == "gurobi":
        if not is_gurobi_installed():
            raise RuntimeError(
                "Gurobi no está instalado. Ejecuta:\n"
                "pip install -r requirements-gurobi.txt"
            )

        return pulp.GUROBI_CMD(
            msg=config.verbose,
            timeLimit=config.time_limit_seconds,
            options=[
                ("MIPGap", config.mip_gap),
            ],
        )

    raise ValueError(f"Solver no soportado: {config.solver}")


def available_solvers() -> list[str]:
    """Devuelve los solvers reconocidos como disponibles por PuLP."""

    return pulp.listSolvers(onlyAvailable=True)