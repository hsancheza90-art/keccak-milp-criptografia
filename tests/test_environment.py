"""Pruebas iniciales de configuración y disponibilidad del solver."""

import sys
from pathlib import Path

import pulp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from keccak_milp.config import ExperimentConfig
from keccak_milp.solver import build_solver


def test_keccak_dimensions_z4() -> None:
    config = ExperimentConfig(z=4, rounds=3)

    assert config.state_bits == 100
    assert config.sboxes_per_round == 20
    assert config.total_sboxes == 60


def test_keccak_dimensions_z8() -> None:
    config = ExperimentConfig(z=8, rounds=3)

    assert config.state_bits == 200
    assert config.sboxes_per_round == 40
    assert config.total_sboxes == 120


def test_cbc_solver_solves_binary_problem() -> None:
    config = ExperimentConfig(
        z=4,
        rounds=1,
        solver="cbc",
        verbose=False,
    )

    problem = pulp.LpProblem(
        name="test_cbc",
        sense=pulp.LpMinimize,
    )

    x = problem.add_variable(
        name="x",
        lowBound=0,
        upBound=1,
        cat=pulp.LpBinary,
    )

    problem += x, "objetivo_prueba"
    problem += x >= 1, "restriccion_prueba"

    problem.solve(build_solver(config))

    assert pulp.LpStatus[problem.status] == "Optimal"
    assert x.value() == 1.0