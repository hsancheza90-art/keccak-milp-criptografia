"""
Configuración central del proyecto MILP para Keccak reducido.

El estado de Keccak tiene dimensiones:
    5 x 5 x z

Para este trabajo:
    z = 4  -> 100 bits
    z = 8  -> 200 bits
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SolverName = Literal["cbc", "gurobi"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Parámetros de un experimento MILP."""

    z: int
    rounds: int
    solver: SolverName = "cbc"
    time_limit_seconds: int = 300
    mip_gap: float = 0.0
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.z not in {4, 8}:
            raise ValueError(
                f"El tamaño de palabra z debe ser 4 u 8. Se recibió z={self.z}."
            )

        if self.rounds not in {1, 2, 3}:
            raise ValueError(
                "El número de rondas debe ser 1, 2 o 3. "
                f"Se recibió rounds={self.rounds}."
            )

        if self.time_limit_seconds <= 0:
            raise ValueError("El límite de tiempo debe ser mayor que cero.")

        if not 0.0 <= self.mip_gap <= 1.0:
            raise ValueError("El MIP gap debe estar entre 0 y 1.")

    @property
    def state_bits(self) -> int:
        """Cantidad de bits del estado: 5 × 5 × z."""

        return 25 * self.z

    @property
    def sboxes_per_round(self) -> int:
        """Cantidad de S-boxes χ disponibles en una ronda: 5 × z."""

        return 5 * self.z

    @property
    def total_sboxes(self) -> int:
        """Cantidad total de S-boxes disponibles en todas las rondas."""

        return self.rounds * self.sboxes_per_round


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"