from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pulp
from pulp.apis.coin_api import COIN_CMD, PULP_CBC_CMD


_SAFE_STEM_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class WarmStartPrepared:
    workdir: Path
    mps_path: Path
    mst_path: Path
    sol_path: Path
    log_path: Path
    variables: tuple[pulp.LpVariable, ...]
    variable_names: dict[str, str]
    constraint_names: dict[str, str]
    objective_name: str | None


@dataclass(frozen=True)
class CbcProcessOutcome:
    exit_code: int
    timed_out: bool
    terminated: bool
    killed: bool
    log_path: Path


@dataclass(frozen=True)
class CbcSolutionOutcome:
    status: int
    solution_status: int
    status_name: str
    solution_status_name: str


def _require_temp_workdir(
    workdir: str | os.PathLike[str],
) -> Path:
    resolved = Path(workdir).resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()

    try:
        common = Path(
            os.path.commonpath(
                [
                    str(resolved),
                    str(temp_root),
                ]
            )
        ).resolve()
    except ValueError as error:
        raise ValueError(
            "El directorio de trabajo debe estar dentro de TEMP."
        ) from error

    if common != temp_root:
        raise ValueError(
            "El directorio de trabajo debe estar dentro de TEMP."
        )

    resolved.mkdir(
        parents=True,
        exist_ok=True,
    )

    return resolved


def _validate_stem(stem: str) -> str:
    if not isinstance(stem, str):
        raise TypeError("stem debe ser una cadena.")

    if not _SAFE_STEM_PATTERN.fullmatch(stem):
        raise ValueError(
            "stem solo puede contener letras, números, punto, "
            "guion y guion bajo."
        )

    if stem in {".", ".."}:
        raise ValueError("stem no puede ser '.' ni '..'.")

    return stem


def resolve_bundled_cbc_path() -> Path:
    solver = PULP_CBC_CMD(msg=False)
    path = Path(str(solver.path)).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el CBC incluido en PuLP: {path}"
        )

    return path


def _coerce_assignment_value(
    variable: pulp.LpVariable,
    raw_value: int | float,
    *,
    tolerance: float,
) -> float:
    value = float(raw_value)

    if not math.isfinite(value):
        raise ValueError(
            f"Valor no finito para {variable.name}: {raw_value!r}"
        )

    if (
        variable.lowBound is not None
        and value < float(variable.lowBound) - tolerance
    ):
        raise ValueError(
            f"Valor por debajo del límite inferior: "
            f"{variable.name}={value}"
        )

    if (
        variable.upBound is not None
        and value > float(variable.upBound) + tolerance
    ):
        raise ValueError(
            f"Valor por encima del límite superior: "
            f"{variable.name}={value}"
        )

    if variable.cat in {
        pulp.LpInteger,
        pulp.LpBinary,
    }:
        rounded = round(value)

        if abs(value - rounded) > tolerance:
            raise ValueError(
                f"Valor no entero para {variable.name}: {value}"
            )

        value = float(rounded)

    return value


def apply_initial_assignment(
    problem: pulp.LpProblem,
    assignment: Mapping[str, int | float],
    *,
    tolerance: float = 1e-9,
) -> int:
    variables = tuple(
        sorted(
            problem.variables(),
            key=lambda variable: variable.name,
        )
    )

    variable_by_name = {
        variable.name: variable
        for variable in variables
    }

    expected_names = set(variable_by_name)
    supplied_names = set(assignment)

    missing = sorted(
        expected_names - supplied_names
    )

    extra = sorted(
        supplied_names - expected_names
    )

    if missing:
        raise ValueError(
            "Asignación incompleta; faltan variables: "
            + ", ".join(missing)
        )

    if extra:
        raise ValueError(
            "La asignación contiene variables desconocidas: "
            + ", ".join(extra)
        )

    prepared_values: dict[str, float] = {}

    for name in sorted(expected_names):
        prepared_values[name] = _coerce_assignment_value(
            variable_by_name[name],
            assignment[name],
            tolerance=tolerance,
        )

    for name in sorted(expected_names):
        variable_by_name[name].setInitialValue(
            prepared_values[name],
            check=True,
        )

    return len(prepared_values)


def validate_complete_initial_assignment(
    problem: pulp.LpProblem,
    *,
    tolerance: float = 1e-9,
) -> int:
    variables = tuple(
        sorted(
            problem.variables(),
            key=lambda variable: variable.name,
        )
    )

    unset: list[str] = []

    for variable in variables:
        if variable.varValue is None:
            unset.append(variable.name)
            continue

        _coerce_assignment_value(
            variable,
            variable.varValue,
            tolerance=tolerance,
        )

    if unset:
        raise ValueError(
            "Existen variables sin valor inicial: "
            + ", ".join(unset)
        )

    return len(variables)


def prepare_warm_start_files(
    problem: pulp.LpProblem,
    *,
    workdir: str | os.PathLike[str],
    stem: str = "warm_start",
) -> WarmStartPrepared:
    resolved_workdir = _require_temp_workdir(
        workdir
    )

    safe_stem = _validate_stem(stem)

    validate_complete_initial_assignment(
        problem
    )

    mps_path = resolved_workdir / f"{safe_stem}.mps"
    mst_path = resolved_workdir / f"{safe_stem}.mst"
    sol_path = resolved_workdir / f"{safe_stem}.sol"
    log_path = resolved_workdir / f"{safe_stem}.log"

    for path in [
        mps_path,
        mst_path,
        sol_path,
        log_path,
    ]:
        if path.exists():
            path.unlink()

    (
        variables,
        variable_names,
        constraint_names,
        objective_name,
    ) = problem.writeMPS(
        str(mps_path),
        rename=1,
    )

    writer = COIN_CMD(
        path=str(resolve_bundled_cbc_path()),
        msg=False,
    )

    writer.writesol(
        str(mst_path),
        problem,
        variables,
        variable_names,
        constraint_names,
    )

    return WarmStartPrepared(
        workdir=resolved_workdir,
        mps_path=mps_path,
        mst_path=mst_path,
        sol_path=sol_path,
        log_path=log_path,
        variables=tuple(variables),
        variable_names=dict(variable_names),
        constraint_names=dict(constraint_names),
        objective_name=objective_name,
    )


def _format_number(value: int | float) -> str:
    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValueError(
            f"Parámetro numérico no finito: {value!r}"
        )

    return format(numeric, ".17g")


def build_cbc_command(
    prepared: WarmStartPrepared,
    *,
    cbc_path: str | os.PathLike[str] | None = None,
    time_limit_seconds: int | float,
    mip_gap: int | float = 0.0,
    threads: int = 1,
    max_nodes: int = 0,
    time_mode: str = "elapsed",
) -> tuple[str, ...]:
    if time_limit_seconds <= 0:
        raise ValueError(
            "time_limit_seconds debe ser positivo."
        )

    if threads != 1:
        raise ValueError(
            "La ejecución reproducible requiere threads=1."
        )

    if max_nodes < 0:
        raise ValueError(
            "max_nodes no puede ser negativo."
        )

    if time_mode != "elapsed":
        raise ValueError(
            "time_mode debe ser 'elapsed'."
        )

    executable = (
        resolve_bundled_cbc_path()
        if cbc_path is None
        else Path(cbc_path).resolve()
    )

    if not executable.is_file():
        raise FileNotFoundError(
            f"No existe CBC: {executable}"
        )

    expected_parent = prepared.workdir.resolve()

    for path in [
        prepared.mps_path,
        prepared.mst_path,
        prepared.sol_path,
        prepared.log_path,
    ]:
        if path.resolve().parent != expected_parent:
            raise ValueError(
                "Todos los artefactos deben pertenecer "
                "al mismo directorio temporal."
            )

    return (
        str(executable),
        prepared.mps_path.name,
        "-sec",
        _format_number(time_limit_seconds),
        "-ratio",
        _format_number(mip_gap),
        "-threads",
        str(threads),
        "-timeMode",
        time_mode,
        "-maxNodes",
        str(max_nodes),
        "-mips",
        prepared.mst_path.name,
        "-branch",
        "-printingOptions",
        "all",
        "-solution",
        prepared.sol_path.name,
    )


def run_cbc_warm_start(
    command: Sequence[str],
    *,
    workdir: str | os.PathLike[str],
    log_path: str | os.PathLike[str],
    watchdog_seconds: int | float,
    termination_grace_seconds: int | float = 5,
) -> CbcProcessOutcome:
    resolved_workdir = _require_temp_workdir(
        workdir
    )

    resolved_log_path = Path(log_path).resolve()

    if resolved_log_path.parent != resolved_workdir:
        raise ValueError(
            "El log debe almacenarse en el directorio temporal."
        )

    if watchdog_seconds <= 0:
        raise ValueError(
            "watchdog_seconds debe ser positivo."
        )

    if termination_grace_seconds <= 0:
        raise ValueError(
            "termination_grace_seconds debe ser positivo."
        )

    if not command:
        raise ValueError("El comando CBC está vacío.")

    executable = Path(command[0]).resolve()

    if not executable.is_file():
        raise FileNotFoundError(
            f"No existe el ejecutable CBC: {executable}"
        )

    for argument in command[1:]:
        if os.path.isabs(argument):
            raise ValueError(
                "Los argumentos de archivos deben ser locales "
                "al directorio de trabajo."
            )

    timed_out = False
    terminated = False
    killed = False

    with resolved_log_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as log_handle:
        process = subprocess.Popen(
            tuple(command),
            cwd=str(resolved_workdir),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            shell=False,
        )

        try:
            exit_code = process.wait(
                timeout=float(watchdog_seconds)
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            terminated = True
            process.terminate()

            try:
                exit_code = process.wait(
                    timeout=float(
                        termination_grace_seconds
                    )
                )
            except subprocess.TimeoutExpired:
                killed = True
                process.kill()
                exit_code = process.wait()

    return CbcProcessOutcome(
        exit_code=int(exit_code),
        timed_out=timed_out,
        terminated=terminated,
        killed=killed,
        log_path=resolved_log_path,
    )


def load_cbc_solution(
    problem: pulp.LpProblem,
    prepared: WarmStartPrepared,
    *,
    cbc_path: str | os.PathLike[str] | None = None,
) -> CbcSolutionOutcome:
    if not prepared.sol_path.is_file():
        raise FileNotFoundError(
            f"No existe la solución CBC: {prepared.sol_path}"
        )

    executable = (
        resolve_bundled_cbc_path()
        if cbc_path is None
        else Path(cbc_path).resolve()
    )

    reader = COIN_CMD(
        path=str(executable),
        msg=False,
    )

    (
        status,
        values,
        reduced_costs,
        shadow_prices,
        slacks,
        solution_status,
    ) = reader.readsol_MPS(
        str(prepared.sol_path),
        problem,
        list(prepared.variables),
        prepared.variable_names,
        prepared.constraint_names,
        prepared.objective_name,
    )

    problem.assignVarsVals(values)
    problem.assignVarsDj(reduced_costs)
    problem.assignConsPi(shadow_prices)
    problem.assignConsSlack(
        slacks,
        activity=True,
    )
    problem.assignStatus(
        status,
        solution_status,
    )

    return CbcSolutionOutcome(
        status=int(status),
        solution_status=int(solution_status),
        status_name=pulp.LpStatus.get(
            status,
            str(status),
        ),
        solution_status_name=pulp.LpSolution.get(
            solution_status,
            str(solution_status),
        ),
    )


def solve_with_cbc_warm_start(
    problem: pulp.LpProblem,
    assignment: Mapping[str, int | float],
    *,
    workdir: str | os.PathLike[str],
    stem: str = "warm_start",
    time_limit_seconds: int | float,
    watchdog_seconds: int | float,
    mip_gap: int | float = 0.0,
    threads: int = 1,
    max_nodes: int = 0,
) -> tuple[
    WarmStartPrepared,
    CbcProcessOutcome,
    CbcSolutionOutcome,
]:
    apply_initial_assignment(
        problem,
        assignment,
    )

    prepared = prepare_warm_start_files(
        problem,
        workdir=workdir,
        stem=stem,
    )

    command = build_cbc_command(
        prepared,
        time_limit_seconds=time_limit_seconds,
        mip_gap=mip_gap,
        threads=threads,
        max_nodes=max_nodes,
    )

    process_outcome = run_cbc_warm_start(
        command,
        workdir=prepared.workdir,
        log_path=prepared.log_path,
        watchdog_seconds=watchdog_seconds,
    )

    if not prepared.sol_path.is_file():
        raise RuntimeError(
            "CBC no produjo un archivo de solución."
        )

    solution_outcome = load_cbc_solution(
        problem,
        prepared,
    )

    return (
        prepared,
        process_outcome,
        solution_outcome,
    )