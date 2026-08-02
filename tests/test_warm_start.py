from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pulp
import pytest

import keccak_milp.warm_start as warm_start
from keccak_milp.warm_start import (
    WarmStartPrepared,
    apply_initial_assignment,
    build_cbc_command,
    load_cbc_solution,
    prepare_warm_start_files,
    run_cbc_warm_start,
    validate_complete_initial_assignment,
)


def _binary_problem(
    name: str = "warm_start_test",
) -> tuple[
    pulp.LpProblem,
    pulp.LpVariable,
    pulp.LpVariable,
]:
    problem = pulp.LpProblem(
        name,
        pulp.LpMinimize,
    )

    x = pulp.LpVariable(
        "x",
        lowBound=0,
        upBound=1,
        cat=pulp.LpBinary,
    )

    y = pulp.LpVariable(
        "y",
        lowBound=0,
        upBound=1,
        cat=pulp.LpBinary,
    )

    problem += x + y
    problem += x + y >= 1, "nonzero"

    return problem, x, y


def test_apply_initial_assignment_accepts_complete_binary_assignment() -> None:
    problem, x, y = _binary_problem()

    count = apply_initial_assignment(
        problem,
        {
            "x": 1,
            "y": 0,
        },
    )

    assert count == 2
    assert x.varValue == 1.0
    assert y.varValue == 0.0
    assert (
        validate_complete_initial_assignment(problem)
        == 2
    )


def test_apply_initial_assignment_rejects_missing_and_extra_names() -> None:
    problem, _, _ = _binary_problem()

    with pytest.raises(
        ValueError,
        match="faltan variables",
    ):
        apply_initial_assignment(
            problem,
            {
                "x": 1,
            },
        )

    with pytest.raises(
        ValueError,
        match="variables desconocidas",
    ):
        apply_initial_assignment(
            problem,
            {
                "x": 1,
                "y": 0,
                "unknown": 1,
            },
        )


def test_apply_initial_assignment_rejects_nonintegral_and_out_of_bounds() -> None:
    problem, _, _ = _binary_problem()

    with pytest.raises(
        ValueError,
        match="Valor no entero",
    ):
        apply_initial_assignment(
            problem,
            {
                "x": 0.5,
                "y": 0.5,
            },
        )

    with pytest.raises(
        ValueError,
        match="límite superior",
    ):
        apply_initial_assignment(
            problem,
            {
                "x": 2,
                "y": 0,
            },
        )


def test_prepare_warm_start_files_generates_mps_and_mst_in_temp(
    tmp_path: Path,
) -> None:
    problem, _, _ = _binary_problem(
        "prepare_files"
    )

    apply_initial_assignment(
        problem,
        {
            "x": 1,
            "y": 0,
        },
    )

    prepared = prepare_warm_start_files(
        problem,
        workdir=tmp_path,
        stem="prepared",
    )

    assert prepared.workdir == tmp_path.resolve()
    assert prepared.mps_path.is_file()
    assert prepared.mst_path.is_file()
    assert not prepared.sol_path.exists()
    assert not prepared.log_path.exists()

    mps_text = prepared.mps_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    for section in (
        "NAME",
        "ROWS",
        "COLUMNS",
        "RHS",
        "BOUNDS",
        "ENDATA",
    ):
        assert section in mps_text

    mst_lines = [
        line
        for line in prepared.mst_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if line.strip()
    ]

    assert len(mst_lines) == 3
    assert len(mst_lines[1:]) == 2


def test_prepare_warm_start_files_rejects_outside_temp_and_unsafe_stem(
    tmp_path: Path,
) -> None:
    problem, _, _ = _binary_problem(
        "reject_paths"
    )

    apply_initial_assignment(
        problem,
        {
            "x": 1,
            "y": 0,
        },
    )

    outside_temp = (
        Path.cwd()
        / "forbidden-warm-start-output"
    )

    with pytest.raises(
        ValueError,
        match="dentro de TEMP",
    ):
        prepare_warm_start_files(
            problem,
            workdir=outside_temp,
            stem="outside",
        )

    with pytest.raises(
        ValueError,
        match="stem",
    ):
        prepare_warm_start_files(
            problem,
            workdir=tmp_path,
            stem="../unsafe",
        )


def test_build_cbc_command_uses_local_names_and_reproducible_limits(
    tmp_path: Path,
) -> None:
    problem, _, _ = _binary_problem(
        "command_contract"
    )

    apply_initial_assignment(
        problem,
        {
            "x": 1,
            "y": 0,
        },
    )

    prepared = prepare_warm_start_files(
        problem,
        workdir=tmp_path,
        stem="command",
    )

    command = build_cbc_command(
        prepared,
        time_limit_seconds=20,
        mip_gap=0.0,
        threads=1,
        max_nodes=0,
        time_mode="elapsed",
    )

    assert Path(command[0]).is_absolute()
    assert Path(command[0]).is_file()

    assert prepared.mps_path.name in command
    assert prepared.mst_path.name in command
    assert prepared.sol_path.name in command

    assert str(prepared.mps_path) not in command
    assert str(prepared.mst_path) not in command
    assert str(prepared.sol_path) not in command

    assert command[
        command.index("-threads") + 1
    ] == "1"

    assert command[
        command.index("-maxNodes") + 1
    ] == "0"

    assert command[
        command.index("-timeMode") + 1
    ] == "elapsed"

    assert "-mips" in command
    assert "-branch" in command
    assert "-solution" in command


def test_build_cbc_command_rejects_unsafe_execution_parameters(
    tmp_path: Path,
) -> None:
    problem, _, _ = _binary_problem(
        "reject_command"
    )

    apply_initial_assignment(
        problem,
        {
            "x": 1,
            "y": 0,
        },
    )

    prepared = prepare_warm_start_files(
        problem,
        workdir=tmp_path,
        stem="reject",
    )

    with pytest.raises(
        ValueError,
        match="threads=1",
    ):
        build_cbc_command(
            prepared,
            time_limit_seconds=20,
            threads=2,
        )

    with pytest.raises(
        ValueError,
        match="positivo",
    ):
        build_cbc_command(
            prepared,
            time_limit_seconds=0,
        )

    with pytest.raises(
        ValueError,
        match="negativo",
    ):
        build_cbc_command(
            prepared,
            time_limit_seconds=20,
            max_nodes=-1,
        )


def test_run_cbc_warm_start_passes_explicit_cwd_and_shell_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "cbc.exe"
    executable.write_bytes(b"not executed")

    captured: dict[str, Any] = {}

    class FakeProcess:
        def wait(
            self,
            timeout: float | None = None,
        ) -> int:
            captured["wait_timeout"] = timeout
            return 0

        def terminate(self) -> None:
            raise AssertionError(
                "terminate no debe ejecutarse"
            )

        def kill(self) -> None:
            raise AssertionError(
                "kill no debe ejecutarse"
            )

    def fake_popen(
        args: tuple[str, ...],
        **kwargs: Any,
    ) -> FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        warm_start.subprocess,
        "Popen",
        fake_popen,
    )

    log_path = tmp_path / "runner.log"

    outcome = run_cbc_warm_start(
        (
            str(executable),
            "model.mps",
        ),
        workdir=tmp_path,
        log_path=log_path,
        watchdog_seconds=10,
    )

    assert outcome.exit_code == 0
    assert outcome.timed_out is False
    assert outcome.terminated is False
    assert outcome.killed is False
    assert outcome.log_path == log_path.resolve()

    kwargs = captured["kwargs"]

    assert kwargs["cwd"] == str(
        tmp_path.resolve()
    )
    assert kwargs["shell"] is False
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.STDOUT


def test_run_cbc_warm_start_terminates_and_kills_after_watchdog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "cbc.exe"
    executable.write_bytes(b"not executed")

    events: list[str] = []

    class FakeTimedOutProcess:
        def __init__(self) -> None:
            self.wait_count = 0

        def wait(
            self,
            timeout: float | None = None,
        ) -> int:
            self.wait_count += 1

            if self.wait_count <= 2:
                raise subprocess.TimeoutExpired(
                    cmd="cbc",
                    timeout=timeout,
                )

            events.append("wait_after_kill")
            return -9

        def terminate(self) -> None:
            events.append("terminate")

        def kill(self) -> None:
            events.append("kill")

    monkeypatch.setattr(
        warm_start.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeTimedOutProcess(),
    )

    outcome = run_cbc_warm_start(
        (
            str(executable),
            "model.mps",
        ),
        workdir=tmp_path,
        log_path=tmp_path / "watchdog.log",
        watchdog_seconds=1,
        termination_grace_seconds=1,
    )

    assert outcome.exit_code == -9
    assert outcome.timed_out is True
    assert outcome.terminated is True
    assert outcome.killed is True
    assert events == [
        "terminate",
        "kill",
        "wait_after_kill",
    ]


def test_load_cbc_solution_reuses_pulp_result_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, x, y = _binary_problem(
        "load_solution"
    )

    mps_path = tmp_path / "load.mps"
    mst_path = tmp_path / "load.mst"
    sol_path = tmp_path / "load.sol"
    log_path = tmp_path / "load.log"
    executable = tmp_path / "cbc.exe"

    mps_path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    mst_path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    sol_path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    executable.write_bytes(b"not executed")

    prepared = WarmStartPrepared(
        workdir=tmp_path.resolve(),
        mps_path=mps_path.resolve(),
        mst_path=mst_path.resolve(),
        sol_path=sol_path.resolve(),
        log_path=log_path.resolve(),
        variables=(x, y),
        variable_names={
            "x": "X0000000",
            "y": "X0000001",
        },
        constraint_names={
            "nonzero": "C0000000",
        },
        objective_name="OBJ",
    )

    class FakeReader:
        def __init__(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            pass

        def readsol_MPS(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[
            int,
            dict[str, float],
            dict[str, float],
            dict[str, float],
            dict[str, float],
            int,
        ]:
            return (
                pulp.LpStatusOptimal,
                {
                    "x": 1.0,
                    "y": 0.0,
                },
                {
                    "x": 0.0,
                    "y": 0.0,
                },
                {
                    "nonzero": 0.0,
                },
                {
                    "nonzero": 1.0,
                },
                pulp.LpSolutionOptimal,
            )

    monkeypatch.setattr(
        warm_start,
        "COIN_CMD",
        FakeReader,
    )

    outcome = load_cbc_solution(
        problem,
        prepared,
        cbc_path=executable,
    )

    assert outcome.status == pulp.LpStatusOptimal
    assert (
        outcome.solution_status
        == pulp.LpSolutionOptimal
    )
    assert outcome.status_name == "Optimal"
    assert x.varValue == 1.0
    assert y.varValue == 0.0
    assert x.dj == 0.0
    assert y.dj == 0.0
    assert problem.status == pulp.LpStatusOptimal