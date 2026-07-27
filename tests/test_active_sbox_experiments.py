from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "run_active_sbox_experiments.py"
)


def run_quick_experiment(
    output_directory: Path,
    selected_z: tuple[int, ...] = (
        4,
        8,
    ),
) -> subprocess.CompletedProcess[str]:
    """
    Ejecuta el experimento en modo quick dentro de un
    directorio temporal y devuelve el proceso completado.
    """
    command = [
        sys.executable,
        str(
            EXPERIMENT_SCRIPT
        ),
        "--mode",
        "quick",
        "--z",
        *[
            str(z)
            for z in selected_z
        ],
        "--output-dir",
        str(
            output_directory
        ),
    ]

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        return list(
            csv.DictReader(
                csv_file
            )
        )


def read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def test_quick_experiment_generates_six_expected_rows(
    tmp_path: Path,
) -> None:
    output_directory = (
        tmp_path
        / "results"
    )

    completed = run_quick_experiment(
        output_directory=output_directory,
        selected_z=(
            4,
            8,
        ),
    )

    assert completed.returncode == 0, (
        completed.stderr
    )

    assert (
        "ACTIVE S-BOX EXPERIMENTS"
        in completed.stdout
    )

    csv_path = (
        output_directory
        / "active_sbox_results.csv"
    )

    assert csv_path.exists()

    rows = read_csv_rows(
        csv_path
    )

    assert len(rows) == 6

    results = {
        (
            int(row["z"]),
            int(row["rounds"]),
        ): row
        for row in rows
    }

    assert set(results) == {
        (4, 1),
        (4, 2),
        (4, 3),
        (8, 1),
        (8, 2),
        (8, 3),
    }

    assert (
        results[
            (4, 1)
        ]["result_status"]
        == "exact"
    )

    assert (
        results[
            (4, 1)
        ]["minimum_exact"]
        == "1"
    )

    assert (
        results[
            (4, 2)
        ]["witness_activity"]
        == "2+2"
    )

    assert (
        results[
            (4, 3)
        ]["result_status"]
        == "bounded"
    )

    assert (
        results[
            (4, 3)
        ]["lower_bound"]
        == "5"
    )

    assert (
        results[
            (4, 3)
        ]["upper_bound"]
        == "13"
    )

    assert (
        results[
            (4, 3)
        ]["witness_activity"]
        == "2+2+9"
    )

    assert (
        results[
            (8, 3)
        ]["upper_bound"]
        == "14"
    )

    assert (
        results[
            (8, 3)
        ]["witness_activity"]
        == "2+2+10"
    )


def test_quick_experiment_generates_valid_json_artifacts(
    tmp_path: Path,
) -> None:
    output_directory = (
        tmp_path
        / "results"
    )

    completed = run_quick_experiment(
        output_directory=output_directory,
    )

    assert completed.returncode == 0, (
        completed.stderr
    )

    witnesses_path = (
        output_directory
        / "active_sbox_witnesses.json"
    )

    environment_path = (
        output_directory
        / "experiment_environment.json"
    )

    summary_path = (
        output_directory
        / "active_sbox_summary.md"
    )

    assert witnesses_path.exists()
    assert environment_path.exists()
    assert summary_path.exists()

    witnesses = read_json(
        witnesses_path
    )

    assert (
        witnesses["schema_version"]
        == 1
    )

    assert (
        witnesses["mode"]
        == "quick"
    )

    assert (
        witnesses[
            "experiments"
        ][
            "z4_rounds3"
        ][
            "minimum_total_activity"
        ]
        == 13
    )

    assert (
        witnesses[
            "experiments"
        ][
            "z8_rounds3"
        ][
            "minimum_total_activity"
        ]
        == 14
    )

    environment = read_json(
        environment_path
    )

    assert (
        environment["mode"]
        == "quick"
    )

    assert (
        environment["selected_z"]
        == [
            4,
            8,
        ]
    )

    assert (
        environment[
            "solvers"
        ][
            "selected"
        ]
        == "PULP_CBC_CMD"
    )

    assert (
        environment[
            "solvers"
        ][
            "cbc_available"
        ]
        is True
    )

    summary = summary_path.read_text(
        encoding="utf-8-sig"
    )

    assert (
        "# Resultados de S-boxes activas"
        in summary
    )

    assert (
        "| 4 | 3 | <30 | bounded |"
        in summary
    )

    assert (
        "`2+2+10`"
        in summary
    )


def test_quick_experiment_supports_single_word_size(
    tmp_path: Path,
) -> None:
    output_directory = (
        tmp_path
        / "results_z4"
    )

    completed = run_quick_experiment(
        output_directory=output_directory,
        selected_z=(
            4,
        ),
    )

    assert completed.returncode == 0, (
        completed.stderr
    )

    rows = read_csv_rows(
        output_directory
        / "active_sbox_results.csv"
    )

    assert len(rows) == 3

    assert {
        int(row["z"])
        for row in rows
    } == {
        4,
    }

    assert {
        int(row["rounds"])
        for row in rows
    } == {
        1,
        2,
        3,
    }

    witnesses = read_json(
        output_directory
        / "active_sbox_witnesses.json"
    )

    assert set(
        witnesses[
            "experiments"
        ]
    ) == {
        "z4_rounds3",
    }

    environment = read_json(
        output_directory
        / "experiment_environment.json"
    )

    assert (
        environment["selected_z"]
        == [
            4,
        ]
    )
