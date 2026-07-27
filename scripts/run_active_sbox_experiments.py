from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pulp

from keccak_milp.active_sbox_search import (
    RestrictedThreeRoundSearchResult,
    search_three_round_two_plus_two,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "results"
)


THREE_ROUND_REFERENCE_RESULTS: dict[
    int,
    dict[str, Any],
] = {
    4: {
        "z": 4,
        "trail_count": 200,
        "realizations_per_trail": 1024,
        "evaluated_realizations": 204_800,
        "minimum_round_2_activity": 9,
        "minimum_total_activity": 13,
        "best_candidate_count": 512,
        "support_round_0": [
            [2, 0],
            [4, 0],
        ],
        "beta_round_0": [
            1,
            1,
        ],
        "support_round_1": [
            [1, 3],
            [2, 2],
        ],
        "left_values_round_1": [
            2,
            8,
        ],
        "support_round_2": [
            [0, 0],
            [0, 2],
            [1, 1],
            [2, 0],
            [2, 1],
            [3, 2],
            [4, 1],
            [4, 2],
            [4, 3],
        ],
        "delta_b2_hamming_weight": 12,
    },
    8: {
        "z": 8,
        "trail_count": 400,
        "realizations_per_trail": 1024,
        "evaluated_realizations": 409_600,
        "minimum_round_2_activity": 10,
        "minimum_total_activity": 14,
        "best_candidate_count": 512,
        "support_round_0": [
            [2, 0],
            [4, 0],
        ],
        "beta_round_0": [
            2,
            2,
        ],
        "support_round_1": [
            [3, 2],
            [4, 2],
        ],
        "left_values_round_1": [
            2,
            0,
        ],
        "support_round_2": [
            [0, 0],
            [0, 2],
            [1, 5],
            [1, 7],
            [2, 4],
            [3, 1],
            [3, 3],
            [3, 6],
            [4, 2],
            [4, 3],
        ],
        "delta_b2_hamming_weight": 13,
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta la matriz reproducible de experimentos "
            "de S-boxes activas para Keccak reducido."
        )
    )

    parser.add_argument(
        "--mode",
        choices=(
            "quick",
            "full",
        ),
        default="quick",
        help=(
            "quick usa resultados certificados previamente; "
            "full recalcula las busquedas exhaustivas 2+2+c."
        ),
    )

    parser.add_argument(
        "--z",
        nargs="+",
        type=int,
        choices=(
            4,
            8,
        ),
        default=[
            4,
            8,
        ],
        help=(
            "Tamanos de palabra que se ejecutaran."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directorio donde se guardaran CSV, JSON y Markdown."
        ),
    )

    return parser.parse_args()


def package_version(
    package_name: str,
) -> str | None:
    try:
        return version(
            package_name
        )
    except PackageNotFoundError:
        return None


def run_git_command(
    arguments: list[str],
) -> str | None:
    try:
        completed = subprocess.run(
            [
                "git",
                *arguments,
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    return completed.stdout.strip()


def git_information() -> dict[str, Any]:
    status = run_git_command(
        [
            "status",
            "--porcelain",
        ]
    )

    return {
        "branch": run_git_command(
            [
                "branch",
                "--show-current",
            ]
        ),
        "commit": run_git_command(
            [
                "rev-parse",
                "HEAD",
            ]
        ),
        "commit_short": run_git_command(
            [
                "rev-parse",
                "--short",
                "HEAD",
            ]
        ),
        "working_tree_clean": (
            status == ""
            if status is not None
            else None
        ),
        "changed_files": (
            status.splitlines()
            if status
            else []
        ),
    }


def environment_information(
    mode: str,
    selected_z: list[int],
    total_elapsed_seconds: float,
) -> dict[str, Any]:
    available_solvers = pulp.listSolvers(
        onlyAvailable=True
    )

    return {
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "mode": mode,
        "selected_z": selected_z,
        "total_elapsed_seconds": round(
            total_elapsed_seconds,
            6,
        ),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            "numpy": np.__version__,
            "pulp": pulp.__version__,
            "pytest": package_version(
                "pytest"
            ),
        },
        "solvers": {
            "selected": "PULP_CBC_CMD",
            "available": available_solvers,
            "cbc_available": (
                "PULP_CBC_CMD"
                in available_solvers
            ),
        },
        "git": git_information(),
    }


def three_round_result(
    z: int,
    mode: str,
) -> tuple[
    dict[str, Any],
    float | None,
]:
    if mode == "quick":
        return (
            dict(
                THREE_ROUND_REFERENCE_RESULTS[
                    z
                ]
            ),
            None,
        )

    start_time = perf_counter()

    result: RestrictedThreeRoundSearchResult = (
        search_three_round_two_plus_two(
            z=z
        )
    )

    elapsed_seconds = (
        perf_counter()
        - start_time
    )

    return (
        result.to_dict(),
        elapsed_seconds,
    )


def build_experiment_rows(
    selected_z: list[int],
    mode: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    rows: list[
        dict[str, Any]
    ] = []

    witnesses: dict[
        str,
        Any,
    ] = {}

    for z in selected_z:
        sboxes_per_round = 5 * z

        rows.append(
            {
                "z": z,
                "rounds": 1,
                "attempts_range": "<10",
                "sboxes_per_round": (
                    sboxes_per_round
                ),
                "result_status": (
                    "exact"
                ),
                "minimum_exact": 1,
                "lower_bound": 1,
                "upper_bound": 1,
                "witness_activity": "1",
                "search_family": (
                    "global"
                ),
                "method": (
                    "mathematical lower bound "
                    "+ constructed MILP witness"
                ),
                "evaluated_realizations": None,
                "elapsed_seconds": None,
                "solver": "CBC",
            }
        )

        rows.append(
            {
                "z": z,
                "rounds": 2,
                "attempts_range": "<20",
                "sboxes_per_round": (
                    sboxes_per_round
                ),
                "result_status": (
                    "exact"
                ),
                "minimum_exact": 4,
                "lower_bound": 4,
                "upper_bound": 4,
                "witness_activity": "2+2",
                "search_family": (
                    "global"
                ),
                "method": (
                    "exhaustive differential search "
                    "+ constructed MILP witness"
                ),
                "evaluated_realizations": (
                    182_590
                    if z == 4
                    else 749_580
                ),
                "elapsed_seconds": None,
                "solver": "CBC",
            }
        )

        (
            restricted_result,
            elapsed_seconds,
        ) = three_round_result(
            z=z,
            mode=mode,
        )

        upper_bound = int(
            restricted_result[
                "minimum_total_activity"
            ]
        )

        third_round_activity = int(
            restricted_result[
                "minimum_round_2_activity"
            ]
        )

        rows.append(
            {
                "z": z,
                "rounds": 3,
                "attempts_range": "<30",
                "sboxes_per_round": (
                    sboxes_per_round
                ),
                "result_status": (
                    "bounded"
                ),
                "minimum_exact": None,
                "lower_bound": 5,
                "upper_bound": (
                    upper_bound
                ),
                "witness_activity": (
                    f"2+2+{third_round_activity}"
                ),
                "search_family": (
                    "restricted 2+2+c"
                ),
                "method": (
                    "exhaustive restricted search "
                    "+ constructed MILP witness"
                ),
                "evaluated_realizations": (
                    restricted_result[
                        "evaluated_realizations"
                    ]
                ),
                "elapsed_seconds": (
                    round(
                        elapsed_seconds,
                        6,
                    )
                    if elapsed_seconds
                    is not None
                    else None
                ),
                "solver": "CBC",
            }
        )

        witnesses[
            f"z{z}_rounds3"
        ] = restricted_result

    return rows, witnesses


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "z",
        "rounds",
        "attempts_range",
        "sboxes_per_round",
        "result_status",
        "minimum_exact",
        "lower_bound",
        "upper_bound",
        "witness_activity",
        "search_family",
        "method",
        "evaluated_realizations",
        "elapsed_seconds",
        "solver",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def write_json(
    path: Path,
    content: Any,
) -> None:
    path.write_text(
        json.dumps(
            content,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def markdown_value(
    value: Any,
) -> str:
    if value is None:
        return "—"

    return str(
        value
    )


def write_markdown_summary(
    path: Path,
    rows: list[dict[str, Any]],
    mode: str,
) -> None:
    lines = [
        "# Resultados de S-boxes activas",
        "",
        f"Modo de ejecución: `{mode}`.",
        "",
        "| z | Rondas | Intentos | Estado | Mínimo | Cota inferior | Cota superior | Testigo |",
        "|---:|---:|:---:|:---|---:|---:|---:|:---|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['z']} | "
            f"{row['rounds']} | "
            f"{row['attempts_range']} | "
            f"{row['result_status']} | "
            f"{markdown_value(row['minimum_exact'])} | "
            f"{row['lower_bound']} | "
            f"{row['upper_bound']} | "
            f"`{row['witness_activity']}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretación",
            "",
            "- Una y dos rondas tienen mínimos globales exactos.",
            "- Para tres rondas se informa una cota inferior global y el mejor testigo validado.",
            "- Los resultados de tres rondas están restringidos a la familia `2+2+c`.",
            "- Los testigos fueron contrastados mediante el modelo MILP exacto y la implementación de referencia.",
            "",
        ]
    )

    path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8-sig",
    )


def print_summary(
    rows: list[dict[str, Any]],
    output_directory: Path,
    mode: str,
    total_elapsed_seconds: float,
) -> None:
    print("=" * 78)
    print("ACTIVE S-BOX EXPERIMENTS")
    print("=" * 78)
    print(f"mode        : {mode}")
    print(f"solver      : CBC")
    print(f"output      : {output_directory}")
    print()

    print(
        f"{'z':>3} "
        f"{'R':>3} "
        f"{'status':>9} "
        f"{'lower':>7} "
        f"{'upper':>7} "
        f"{'witness':>12}"
    )

    print("-" * 52)

    for row in rows:
        print(
            f"{row['z']:>3} "
            f"{row['rounds']:>3} "
            f"{row['result_status']:>9} "
            f"{row['lower_bound']:>7} "
            f"{row['upper_bound']:>7} "
            f"{row['witness_activity']:>12}"
        )

    print()
    print(
        "total time  : "
        f"{total_elapsed_seconds:.3f} s"
    )
    print("=" * 78)


def main() -> int:
    arguments = parse_arguments()

    selected_z = sorted(
        set(
            arguments.z
        )
    )

    output_directory = (
        arguments.output_dir.resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        "PULP_CBC_CMD"
        not in pulp.listSolvers(
            onlyAvailable=True
        )
    ):
        raise RuntimeError(
            "CBC no esta disponible en el entorno actual."
        )

    start_time = perf_counter()

    rows, witnesses = build_experiment_rows(
        selected_z=selected_z,
        mode=arguments.mode,
    )

    total_elapsed_seconds = (
        perf_counter()
        - start_time
    )

    csv_path = (
        output_directory
        / "active_sbox_results.csv"
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

    write_csv(
        path=csv_path,
        rows=rows,
    )

    write_json(
        path=witnesses_path,
        content={
            "schema_version": 1,
            "mode": arguments.mode,
            "experiments": witnesses,
        },
    )

    write_json(
        path=environment_path,
        content=environment_information(
            mode=arguments.mode,
            selected_z=selected_z,
            total_elapsed_seconds=(
                total_elapsed_seconds
            ),
        ),
    )

    write_markdown_summary(
        path=summary_path,
        rows=rows,
        mode=arguments.mode,
    )

    print_summary(
        rows=rows,
        output_directory=(
            output_directory
        ),
        mode=arguments.mode,
        total_elapsed_seconds=(
            total_elapsed_seconds
        ),
    )

    print()
    print("Generated files:")
    print(f"- {csv_path}")
    print(f"- {witnesses_path}")
    print(f"- {environment_path}")
    print(f"- {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
