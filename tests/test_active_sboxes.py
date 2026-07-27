
"""
Pruebas del conteo de S-boxes activas de chi.

En esta primera etapa se valida únicamente que las salidas de rho-pi,
que constituyen la entrada de chi, sean accesibles desde las dos
ejecuciones del modelo emparejado.

Todavía no se crean diferencias internas ni variables de actividad.
"""

from __future__ import annotations

import pytest

from keccak_milp.config import ExperimentConfig
from keccak_milp.differential import PairedKeccakMILPModel


def build_structural_paired_model() -> PairedKeccakMILPModel:
    """
    Construye un modelo emparejado pequeño para pruebas estructurales.

    No se invoca al solver.
    """
    config = ExperimentConfig(
        z=4,
        rounds=2,
        solver="cbc",
        time_limit_seconds=30,
        verbose=False,
    )

    model = PairedKeccakMILPModel(config)
    model.build_paired_model()

    return model


def test_rho_pi_variables_do_not_exist_before_build() -> None:
    """
    Antes de construir las rondas, las variables rho-pi todavía no existen.
    """
    config = ExperimentConfig(
        z=4,
        rounds=1,
        solver="cbc",
        time_limit_seconds=30,
        verbose=False,
    )

    model = PairedKeccakMILPModel(config)

    with pytest.raises(
        KeyError,
        match="rho-pi",
    ):
        model.left.rho_pi_output_variable(
            round_index=0,
            x=0,
            y=0,
            k=0,
        )

    with pytest.raises(
        KeyError,
        match="rho-pi",
    ):
        model.right.rho_pi_output_variable(
            round_index=0,
            x=0,
            y=0,
            k=0,
        )


def test_paired_model_exposes_rho_pi_variables() -> None:
    """
    Las dos ejecuciones deben exponer las variables que alimentan a chi.

    Para z=4 y dos rondas, cada ejecución debe contener:

        rounds * 5 * 5 * z
        = 2 * 5 * 5 * 4
        = 200

    variables de salida rho-pi.
    """
    model = build_structural_paired_model()

    expected_variables_per_side = (
        model.config.rounds
        * 5
        * 5
        * model.config.z
    )

    assert (
        len(model.left.rho_pi_output)
        == expected_variables_per_side
    )

    assert (
        len(model.right.rho_pi_output)
        == expected_variables_per_side
    )

    sample_indices = [
        (0, 0, 0, 0),
        (0, 4, 4, 3),
        (1, 2, 3, 1),
    ]

    attached_variable_ids = {
        id(variable)
        for variable in model.problem.variables()
    }

    for round_index, x, y, k in sample_indices:
        index = (
            round_index,
            x,
            y,
            k,
        )

        left_variable = (
            model.left.rho_pi_output_variable(
                round_index=round_index,
                x=x,
                y=y,
                k=k,
            )
        )

        right_variable = (
            model.right.rho_pi_output_variable(
                round_index=round_index,
                x=x,
                y=y,
                k=k,
            )
        )

        # El accesor devuelve exactamente la variable almacenada.
        assert (
            left_variable
            is model.left.rho_pi_output[index]
        )

        assert (
            right_variable
            is model.right.rho_pi_output[index]
        )

        # Las ejecuciones izquierda y derecha son independientes.
        assert left_variable is not right_variable

        # Las variables recibieron los prefijos durante la fusión.
        assert left_variable.name.startswith(
            "left_rho_pi_"
        )

        assert right_variable.name.startswith(
            "right_rho_pi_"
        )

        # Ambas variables están conectadas al problema combinado.
        assert id(left_variable) in attached_variable_ids
        assert id(right_variable) in attached_variable_ids


def test_chi_input_difference_variables_are_created() -> None:
    """
    El modelo ampliado debe crear una diferencia XOR para cada bit
    de la entrada de chi en cada ronda.

    Esta prueba especifica la interfaz antes de implementarla.
    """
    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=4,
        rounds=2,
        solver="cbc",
        time_limit_seconds=30,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)
    model.build_paired_model()

    expected_bit_differences = (
        config.rounds
        * 5
        * 5
        * config.z
    )

    assert (
        len(model.delta_chi_input)
        == expected_bit_differences
    )

    assert (
        len(model.delta_chi_input_q)
        == expected_bit_differences
    )

    sample_index = (
        1,
        2,
        3,
        1,
    )

    variable = model.chi_input_difference_variable(
        round_index=sample_index[0],
        x=sample_index[1],
        y=sample_index[2],
        k=sample_index[3],
    )

    assert (
        variable
        is model.delta_chi_input[sample_index]
    )

    assert variable.isBinary()
    assert variable.lowBound == 0
    assert variable.upBound == 1

    attached_variable_ids = {
        id(attached_variable)
        for attached_variable in model.problem.variables()
    }

    assert id(variable) in attached_variable_ids


def test_chi_input_difference_matches_fixed_pair() -> None:
    """
    Para dos entradas concretas, delta_chi_input debe coincidir
    bit a bit con el XOR de las salidas rho-pi.

    Esta prueba requiere una solución factible, pero todavía no
    optimiza el número de S-boxes activas.
    """
    import numpy as np
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=4,
        rounds=1,
        solver="cbc",
        time_limit_seconds=30,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)
    model.build_paired_model()

    left_input = np.zeros(
        (5, 5, config.z),
        dtype=np.int64,
    )

    right_input = np.zeros(
        (5, 5, config.z),
        dtype=np.int64,
    )

    # Diferencia inicial concreta y dispersa.
    right_input[0, 0, 0] = 1
    right_input[2, 1, 3] = 1
    right_input[4, 4, 2] = 1

    for x in range(5):
        for y in range(5):
            for k in range(config.z):
                model.problem += (
                    model.left.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(left_input[x, y, k]),
                    f"fix_left_input_x{x}_y{y}_k{k}",
                )

                model.problem += (
                    model.right.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(right_input[x, y, k]),
                    f"fix_right_input_x{x}_y{y}_k{k}",
                )

    # El método solve() exige una función objetivo explícita.
    #
    # Como ambos estados de entrada están completamente fijados,
    # el peso de la diferencia en la frontera 0 también está fijado.
    # Por tanto, este objetivo no introduce una búsqueda abierta:
    # permite resolver el modelo como validación de factibilidad.
    model.set_boundary_difference_weight_objective(
        boundary_index=0,
    )

    solve_status = model.solve()

    status_name = pulp.LpStatus[
        model.problem.status
    ]

    assert solve_status == "Optimal"

    assert status_name == "Optimal"

    left_rho_pi = np.asarray(
        model.left.rho_pi_output_values(
            round_index=0,
        ),
        dtype=np.int64,
    )

    right_rho_pi = np.asarray(
        model.right.rho_pi_output_values(
            round_index=0,
        ),
        dtype=np.int64,
    )

    expected_difference = np.bitwise_xor(
        left_rho_pi,
        right_rho_pi,
    )

    recovered_difference = np.zeros(
        (5, 5, config.z),
        dtype=np.int64,
    )

    for x in range(5):
        for y in range(5):
            for k in range(config.z):
                variable = (
                    model.chi_input_difference_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                )

                value = variable.value()

                assert value is not None

                recovered_difference[x, y, k] = int(
                    value > 0.5
                )

    np.testing.assert_array_equal(
        recovered_difference,
        expected_difference,
    )


def test_active_chi_variables_are_created() -> None:
    """
    Debe existir una variable binaria por cada S-box de chi.

    Una S-box se identifica mediante:

        ronda r,
        fila y,
        posición k.

    Por tanto, el número esperado es:

        rounds * 5 * z.
    """
    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=4,
        rounds=2,
        solver="cbc",
        time_limit_seconds=30,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)
    model.build_paired_model()

    expected_active_sboxes = (
        config.rounds
        * 5
        * config.z
    )

    assert (
        len(model.active_chi)
        == expected_active_sboxes
    )

    sample_index = (
        1,
        3,
        2,
    )

    variable = model.active_chi_variable(
        round_index=sample_index[0],
        y=sample_index[1],
        k=sample_index[2],
    )

    assert variable is model.active_chi[sample_index]

    assert variable.isBinary()
    assert variable.lowBound == 0
    assert variable.upBound == 1




def test_active_chi_matches_reference_for_fixed_pair() -> None:
    """
    Para un par concreto, active_chi debe valer 1 exactamente cuando
    alguno de los cinco bits de diferencia de entrada a chi sea 1.

    La prueba valida la semántica OR de cada S-box.
    """
    import numpy as np
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=4,
        rounds=1,
        solver="cbc",
        time_limit_seconds=30,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)
    model.build_paired_model()

    left_input = np.zeros(
        (5, 5, config.z),
        dtype=np.int64,
    )

    right_input = np.zeros(
        (5, 5, config.z),
        dtype=np.int64,
    )

    right_input[0, 0, 0] = 1
    right_input[2, 1, 3] = 1
    right_input[4, 4, 2] = 1

    for x in range(5):
        for y in range(5):
            for k in range(config.z):
                model.problem += (
                    model.left.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(left_input[x, y, k]),
                    f"active_fix_left_x{x}_y{y}_k{k}",
                )

                model.problem += (
                    model.right.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(right_input[x, y, k]),
                    f"active_fix_right_x{x}_y{y}_k{k}",
                )

    # El objetivo de frontera queda completamente determinado por las
    # entradas fijadas. No se minimizan todavía S-boxes activas.
    model.set_boundary_difference_weight_objective(
        boundary_index=0,
    )

    solve_status = model.solve()

    assert solve_status == "Optimal"
    assert pulp.LpStatus[model.problem.status] == "Optimal"

    left_rho_pi = np.asarray(
        model.left.rho_pi_output_values(
            round_index=0,
        ),
        dtype=np.int64,
    )

    right_rho_pi = np.asarray(
        model.right.rho_pi_output_values(
            round_index=0,
        ),
        dtype=np.int64,
    )

    delta_chi_input_reference = np.bitwise_xor(
        left_rho_pi,
        right_rho_pi,
    )

    # El eje x contiene los cinco bits procesados por una S-box.
    expected_active = np.any(
        delta_chi_input_reference != 0,
        axis=0,
    ).astype(np.int64)

    recovered_active = np.zeros(
        (5, config.z),
        dtype=np.int64,
    )

    attached_variable_ids = {
        id(variable)
        for variable in model.problem.variables()
    }

    for y in range(5):
        for k in range(config.z):
            variable = model.active_chi_variable(
                round_index=0,
                y=y,
                k=k,
            )

            # Esta comprobación pasará cuando active_chi esté conectado
            # mediante las restricciones OR.
            assert id(variable) in attached_variable_ids

            value = variable.value()

            assert value is not None

            recovered_active[y, k] = int(
                value > 0.5
            )

    np.testing.assert_array_equal(
        recovered_active,
        expected_active,
    )

    assert int(recovered_active.sum()) == int(
        expected_active.sum()
    )


def test_zero_difference_produces_zero_active_sboxes() -> None:
    """
    Dos ejecuciones con la misma entrada no deben generar diferencias
    ni S-boxes activas en ninguna ronda.

    Se usan dos rondas para verificar que la igualdad entre las dos
    ejecuciones se conserva durante toda la propagación.
    """
    import numpy as np
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=4,
        rounds=2,
        solver="cbc",
        time_limit_seconds=30,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)
    model.build_paired_model()

    common_input = np.zeros(
        (5, 5, config.z),
        dtype=np.int64,
    )

    # Estado no trivial, pero idéntico en ambas ejecuciones.
    common_input[0, 0, 0] = 1
    common_input[1, 3, 2] = 1
    common_input[2, 4, 1] = 1
    common_input[4, 2, 3] = 1

    for x in range(5):
        for y in range(5):
            for k in range(config.z):
                bit_value = int(
                    common_input[x, y, k]
                )

                model.problem += (
                    model.left.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == bit_value,
                    f"zero_diff_fix_left_x{x}_y{y}_k{k}",
                )

                model.problem += (
                    model.right.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == bit_value,
                    f"zero_diff_fix_right_x{x}_y{y}_k{k}",
                )

    # La diferencia de entrada está completamente fijada en cero.
    model.set_boundary_difference_weight_objective(
        boundary_index=0,
    )

    solve_status = model.solve()

    assert solve_status == "Optimal"
    assert pulp.LpStatus[model.problem.status] == "Optimal"
    assert model.objective_value() == pytest.approx(0.0)

    for round_index in range(config.rounds):
        for x in range(5):
            for y in range(5):
                for k in range(config.z):
                    difference_variable = (
                        model.chi_input_difference_variable(
                            round_index=round_index,
                            x=x,
                            y=y,
                            k=k,
                        )
                    )

                    difference_value = (
                        difference_variable.value()
                    )

                    assert difference_value is not None
                    assert difference_value == pytest.approx(
                        0.0
                    )

        for y in range(5):
            for k in range(config.z):
                active_variable = (
                    model.active_chi_variable(
                        round_index=round_index,
                        y=y,
                        k=k,
                    )
                )

                active_value = active_variable.value()

                assert active_value is not None
                assert active_value == pytest.approx(0.0)


def test_active_sbox_model_build_is_idempotent() -> None:
    """
    Construir dos veces la extensión de S-boxes activas no debe
    duplicar variables ni restricciones.
    """
    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=4,
        rounds=2,
        solver="cbc",
        time_limit_seconds=30,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)

    model.build_paired_model()

    sample_difference_index = (
        1,
        2,
        3,
        1,
    )

    sample_active_index = (
        1,
        3,
        2,
    )

    difference_variable_before = (
        model.delta_chi_input[
            sample_difference_index
        ]
    )

    parity_variable_before = (
        model.delta_chi_input_q[
            sample_difference_index
        ]
    )

    active_variable_before = (
        model.active_chi[
            sample_active_index
        ]
    )

    counts_before = {
        "delta_chi_input": len(
            model.delta_chi_input
        ),
        "delta_chi_input_q": len(
            model.delta_chi_input_q
        ),
        "active_chi": len(
            model.active_chi
        ),
        "attached_variables": (
            model.attached_variable_count()
        ),
        "constraints": model.constraint_count(),
    }

    model.build_paired_model()

    counts_after = {
        "delta_chi_input": len(
            model.delta_chi_input
        ),
        "delta_chi_input_q": len(
            model.delta_chi_input_q
        ),
        "active_chi": len(
            model.active_chi
        ),
        "attached_variables": (
            model.attached_variable_count()
        ),
        "constraints": model.constraint_count(),
    }

    assert counts_after == counts_before

    assert (
        model.delta_chi_input[
            sample_difference_index
        ]
        is difference_variable_before
    )

    assert (
        model.delta_chi_input_q[
            sample_difference_index
        ]
        is parity_variable_before
    )

    assert (
        model.active_chi[
            sample_active_index
        ]
        is active_variable_before
    )

    assert len(model.delta_chi_input) == (
        config.rounds
        * 5
        * 5
        * config.z
    )

    assert len(model.delta_chi_input_q) == (
        config.rounds
        * 5
        * 5
        * config.z
    )

    assert len(model.active_chi) == (
        config.rounds
        * 5
        * config.z
    )


@pytest.mark.parametrize(
    ("z", "rounds"),
    [
        (4, 1),
        (4, 2),
        (4, 3),
        (8, 1),
        (8, 2),
        (8, 3),
    ],
)
def test_active_sbox_structure_for_supported_sizes(
    z: int,
    rounds: int,
) -> None:
    """
    Valida las cantidades estructurales para todas las combinaciones
    requeridas por la práctica.

    Esta prueba no invoca al solver.
    """
    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=z,
        rounds=rounds,
        solver="cbc",
        time_limit_seconds=30,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)
    model.build_paired_model()

    expected_chi_input_bits = (
        rounds
        * 5
        * 5
        * z
    )

    expected_active_sboxes = (
        rounds
        * 5
        * z
    )

    expected_lower_constraints = (
        expected_active_sboxes
        * 5
    )

    expected_upper_constraints = (
        expected_active_sboxes
    )

    assert len(model.delta_chi_input) == (
        expected_chi_input_bits
    )

    assert len(model.delta_chi_input_q) == (
        expected_chi_input_bits
    )

    assert len(model.active_chi) == (
        expected_active_sboxes
    )

    constraint_names = list(
        model.problem.constraints.keys()
    )

    lower_constraint_names = [
        name
        for name in constraint_names
        if name.startswith(
            "active_chi_lower_"
        )
    ]

    upper_constraint_names = [
        name
        for name in constraint_names
        if name.startswith(
            "active_chi_upper_"
        )
    ]

    assert len(lower_constraint_names) == (
        expected_lower_constraints
    )

    assert len(upper_constraint_names) == (
        expected_upper_constraints
    )

    assert (
        len(lower_constraint_names)
        + len(upper_constraint_names)
        == 6 * expected_active_sboxes
    )

    # Primera y última S-box válidas.
    first_active = model.active_chi_variable(
        round_index=0,
        y=0,
        k=0,
    )

    last_active = model.active_chi_variable(
        round_index=rounds - 1,
        y=4,
        k=z - 1,
    )

    assert first_active.isBinary()
    assert last_active.isBinary()

    attached_variable_ids = {
        id(variable)
        for variable in model.problem.variables()
    }

    assert id(first_active) in attached_variable_ids
    assert id(last_active) in attached_variable_ids


def test_active_sbox_objective_contains_all_activity_variables() -> None:
    """
    La función objetivo debe minimizar exactamente la suma de todas
    las variables active_chi.

    Esta prueba es estructural y no invoca al solver.
    """
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=4,
        rounds=2,
        solver="cbc",
        time_limit_seconds=30,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)
    model.build_paired_model()

    model.set_active_sbox_objective()

    assert model.problem.sense == pulp.LpMinimize
    assert model.problem.objective is not None
    assert model._objective_added is True

    objective_coefficients = dict(
        model.problem.objective.items()
    )

    expected_variables = set(
        model.active_chi.values()
    )

    objective_variables = set(
        objective_coefficients.keys()
    )

    assert objective_variables == expected_variables

    assert len(objective_variables) == (
        config.rounds
        * 5
        * config.z
    )

    for variable in expected_variables:
        assert objective_coefficients[variable] == pytest.approx(
            1.0
        )

    assert model.problem.objective.constant == pytest.approx(
        0.0
    )


def test_active_sbox_objective_value_matches_fixed_pair() -> None:
    """
    Para un par de entradas completamente fijado, el valor del
    objetivo debe coincidir con el número de S-boxes activas
    calculado directamente desde las salidas rho-pi.

    No se realiza una búsqueda diferencial abierta.
    """
    import numpy as np
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=4,
        rounds=1,
        solver="cbc",
        time_limit_seconds=30,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)
    model.build_paired_model()

    left_input = np.zeros(
        (5, 5, config.z),
        dtype=np.int64,
    )

    right_input = np.zeros(
        (5, 5, config.z),
        dtype=np.int64,
    )

    # Estado izquierdo no trivial.
    left_input[1, 0, 2] = 1
    left_input[3, 4, 1] = 1

    # Estado derecho con una diferencia inicial concreta.
    right_input[:] = left_input
    right_input[0, 0, 0] ^= 1
    right_input[2, 1, 3] ^= 1
    right_input[4, 4, 2] ^= 1

    for x in range(5):
        for y in range(5):
            for k in range(config.z):
                model.problem += (
                    model.left.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(left_input[x, y, k]),
                    (
                        f"objective_fix_left"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

                model.problem += (
                    model.right.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(right_input[x, y, k]),
                    (
                        f"objective_fix_right"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

    model.set_active_sbox_objective()

    solve_status = model.solve()

    assert solve_status == "Optimal"
    assert pulp.LpStatus[model.problem.status] == "Optimal"

    left_rho_pi = np.asarray(
        model.left.rho_pi_output_values(
            round_index=0,
        ),
        dtype=np.int64,
    )

    right_rho_pi = np.asarray(
        model.right.rho_pi_output_values(
            round_index=0,
        ),
        dtype=np.int64,
    )

    reference_difference = np.bitwise_xor(
        left_rho_pi,
        right_rho_pi,
    )

    reference_activity = np.any(
        reference_difference != 0,
        axis=0,
    ).astype(np.int64)

    expected_active_count = int(
        reference_activity.sum()
    )

    recovered_active_count = 0

    for y in range(5):
        for k in range(config.z):
            active_variable = model.active_chi_variable(
                round_index=0,
                y=y,
                k=k,
            )

            active_value = active_variable.value()

            assert active_value is not None

            recovered_active_count += int(
                active_value > 0.5
            )

    assert recovered_active_count == expected_active_count

    assert model.objective_value() == pytest.approx(
        float(expected_active_count)
    )

    assert model.objective_value() == pytest.approx(
        float(recovered_active_count)
    )


def test_active_sbox_objective_value_matches_fixed_pair() -> None:
    """
    Para un par de entradas completamente fijado, el valor del
    objetivo debe coincidir con el número de S-boxes activas
    calculado directamente desde las salidas rho-pi.

    No se realiza una búsqueda diferencial abierta.
    """
    import numpy as np
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=4,
        rounds=1,
        solver="cbc",
        time_limit_seconds=30,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)
    model.build_paired_model()

    left_input = np.zeros(
        (5, 5, config.z),
        dtype=np.int64,
    )

    right_input = np.zeros(
        (5, 5, config.z),
        dtype=np.int64,
    )

    # Estado izquierdo no trivial.
    left_input[1, 0, 2] = 1
    left_input[3, 4, 1] = 1

    # Estado derecho con una diferencia inicial concreta.
    right_input[:] = left_input
    right_input[0, 0, 0] ^= 1
    right_input[2, 1, 3] ^= 1
    right_input[4, 4, 2] ^= 1

    for x in range(5):
        for y in range(5):
            for k in range(config.z):
                model.problem += (
                    model.left.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(left_input[x, y, k]),
                    (
                        f"objective_fix_left"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

                model.problem += (
                    model.right.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(right_input[x, y, k]),
                    (
                        f"objective_fix_right"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

    model.set_active_sbox_objective()

    solve_status = model.solve()

    assert solve_status == "Optimal"
    assert pulp.LpStatus[model.problem.status] == "Optimal"

    left_rho_pi = np.asarray(
        model.left.rho_pi_output_values(
            round_index=0,
        ),
        dtype=np.int64,
    )

    right_rho_pi = np.asarray(
        model.right.rho_pi_output_values(
            round_index=0,
        ),
        dtype=np.int64,
    )

    reference_difference = np.bitwise_xor(
        left_rho_pi,
        right_rho_pi,
    )

    reference_activity = np.any(
        reference_difference != 0,
        axis=0,
    ).astype(np.int64)

    expected_active_count = int(
        reference_activity.sum()
    )

    recovered_active_count = 0

    for y in range(5):
        for k in range(config.z):
            active_variable = model.active_chi_variable(
                round_index=0,
                y=y,
                k=k,
            )

            active_value = active_variable.value()

            assert active_value is not None

            recovered_active_count += int(
                active_value > 0.5
            )

    assert recovered_active_count == expected_active_count

    assert model.objective_value() == pytest.approx(
        float(expected_active_count)
    )

    assert model.objective_value() == pytest.approx(
        float(recovered_active_count)
    )


def _solve_linear_system_gf2(
    matrix: "np.ndarray",
    right_hand_side: "np.ndarray",
) -> tuple["np.ndarray", int]:
    """
    Resuelve un sistema lineal binario mediante eliminación de
    Gauss-Jordan sobre GF(2).

    Devuelve una solución y el rango de la matriz.
    """
    import numpy as np

    coefficient_matrix = np.asarray(
        matrix,
        dtype=np.uint8,
    ) % 2

    result_vector = np.asarray(
        right_hand_side,
        dtype=np.uint8,
    ).reshape(-1) % 2

    row_count, column_count = (
        coefficient_matrix.shape
    )

    if result_vector.shape != (row_count,):
        raise ValueError(
            "El vector independiente tiene una dimensión inválida."
        )

    augmented = np.concatenate(
        [
            coefficient_matrix.copy(),
            result_vector[:, np.newaxis],
        ],
        axis=1,
    )

    pivot_row = 0
    pivot_columns: list[int] = []

    for column in range(column_count):
        candidate_rows = np.flatnonzero(
            augmented[
                pivot_row:,
                column,
            ]
        )

        if candidate_rows.size == 0:
            continue

        selected_row = (
            pivot_row
            + int(candidate_rows[0])
        )

        if selected_row != pivot_row:
            augmented[
                [pivot_row, selected_row]
            ] = augmented[
                [selected_row, pivot_row]
            ]

        for row in range(row_count):
            if (
                row != pivot_row
                and augmented[row, column] == 1
            ):
                augmented[row, :] ^= (
                    augmented[pivot_row, :]
                )

        pivot_columns.append(column)
        pivot_row += 1

        if pivot_row == row_count:
            break

    rank = len(pivot_columns)

    for row in range(rank, row_count):
        coefficient_part_is_zero = not np.any(
            augmented[
                row,
                :column_count,
            ]
        )

        if (
            coefficient_part_is_zero
            and augmented[row, column_count] == 1
        ):
            raise ValueError(
                "El sistema binario es incompatible."
            )

    solution = np.zeros(
        column_count,
        dtype=np.uint8,
    )

    for row, column in enumerate(
        pivot_columns
    ):
        solution[column] = augmented[
            row,
            column_count,
        ]

    return solution, rank


def test_constructed_single_active_sbox_witness_z4() -> None:
    """
    Construye una preimagen exacta de una diferencia de un solo bit
    antes de chi y la valida posteriormente con el MILP.

    La preimagen se calcula sobre GF(2), por lo que CBC no tiene que
    descubrirla mediante una búsqueda abierta.
    """
    import numpy as np
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )
    from keccak_milp.layers import (
        rho_pi,
        theta,
    )

    z = 4
    state_bit_count = 5 * 5 * z

    # ============================================================
    # MATRIZ DE LA TRANSFORMACIÓN LINEAL L = rho_pi o theta
    # ============================================================

    linear_matrix = np.zeros(
        (
            state_bit_count,
            state_bit_count,
        ),
        dtype=np.uint8,
    )

    for input_bit_index in range(
        state_bit_count
    ):
        basis_state = np.zeros(
            (5, 5, z),
            dtype=np.int64,
        )

        basis_state.reshape(-1)[
            input_bit_index
        ] = 1

        transformed_basis = rho_pi(
            theta(basis_state)
        )

        linear_matrix[
            :,
            input_bit_index,
        ] = transformed_basis.reshape(
            -1
        ).astype(np.uint8)

    # ============================================================
    # OBJETIVO: UN SOLO BIT ACTIVO EN DELTA B_0
    # ============================================================

    target_delta_b = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    target_delta_b[0, 0, 0] = 1

    delta_a_flat, matrix_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                target_delta_b.reshape(-1)
            ),
        )
    )

    # La transformación debe ser invertible para este tamaño.
    assert matrix_rank == state_bit_count

    delta_a0 = delta_a_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    assert int(delta_a0.sum()) >= 1

    # Validación independiente antes de construir el MILP.
    reference_delta_b = rho_pi(
        theta(delta_a0)
    )

    np.testing.assert_array_equal(
        reference_delta_b,
        target_delta_b,
    )

    # ============================================================
    # VALIDACIÓN DEL TESTIGO EN EL MODELO MILP
    # ============================================================

    config = ExperimentConfig(
        z=z,
        rounds=1,
        solver="cbc",
        time_limit_seconds=60,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(
        config=config,
        name="constructed_single_active_sbox_z4",
    )

    model.build_paired_model()
    model.add_nonzero_input_difference_constraint()

    left_input = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    right_input = delta_a0.copy()

    for x in range(5):
        for y in range(5):
            for k in range(z):
                model.problem += (
                    model.left.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        left_input[x, y, k]
                    ),
                    (
                        f"single_witness_fix_left"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

                model.problem += (
                    model.right.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        right_input[x, y, k]
                    ),
                    (
                        f"single_witness_fix_right"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

    model.set_active_sbox_objective()

    solve_status = model.solve()

    assert solve_status == "Optimal"
    assert (
        pulp.LpStatus[model.problem.status]
        == "Optimal"
    )

    assert model.problem.valid(
        eps=1e-7
    )

    recovered_delta_a0 = np.asarray(
        model.difference_state_values(
            boundary_index=0,
        ),
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        recovered_delta_a0,
        delta_a0,
    )

    recovered_delta_b = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    for x in range(5):
        for y in range(5):
            for k in range(z):
                variable = (
                    model.chi_input_difference_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                )

                value = variable.value()

                assert value is not None

                recovered_delta_b[x, y, k] = int(
                    value > 0.5
                )

    np.testing.assert_array_equal(
        recovered_delta_b,
        target_delta_b,
    )

    active_positions = []

    for y in range(5):
        for k in range(z):
            variable = model.active_chi_variable(
                round_index=0,
                y=y,
                k=k,
            )

            value = variable.value()

            assert value is not None

            if value > 0.5:
                active_positions.append(
                    (
                        0,
                        y,
                        k,
                    )
                )

    assert active_positions == [
        (
            0,
            0,
            0,
        )
    ]

    assert model.objective_value() == pytest.approx(
        1.0
    )


def test_constructed_single_active_sbox_witness_z8() -> None:
    """
    Construye y valida un testigo con una sola S-box activa para z=8.

    La preimagen se obtiene resolviendo sobre GF(2):

        (rho_pi o theta)(Delta A_0) = Delta B_0

    donde Delta B_0 contiene exactamente un bit activo.
    """
    import numpy as np
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )
    from keccak_milp.layers import (
        rho_pi,
        theta,
    )

    z = 8
    state_bit_count = 5 * 5 * z

    # ============================================================
    # MATRIZ DE L = rho_pi o theta
    # ============================================================

    linear_matrix = np.zeros(
        (
            state_bit_count,
            state_bit_count,
        ),
        dtype=np.uint8,
    )

    for input_bit_index in range(
        state_bit_count
    ):
        basis_state = np.zeros(
            (5, 5, z),
            dtype=np.int64,
        )

        basis_state.reshape(-1)[
            input_bit_index
        ] = 1

        transformed_basis = rho_pi(
            theta(basis_state)
        )

        linear_matrix[
            :,
            input_bit_index,
        ] = transformed_basis.reshape(
            -1
        ).astype(np.uint8)

    # ============================================================
    # UN ÚNICO BIT ACTIVO ANTES DE CHI
    # ============================================================

    target_delta_b = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    target_delta_b[0, 0, 0] = 1

    delta_a_flat, matrix_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                target_delta_b.reshape(-1)
            ),
        )
    )

    assert matrix_rank == state_bit_count

    delta_a0 = delta_a_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    assert int(delta_a0.sum()) >= 1

    reference_delta_b = rho_pi(
        theta(delta_a0)
    )

    np.testing.assert_array_equal(
        reference_delta_b,
        target_delta_b,
    )

    # ============================================================
    # VALIDACIÓN MEDIANTE MILP
    # ============================================================

    config = ExperimentConfig(
        z=z,
        rounds=1,
        solver="cbc",
        time_limit_seconds=60,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(
        config=config,
        name="constructed_single_active_sbox_z8",
    )

    model.build_paired_model()
    model.add_nonzero_input_difference_constraint()

    left_input = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    right_input = delta_a0.copy()

    for x in range(5):
        for y in range(5):
            for k in range(z):
                model.problem += (
                    model.left.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        left_input[x, y, k]
                    ),
                    (
                        f"single_z8_fix_left"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

                model.problem += (
                    model.right.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        right_input[x, y, k]
                    ),
                    (
                        f"single_z8_fix_right"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

    model.set_active_sbox_objective()

    solve_status = model.solve()

    assert solve_status == "Optimal"
    assert (
        pulp.LpStatus[model.problem.status]
        == "Optimal"
    )

    assert model.problem.valid(
        eps=1e-7
    )

    recovered_delta_b = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    for x in range(5):
        for y in range(5):
            for k in range(z):
                variable = (
                    model.chi_input_difference_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                )

                value = variable.value()

                assert value is not None

                recovered_delta_b[x, y, k] = int(
                    value > 0.5
                )

    np.testing.assert_array_equal(
        recovered_delta_b,
        target_delta_b,
    )

    active_positions = []

    for y in range(5):
        for k in range(z):
            variable = model.active_chi_variable(
                round_index=0,
                y=y,
                k=k,
            )

            value = variable.value()

            assert value is not None

            if value > 0.5:
                active_positions.append(
                    (
                        0,
                        y,
                        k,
                    )
                )

    assert active_positions == [
        (
            0,
            0,
            0,
        )
    ]

    assert model.objective_value() == pytest.approx(
        1.0
    )


def test_two_round_fixed_witness_has_one_plus_eight_active_sboxes() -> None:
    """
    Valida mediante el modelo MILP exacto el mejor testigo encontrado
    entre las trayectorias que activan una sola S-box en la primera
    ronda.

    Actividad esperada:

        ronda 0: 1 S-box;
        ronda 1: 8 S-boxes;
        total  : 9 S-boxes.

    La prueba valida un par completamente fijado. No certifica todavía
    que 9 sea el mínimo global entre todas las trayectorias de dos
    rondas.
    """
    import numpy as np
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )
    from keccak_milp.layers import (
        chi,
        rho_pi,
        theta,
    )

    z = 4
    state_bit_count = 5 * 5 * z

    # ============================================================
    # MATRIZ DE LA TRANSFORMACIÓN L = rho_pi o theta
    # ============================================================

    linear_matrix = np.zeros(
        (
            state_bit_count,
            state_bit_count,
        ),
        dtype=np.uint8,
    )

    for input_bit_index in range(
        state_bit_count
    ):
        basis_state = np.zeros(
            (5, 5, z),
            dtype=np.int64,
        )

        basis_state.reshape(-1)[
            input_bit_index
        ] = 1

        transformed_basis = rho_pi(
            theta(basis_state)
        )

        linear_matrix[
            :,
            input_bit_index,
        ] = transformed_basis.reshape(
            -1
        ).astype(np.uint8)

    # ============================================================
    # VALORES CONCRETOS A LA ENTRADA DE CHI EN LA RONDA 0
    # ============================================================

    left_b0 = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    right_b0 = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    # Bits ordenados mediante x = 0, ..., 4.
    left_b0[:, 0, 0] = np.asarray(
        [0, 0, 0, 0, 1],
        dtype=np.int64,
    )

    right_b0[:, 0, 0] = np.asarray(
        [1, 0, 0, 0, 1],
        dtype=np.int64,
    )

    expected_delta_b0 = np.bitwise_xor(
        left_b0,
        right_b0,
    )

    expected_active_round_0 = [
        (0, 0, 0),
    ]

    # ============================================================
    # CALCULAR LAS PREIMÁGENES A_0 IZQUIERDA Y DERECHA
    # ============================================================

    left_a0_flat, left_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                left_b0.reshape(-1)
            ),
        )
    )

    right_a0_flat, right_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                right_b0.reshape(-1)
            ),
        )
    )

    assert left_rank == state_bit_count
    assert right_rank == state_bit_count

    left_a0 = left_a0_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    right_a0 = right_a0_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    np.testing.assert_array_equal(
        rho_pi(theta(left_a0)),
        left_b0,
    )

    np.testing.assert_array_equal(
        rho_pi(theta(right_a0)),
        right_b0,
    )

    assert np.any(
        np.bitwise_xor(
            left_a0,
            right_a0,
        )
    )

    # ============================================================
    # PROPAGACIÓN DE REFERENCIA HACIA LA SEGUNDA RONDA
    # ============================================================

    left_after_chi = chi(left_b0)
    right_after_chi = chi(right_b0)

    # Iota aplica la misma constante a ambas ejecuciones, por lo que
    # desaparece al calcular la diferencia XOR.
    expected_delta_a1 = np.bitwise_xor(
        left_after_chi,
        right_after_chi,
    )

    expected_delta_b1 = rho_pi(
        theta(expected_delta_a1)
    )

    expected_active_round_1 = [
        (1, y, k)
        for y in range(5)
        for k in range(z)
        if np.any(
            expected_delta_b1[:, y, k]
        )
    ]

    assert int(expected_delta_b0.sum()) == 1
    assert int(expected_delta_a1.sum()) == 1
    assert int(expected_delta_b1.sum()) == 11

    assert expected_active_round_1 == [
        (1, 0, 0),
        (1, 0, 3),
        (1, 1, 1),
        (1, 2, 1),
        (1, 3, 0),
        (1, 3, 2),
        (1, 4, 0),
        (1, 4, 2),
    ]

    # ============================================================
    # CONSTRUIR Y FIJAR EL MODELO MILP DE DOS RONDAS
    # ============================================================

    config = ExperimentConfig(
        z=z,
        rounds=2,
        solver="cbc",
        time_limit_seconds=60,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(
        config=config,
        name="two_round_fixed_witness_one_plus_eight",
    )

    model.build_paired_model()
    model.add_nonzero_input_difference_constraint()

    for x in range(5):
        for y in range(5):
            for k in range(z):
                model.problem += (
                    model.left.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        left_a0[x, y, k]
                    ),
                    (
                        f"two_round_witness_fix_left"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

                model.problem += (
                    model.right.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        right_a0[x, y, k]
                    ),
                    (
                        f"two_round_witness_fix_right"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

    model.set_active_sbox_objective()

    solve_status = model.solve()

    assert solve_status == "Optimal"
    assert (
        pulp.LpStatus[model.problem.status]
        == "Optimal"
    )

    assert model.problem.valid(
        eps=1e-7
    )

    # ============================================================
    # RECUPERAR Y COMPARAR DELTA B EN AMBAS RONDAS
    # ============================================================

    recovered_delta_b = []

    for round_index in range(2):
        round_difference = np.zeros(
            (5, 5, z),
            dtype=np.int64,
        )

        for x in range(5):
            for y in range(5):
                for k in range(z):
                    variable = (
                        model.chi_input_difference_variable(
                            round_index=round_index,
                            x=x,
                            y=y,
                            k=k,
                        )
                    )

                    value = variable.value()

                    assert value is not None

                    round_difference[
                        x,
                        y,
                        k,
                    ] = int(
                        value > 0.5
                    )

        recovered_delta_b.append(
            round_difference
        )

    np.testing.assert_array_equal(
        recovered_delta_b[0],
        expected_delta_b0,
    )

    np.testing.assert_array_equal(
        recovered_delta_b[1],
        expected_delta_b1,
    )

    # ============================================================
    # RECUPERAR ACTIVIDAD POR RONDA
    # ============================================================

    recovered_active_by_round: dict[
        int,
        list[tuple[int, int, int]],
    ] = {
        0: [],
        1: [],
    }

    for round_index in range(2):
        for y in range(5):
            for k in range(z):
                variable = model.active_chi_variable(
                    round_index=round_index,
                    y=y,
                    k=k,
                )

                value = variable.value()

                assert value is not None

                if value > 0.5:
                    recovered_active_by_round[
                        round_index
                    ].append(
                        (
                            round_index,
                            y,
                            k,
                        )
                    )

    assert recovered_active_by_round[0] == (
        expected_active_round_0
    )

    assert recovered_active_by_round[1] == (
        expected_active_round_1
    )

    assert len(
        recovered_active_by_round[0]
    ) == 1

    assert len(
        recovered_active_by_round[1]
    ) == 8

    total_active = (
        len(recovered_active_by_round[0])
        + len(recovered_active_by_round[1])
    )

    assert total_active == 9

    assert model.objective_value() == pytest.approx(
        9.0
    )


def test_active_sbox_upper_bound_is_added_idempotently() -> None:
    """
    La cota superior sobre el número total de S-boxes activas debe
    agregar una sola restricción y ser idempotente para el mismo
    valor.

    Una segunda cota distinta debe rechazarse para evitar modificar
    silenciosamente el experimento.
    """
    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=4,
        rounds=2,
        solver="cbc",
        time_limit_seconds=60,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)
    model.build_paired_model()

    constraints_before = model.constraint_count()

    model.add_active_sbox_upper_bound(
        max_active_sboxes=8,
    )

    constraints_after_first_call = (
        model.constraint_count()
    )

    assert constraints_after_first_call == (
        constraints_before + 1
    )

    assert model._active_sbox_upper_bound == 8

    # La misma cota no debe duplicar la restricción.
    model.add_active_sbox_upper_bound(
        max_active_sboxes=8,
    )

    assert model.constraint_count() == (
        constraints_after_first_call
    )

    # Cambiar silenciosamente la cota alteraría el experimento.
    with pytest.raises(
        RuntimeError,
        match="cota superior",
    ):
        model.add_active_sbox_upper_bound(
            max_active_sboxes=7,
        )


def test_round_active_sbox_count_is_added_idempotently() -> None:
    """
    Debe poder fijarse exactamente el número de S-boxes activas
    de una ronda concreta.

    La misma configuración debe ser idempotente. Un valor diferente
    para una ronda ya configurada debe rechazarse.
    """
    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=4,
        rounds=2,
        solver="cbc",
        time_limit_seconds=60,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)
    model.build_paired_model()

    constraints_before = model.constraint_count()

    model.add_round_active_sbox_count(
        round_index=0,
        active_sboxes=2,
    )

    constraints_after_first_call = (
        model.constraint_count()
    )

    assert constraints_after_first_call == (
        constraints_before + 1
    )

    assert model._round_active_sbox_counts == {
        0: 2,
    }

    constraint = model.problem.constraints[
        "round_active_sbox_count_r0"
    ]

    coefficients = dict(
        constraint.items()
    )

    expected_variables = {
        model.active_chi_variable(
            round_index=0,
            y=y,
            k=k,
        )
        for y in range(5)
        for k in range(config.z)
    }

    assert set(coefficients) == expected_variables

    for variable in expected_variables:
        assert coefficients[variable] == pytest.approx(
            1.0
        )

    # En PuLP, una igualdad sum(active) == 2 se almacena como:
    #
    #     sum(active) - 2 == 0
    #
    assert constraint.constant == pytest.approx(
        -2.0
    )

    # Repetir la misma configuración no debe duplicarla.
    model.add_round_active_sbox_count(
        round_index=0,
        active_sboxes=2,
    )

    assert model.constraint_count() == (
        constraints_after_first_call
    )

    # Una ronda diferente puede configurarse independientemente.
    model.add_round_active_sbox_count(
        round_index=1,
        active_sboxes=4,
    )

    assert model.constraint_count() == (
        constraints_after_first_call + 1
    )

    assert model._round_active_sbox_counts == {
        0: 2,
        1: 4,
    }

    # No se debe cambiar silenciosamente el experimento.
    with pytest.raises(
        RuntimeError,
        match="ronda 0",
    ):
        model.add_round_active_sbox_count(
            round_index=0,
            active_sboxes=3,
        )


def test_round_active_sbox_support_is_fixed_idempotently() -> None:
    """
    Debe poder fijarse exactamente el soporte de S-boxes activas
    de una ronda.

    La misma configuración debe ser idempotente. Un soporte diferente
    para una ronda ya fijada debe rechazarse.
    """
    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )

    config = ExperimentConfig(
        z=4,
        rounds=2,
        solver="cbc",
        time_limit_seconds=60,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(config)
    model.build_paired_model()

    selected_support = {
        (0, 0),
        (2, 3),
    }

    constraints_before = model.constraint_count()

    model.fix_round_active_sbox_support(
        round_index=0,
        active_positions=selected_support,
    )

    constraints_after_first_call = (
        model.constraint_count()
    )

    # Se fija cada una de las 5*z variables de actividad.
    assert constraints_after_first_call == (
        constraints_before + 5 * config.z
    )

    assert model._round_active_sbox_supports == {
        0: frozenset(selected_support),
    }

    for y in range(5):
        for k in range(config.z):
            constraint_name = (
                f"round_active_support_r0_y{y}_k{k}"
            )

            constraint = model.problem.constraints[
                constraint_name
            ]

            variable = model.active_chi_variable(
                round_index=0,
                y=y,
                k=k,
            )

            coefficients = dict(
                constraint.items()
            )

            assert coefficients == {
                variable: pytest.approx(1.0),
            }

            expected_value = int(
                (y, k) in selected_support
            )

            # PuLP almacena:
            #
            #     variable == expected_value
            #
            # como:
            #
            #     variable - expected_value == 0
            #
            assert constraint.constant == pytest.approx(
                -float(expected_value)
            )

    # Repetir el mismo soporte no debe duplicar restricciones.
    model.fix_round_active_sbox_support(
        round_index=0,
        active_positions={
            (2, 3),
            (0, 0),
        },
    )

    assert model.constraint_count() == (
        constraints_after_first_call
    )

    # Otra ronda puede fijarse independientemente.
    model.fix_round_active_sbox_support(
        round_index=1,
        active_positions={
            (1, 1),
        },
    )

    assert model.constraint_count() == (
        constraints_after_first_call
        + 5 * config.z
    )

    assert model._round_active_sbox_supports == {
        0: frozenset({
            (0, 0),
            (2, 3),
        }),
        1: frozenset({
            (1, 1),
        }),
    }

    # No se debe cambiar silenciosamente el soporte.
    with pytest.raises(
        RuntimeError,
        match="ronda 0",
    ):
        model.fix_round_active_sbox_support(
            round_index=0,
            active_positions={
                (0, 0),
                (1, 0),
            },
        )


def test_two_round_constructed_witness_has_two_plus_two_active_sboxes() -> None:
    """
    Valida mediante el MILP exacto el testigo óptimo para z=4 y
    dos rondas:

        ronda 0: 2 S-boxes activas;
        ronda 1: 2 S-boxes activas;
        total  : 4 S-boxes activas.
    """
    import numpy as np
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )
    from keccak_milp.layers import (
        chi,
        rho_pi,
        theta,
    )

    z = 4
    state_bit_count = 5 * 5 * z

    # ============================================================
    # MATRIZ DE L = rho_pi o theta
    # ============================================================

    linear_matrix = np.zeros(
        (
            state_bit_count,
            state_bit_count,
        ),
        dtype=np.uint8,
    )

    for input_bit_index in range(
        state_bit_count
    ):
        basis_state = np.zeros(
            (5, 5, z),
            dtype=np.int64,
        )

        basis_state.reshape(-1)[
            input_bit_index
        ] = 1

        transformed_basis = rho_pi(
            theta(basis_state)
        )

        linear_matrix[
            :,
            input_bit_index,
        ] = transformed_basis.reshape(
            -1
        ).astype(np.uint8)

    # ============================================================
    # ENTRADAS CONCRETAS DE CHI EN LA RONDA 0
    # ============================================================

    left_b0 = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    right_b0 = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    # Valor local decimal 11:
    #
    #     [1, 1, 0, 1, 0]
    #
    # en las posiciones (y, k) = (0, 0) y (1, 0).
    right_local_input = np.asarray(
        [1, 1, 0, 1, 0],
        dtype=np.int64,
    )

    right_b0[:, 0, 0] = right_local_input
    right_b0[:, 1, 0] = right_local_input

    expected_delta_b0 = np.bitwise_xor(
        left_b0,
        right_b0,
    )

    expected_active_round_0 = [
        (0, 0, 0),
        (0, 1, 0),
    ]

    assert [
        (0, y, k)
        for y in range(5)
        for k in range(z)
        if np.any(
            expected_delta_b0[:, y, k]
        )
    ] == expected_active_round_0

    # ============================================================
    # CALCULAR LAS PREIMÁGENES A_0
    # ============================================================

    left_a0_flat, left_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                left_b0.reshape(-1)
            ),
        )
    )

    right_a0_flat, right_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                right_b0.reshape(-1)
            ),
        )
    )

    assert left_rank == state_bit_count
    assert right_rank == state_bit_count

    left_a0 = left_a0_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    right_a0 = right_a0_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    np.testing.assert_array_equal(
        rho_pi(theta(left_a0)),
        left_b0,
    )

    np.testing.assert_array_equal(
        rho_pi(theta(right_a0)),
        right_b0,
    )

    assert np.any(
        np.bitwise_xor(
            left_a0,
            right_a0,
        )
    )

    # ============================================================
    # PROPAGACIÓN DE REFERENCIA HACIA LA RONDA 1
    # ============================================================

    left_after_chi = chi(left_b0)
    right_after_chi = chi(right_b0)

    expected_delta_a1 = np.bitwise_xor(
        left_after_chi,
        right_after_chi,
    )

    expected_delta_b1 = rho_pi(
        theta(expected_delta_a1)
    )

    expected_active_round_1 = [
        (1, y, k)
        for y in range(5)
        for k in range(z)
        if np.any(
            expected_delta_b1[:, y, k]
        )
    ]

    assert expected_active_round_1 == [
        (1, 0, 0),
        (1, 3, 0),
    ]

    assert int(
        expected_delta_b1.sum()
    ) == 2

    # ============================================================
    # CONSTRUIR EL MODELO MILP
    # ============================================================

    config = ExperimentConfig(
        z=z,
        rounds=2,
        solver="cbc",
        time_limit_seconds=60,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(
        config=config,
        name="two_round_constructed_witness_two_plus_two",
    )

    model.build_paired_model()
    model.add_nonzero_input_difference_constraint()

    # La cota coincide con el óptimo demostrado.
    model.add_active_sbox_upper_bound(
        max_active_sboxes=4,
    )

    # Fijar los dos estados iniciales completos.
    for x in range(5):
        for y in range(5):
            for k in range(z):
                model.problem += (
                    model.left.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        left_a0[x, y, k]
                    ),
                    (
                        f"two_plus_two_fix_left"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

                model.problem += (
                    model.right.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        right_a0[x, y, k]
                    ),
                    (
                        f"two_plus_two_fix_right"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

    model.set_active_sbox_objective()

    solve_status = model.solve()

    assert solve_status == "Optimal"

    assert (
        pulp.LpStatus[
            model.problem.status
        ]
        == "Optimal"
    )

    assert model.problem.valid(
        eps=1e-7
    )

    # ============================================================
    # RECUPERAR DIFERENCIAS ANTES DE CHI
    # ============================================================

    recovered_delta_b = []

    for round_index in range(2):
        round_difference = np.zeros(
            (5, 5, z),
            dtype=np.int64,
        )

        for x in range(5):
            for y in range(5):
                for k in range(z):
                    variable = (
                        model.chi_input_difference_variable(
                            round_index=round_index,
                            x=x,
                            y=y,
                            k=k,
                        )
                    )

                    value = variable.value()

                    assert value is not None

                    round_difference[
                        x,
                        y,
                        k,
                    ] = int(
                        value > 0.5
                    )

        recovered_delta_b.append(
            round_difference
        )

    np.testing.assert_array_equal(
        recovered_delta_b[0],
        expected_delta_b0,
    )

    np.testing.assert_array_equal(
        recovered_delta_b[1],
        expected_delta_b1,
    )

    # ============================================================
    # RECUPERAR ACTIVIDAD
    # ============================================================

    recovered_active_by_round: dict[
        int,
        list[tuple[int, int, int]],
    ] = {
        0: [],
        1: [],
    }

    for round_index in range(2):
        for y in range(5):
            for k in range(z):
                variable = model.active_chi_variable(
                    round_index=round_index,
                    y=y,
                    k=k,
                )

                value = variable.value()

                assert value is not None

                if value > 0.5:
                    recovered_active_by_round[
                        round_index
                    ].append(
                        (
                            round_index,
                            y,
                            k,
                        )
                    )

    assert recovered_active_by_round[0] == (
        expected_active_round_0
    )

    assert recovered_active_by_round[1] == (
        expected_active_round_1
    )

    assert len(
        recovered_active_by_round[0]
    ) == 2

    assert len(
        recovered_active_by_round[1]
    ) == 2

    total_active = sum(
        len(positions)
        for positions
        in recovered_active_by_round.values()
    )

    assert total_active == 4

    assert model.objective_value() == pytest.approx(
        4.0
    )


def test_two_round_constructed_witness_z8_has_two_plus_two_active_sboxes() -> None:
    """
    Valida mediante el MILP exacto el testigo óptimo para z=8 y
    dos rondas:

        ronda 0: 2 S-boxes activas;
        ronda 1: 2 S-boxes activas;
        total  : 4 S-boxes activas.
    """
    import numpy as np
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )
    from keccak_milp.layers import (
        chi,
        rho_pi,
        theta,
    )

    z = 8
    state_bit_count = 5 * 5 * z

    # ============================================================
    # MATRIZ DE L = rho_pi o theta
    # ============================================================

    linear_matrix = np.zeros(
        (
            state_bit_count,
            state_bit_count,
        ),
        dtype=np.uint8,
    )

    for input_bit_index in range(
        state_bit_count
    ):
        basis_state = np.zeros(
            (5, 5, z),
            dtype=np.int64,
        )

        basis_state.reshape(-1)[
            input_bit_index
        ] = 1

        transformed_basis = rho_pi(
            theta(basis_state)
        )

        linear_matrix[
            :,
            input_bit_index,
        ] = transformed_basis.reshape(
            -1
        ).astype(np.uint8)

    # ============================================================
    # ENTRADAS CONCRETAS DE CHI EN LA RONDA 0
    # ============================================================

    left_b0 = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    right_b0 = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    # Valor local decimal 11:
    #
    #     [1, 1, 0, 1, 0]
    #
    # colocado en:
    #
    #     (y, k) = (0, 0)
    #     (y, k) = (1, 0)
    right_local_input = np.asarray(
        [1, 1, 0, 1, 0],
        dtype=np.int64,
    )

    right_b0[:, 0, 0] = right_local_input
    right_b0[:, 1, 0] = right_local_input

    expected_delta_b0 = np.bitwise_xor(
        left_b0,
        right_b0,
    )

    expected_active_round_0 = [
        (0, 0, 0),
        (0, 1, 0),
    ]

    recovered_reference_active_round_0 = [
        (0, y, k)
        for y in range(5)
        for k in range(z)
        if np.any(
            expected_delta_b0[:, y, k]
        )
    ]

    assert recovered_reference_active_round_0 == (
        expected_active_round_0
    )

    # ============================================================
    # CALCULAR LAS PREIMÁGENES A_0
    # ============================================================

    left_a0_flat, left_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                left_b0.reshape(-1)
            ),
        )
    )

    right_a0_flat, right_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                right_b0.reshape(-1)
            ),
        )
    )

    assert left_rank == state_bit_count
    assert right_rank == state_bit_count

    left_a0 = left_a0_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    right_a0 = right_a0_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    np.testing.assert_array_equal(
        rho_pi(theta(left_a0)),
        left_b0,
    )

    np.testing.assert_array_equal(
        rho_pi(theta(right_a0)),
        right_b0,
    )

    input_difference = np.bitwise_xor(
        left_a0,
        right_a0,
    )

    assert np.any(input_difference)

    # ============================================================
    # PROPAGACIÓN DE REFERENCIA HACIA LA RONDA 1
    # ============================================================

    left_after_chi = chi(left_b0)
    right_after_chi = chi(right_b0)

    expected_delta_a1 = np.bitwise_xor(
        left_after_chi,
        right_after_chi,
    )

    expected_delta_b1 = rho_pi(
        theta(expected_delta_a1)
    )

    expected_active_round_1 = [
        (1, y, k)
        for y in range(5)
        for k in range(z)
        if np.any(
            expected_delta_b1[:, y, k]
        )
    ]

    assert expected_active_round_1 == [
        (1, 0, 0),
        (1, 3, 4),
    ]

    assert int(
        expected_delta_b1.sum()
    ) == 2

    # ============================================================
    # MODELO MILP DE DOS RONDAS
    # ============================================================

    config = ExperimentConfig(
        z=z,
        rounds=2,
        solver="cbc",
        time_limit_seconds=60,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(
        config=config,
        name="two_round_constructed_witness_z8_two_plus_two",
    )

    model.build_paired_model()

    model.add_nonzero_input_difference_constraint()

    model.add_active_sbox_upper_bound(
        max_active_sboxes=4,
    )

    # Fijar completamente los estados iniciales izquierdo
    # y derecho.
    for x in range(5):
        for y in range(5):
            for k in range(z):
                model.problem += (
                    model.left.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        left_a0[x, y, k]
                    ),
                    (
                        f"z8_two_plus_two_fix_left"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

                model.problem += (
                    model.right.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        right_a0[x, y, k]
                    ),
                    (
                        f"z8_two_plus_two_fix_right"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

    model.set_active_sbox_objective()

    solve_status = model.solve()

    assert solve_status == "Optimal"

    assert (
        pulp.LpStatus[
            model.problem.status
        ]
        == "Optimal"
    )

    assert model.problem.valid(
        eps=1e-7
    )

    # ============================================================
    # RECUPERAR DELTA B EN AMBAS RONDAS
    # ============================================================

    recovered_delta_b = []

    for round_index in range(2):
        round_difference = np.zeros(
            (5, 5, z),
            dtype=np.int64,
        )

        for x in range(5):
            for y in range(5):
                for k in range(z):
                    variable = (
                        model.chi_input_difference_variable(
                            round_index=round_index,
                            x=x,
                            y=y,
                            k=k,
                        )
                    )

                    value = variable.value()

                    assert value is not None

                    round_difference[
                        x,
                        y,
                        k,
                    ] = int(
                        value > 0.5
                    )

        recovered_delta_b.append(
            round_difference
        )

    np.testing.assert_array_equal(
        recovered_delta_b[0],
        expected_delta_b0,
    )

    np.testing.assert_array_equal(
        recovered_delta_b[1],
        expected_delta_b1,
    )

    # ============================================================
    # RECUPERAR LAS S-BOXES ACTIVAS
    # ============================================================

    recovered_active_by_round: dict[
        int,
        list[tuple[int, int, int]],
    ] = {
        0: [],
        1: [],
    }

    for round_index in range(2):
        for y in range(5):
            for k in range(z):
                variable = model.active_chi_variable(
                    round_index=round_index,
                    y=y,
                    k=k,
                )

                value = variable.value()

                assert value is not None

                if value > 0.5:
                    recovered_active_by_round[
                        round_index
                    ].append(
                        (
                            round_index,
                            y,
                            k,
                        )
                    )

    assert recovered_active_by_round[0] == (
        expected_active_round_0
    )

    assert recovered_active_by_round[1] == (
        expected_active_round_1
    )

    assert len(
        recovered_active_by_round[0]
    ) == 2

    assert len(
        recovered_active_by_round[1]
    ) == 2

    total_active = sum(
        len(positions)
        for positions
        in recovered_active_by_round.values()
    )

    assert total_active == 4

    assert model.objective_value() == pytest.approx(
        4.0
    )


def test_three_round_constructed_witness_z4_has_two_plus_two_plus_nine_active_sboxes() -> None:
    """
    Valida mediante el MILP exacto el mejor testigo encontrado
    dentro de las trayectorias 2+2 para z=4 y tres rondas:

        ronda 0: 2 S-boxes activas;
        ronda 1: 2 S-boxes activas;
        ronda 2: 9 S-boxes activas;
        total  : 13 S-boxes activas.
    """
    import numpy as np
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )
    from keccak_milp.layers import (
        chi,
        iota,
        keccak_round,
        rho_pi,
        theta,
    )

    z = 4
    rounds = 3
    state_bit_count = 5 * 5 * z

    # ============================================================
    # UTILIDADES LOCALES
    # ============================================================

    def integer_to_five_bits(
        value: int,
    ) -> np.ndarray:
        return np.asarray(
            [
                (value >> x) & 1
                for x in range(5)
            ],
            dtype=np.int64,
        )

    def five_bits_to_integer(
        bits: np.ndarray,
    ) -> int:
        return sum(
            int(bits[x]) << x
            for x in range(5)
        )

    def chi_five_bits(
        bits: np.ndarray,
    ) -> np.ndarray:
        input_bits = np.asarray(
            bits,
            dtype=np.int64,
        )

        output_bits = np.zeros(
            5,
            dtype=np.int64,
        )

        for x in range(5):
            output_bits[x] = (
                int(input_bits[x])
                ^ (
                    (
                        1
                        - int(
                            input_bits[
                                (x + 1) % 5
                            ]
                        )
                    )
                    & int(
                        input_bits[
                            (x + 2) % 5
                        ]
                    )
                )
            )

        return output_bits

    def active_positions(
        difference_state: np.ndarray,
        round_index: int,
    ) -> list[tuple[int, int, int]]:
        return [
            (
                round_index,
                y,
                k,
            )
            for y in range(5)
            for k in range(z)
            if np.any(
                difference_state[:, y, k]
            )
        ]

    def inverse_chi_state(
        output_state: np.ndarray,
    ) -> np.ndarray:
        chi_inverse: dict[int, int] = {}

        for input_integer in range(32):
            output_integer = five_bits_to_integer(
                chi_five_bits(
                    integer_to_five_bits(
                        input_integer
                    )
                )
            )

            chi_inverse[
                output_integer
            ] = input_integer

        assert len(chi_inverse) == 32

        input_state = np.zeros(
            (5, 5, z),
            dtype=np.int64,
        )

        for y in range(5):
            for k in range(z):
                output_integer = (
                    five_bits_to_integer(
                        output_state[:, y, k]
                    )
                )

                input_state[
                    :,
                    y,
                    k,
                ] = integer_to_five_bits(
                    chi_inverse[
                        output_integer
                    ]
                )

        np.testing.assert_array_equal(
            chi(input_state),
            output_state,
        )

        return input_state


    # ============================================================
    # MATRIZ DE L = rho_pi o theta
    # ============================================================

    linear_matrix = np.zeros(
        (
            state_bit_count,
            state_bit_count,
        ),
        dtype=np.uint8,
    )

    for input_bit_index in range(
        state_bit_count
    ):
        basis_state = np.zeros(
            (5, 5, z),
            dtype=np.int64,
        )

        basis_state.reshape(-1)[
            input_bit_index
        ] = 1

        transformed_basis = rho_pi(
            theta(basis_state)
        )

        linear_matrix[
            :,
            input_bit_index,
        ] = transformed_basis.reshape(
            -1
        ).astype(np.uint8)

    # ============================================================
    # TRAYECTORIA DIFERENCIAL ÓPTIMA 2+2
    # ============================================================

    # Diferencias de salida de chi en la ronda 0:
    #
    # soporte:
    #     (y, k) = (2, 0)
    #     (y, k) = (4, 0)
    #
    # beta decimal:
    #     1, 1
    delta_a1 = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    beta_bits = integer_to_five_bits(
        1
    )

    delta_a1[:, 2, 0] = beta_bits
    delta_a1[:, 4, 0] = beta_bits

    delta_b1 = rho_pi(
        theta(delta_a1)
    )

    expected_active_round_1 = (
        active_positions(
            delta_b1,
            round_index=1,
        )
    )

    assert expected_active_round_1 == [
        (1, 1, 3),
        (1, 2, 2),
    ]

    # ============================================================
    # REALIZACIÓN CONCRETA DE LA RONDA 1
    # ============================================================

    # Valores izquierdos encontrados:
    #
    #     posición (1, 3): decimal 2
    #     posición (2, 2): decimal 8
    left_b1 = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    left_b1[:, 1, 3] = (
        integer_to_five_bits(2)
    )

    left_b1[:, 2, 2] = (
        integer_to_five_bits(8)
    )

    right_b1 = np.bitwise_xor(
        left_b1,
        delta_b1,
    )

    delta_a2 = np.bitwise_xor(
        chi(left_b1),
        chi(right_b1),
    )

    delta_b2 = rho_pi(
        theta(delta_a2)
    )

    expected_active_round_2 = (
        active_positions(
            delta_b2,
            round_index=2,
        )
    )

    assert expected_active_round_2 == [
        (2, 0, 0),
        (2, 0, 2),
        (2, 1, 1),
        (2, 2, 0),
        (2, 2, 1),
        (2, 3, 2),
        (2, 4, 1),
        (2, 4, 2),
        (2, 4, 3),
    ]

    assert int(
        delta_b2.sum()
    ) == 12

    # ============================================================
    # RECUPERAR A_1 MEDIANTE L^{-1}
    # ============================================================

    left_a1_flat, left_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                left_b1.reshape(-1)
            ),
        )
    )

    right_a1_flat, right_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                right_b1.reshape(-1)
            ),
        )
    )

    assert left_rank == state_bit_count
    assert right_rank == state_bit_count

    left_a1 = left_a1_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    right_a1 = right_a1_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    np.testing.assert_array_equal(
        rho_pi(theta(left_a1)),
        left_b1,
    )

    np.testing.assert_array_equal(
        rho_pi(theta(right_a1)),
        right_b1,
    )

    # ============================================================
    # RECUPERAR B_0 Y A_0
    # ============================================================

    # A_1 = iota(C_0, 0), por lo que iota vuelve a aplicar
    # la misma constante y recupera C_0.
    left_c0 = iota(
        left_a1,
        round_index=0,
    )

    right_c0 = iota(
        right_a1,
        round_index=0,
    )

    np.testing.assert_array_equal(
        np.bitwise_xor(
            left_c0,
            right_c0,
        ),
        delta_a1,
    )

    left_b0 = inverse_chi_state(
        left_c0
    )

    right_b0 = inverse_chi_state(
        right_c0
    )

    delta_b0 = np.bitwise_xor(
        left_b0,
        right_b0,
    )

    expected_active_round_0 = (
        active_positions(
            delta_b0,
            round_index=0,
        )
    )

    assert len(
        expected_active_round_0
    ) == 2

    left_a0_flat, left_a0_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                left_b0.reshape(-1)
            ),
        )
    )

    right_a0_flat, right_a0_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                right_b0.reshape(-1)
            ),
        )
    )

    assert left_a0_rank == state_bit_count
    assert right_a0_rank == state_bit_count

    left_a0 = left_a0_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    right_a0 = right_a0_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    # ============================================================
    # VALIDACIÓN DE REFERENCIA
    # ============================================================

    np.testing.assert_array_equal(
        keccak_round(
            left_a0,
            round_index=0,
        ),
        left_a1,
    )

    np.testing.assert_array_equal(
        keccak_round(
            right_a0,
            round_index=0,
        ),
        right_a1,
    )

    left_a2 = keccak_round(
        left_a1,
        round_index=1,
    )

    right_a2 = keccak_round(
        right_a1,
        round_index=1,
    )

    recovered_delta_b2 = np.bitwise_xor(
        rho_pi(theta(left_a2)),
        rho_pi(theta(right_a2)),
    )

    np.testing.assert_array_equal(
        recovered_delta_b2,
        delta_b2,
    )

    # ============================================================
    # MODELO MILP DE TRES RONDAS
    # ============================================================

    config = ExperimentConfig(
        z=z,
        rounds=rounds,
        solver="cbc",
        time_limit_seconds=60,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(
        config=config,
        name="three_round_constructed_z4_two_two_nine",
    )

    model.build_paired_model()
    model.add_nonzero_input_difference_constraint()

    model.add_active_sbox_upper_bound(
        max_active_sboxes=13,
    )

    for x in range(5):
        for y in range(5):
            for k in range(z):
                model.problem += (
                    model.left.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        left_a0[x, y, k]
                    ),
                    (
                        f"three_round_13_fix_left"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

                model.problem += (
                    model.right.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        right_a0[x, y, k]
                    ),
                    (
                        f"three_round_13_fix_right"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

    model.set_active_sbox_objective()

    solve_status = model.solve()

    assert solve_status == "Optimal"

    assert (
        pulp.LpStatus[
            model.problem.status
        ]
        == "Optimal"
    )

    assert model.problem.valid(
        eps=1e-7
    )

    # ============================================================
    # RECUPERAR ACTIVIDAD DEL MILP
    # ============================================================

    recovered_active_by_round: dict[
        int,
        list[tuple[int, int, int]],
    ] = {
        0: [],
        1: [],
        2: [],
    }

    for round_index in range(rounds):
        for y in range(5):
            for k in range(z):
                variable = model.active_chi_variable(
                    round_index=round_index,
                    y=y,
                    k=k,
                )

                value = variable.value()

                assert value is not None

                if value > 0.5:
                    recovered_active_by_round[
                        round_index
                    ].append(
                        (
                            round_index,
                            y,
                            k,
                        )
                    )

    assert recovered_active_by_round[0] == (
        expected_active_round_0
    )

    assert recovered_active_by_round[1] == (
        expected_active_round_1
    )

    assert recovered_active_by_round[2] == (
        expected_active_round_2
    )

    assert len(
        recovered_active_by_round[0]
    ) == 2

    assert len(
        recovered_active_by_round[1]
    ) == 2

    assert len(
        recovered_active_by_round[2]
    ) == 9

    total_active = sum(
        len(positions)
        for positions
        in recovered_active_by_round.values()
    )

    assert total_active == 13

    assert model.objective_value() == pytest.approx(
        13.0
    )

def test_three_round_constructed_witness_z8_has_two_plus_two_plus_ten_active_sboxes() -> None:
    """
    Valida mediante el MILP exacto el mejor testigo encontrado
    dentro de las trayectorias 2+2 para z=8 y tres rondas:

        ronda 0: 2 S-boxes activas;
        ronda 1: 2 S-boxes activas;
        ronda 2: 10 S-boxes activas;
        total  : 14 S-boxes activas.
    """
    import numpy as np
    import pulp

    from keccak_milp.active_sboxes import (
        ActiveSBoxPairedKeccakMILPModel,
    )
    from keccak_milp.config import ExperimentConfig
    from keccak_milp.layers import (
        chi,
        iota,
        keccak_round,
        rho_pi,
        theta,
    )

    z = 8
    rounds = 3
    state_bit_count = 5 * 5 * z

    # ============================================================
    # UTILIDADES LOCALES
    # ============================================================

    def integer_to_five_bits(
        value: int,
    ) -> np.ndarray:
        if value not in range(32):
            raise ValueError(
                "El valor debe encontrarse entre 0 y 31."
            )

        return np.asarray(
            [
                (value >> x) & 1
                for x in range(5)
            ],
            dtype=np.int64,
        )

    def five_bits_to_integer(
        bits: np.ndarray,
    ) -> int:
        array = np.asarray(
            bits,
            dtype=np.int64,
        )

        if array.shape != (5,):
            raise ValueError(
                "Se esperaban exactamente cinco bits."
            )

        return sum(
            int(array[x]) << x
            for x in range(5)
        )

    def chi_five_bits(
        bits: np.ndarray,
    ) -> np.ndarray:
        input_bits = np.asarray(
            bits,
            dtype=np.int64,
        )

        if input_bits.shape != (5,):
            raise ValueError(
                "La entrada local de chi debe tener cinco bits."
            )

        output_bits = np.zeros(
            5,
            dtype=np.int64,
        )

        for x in range(5):
            output_bits[x] = (
                int(input_bits[x])
                ^ (
                    (
                        1
                        - int(
                            input_bits[
                                (x + 1) % 5
                            ]
                        )
                    )
                    & int(
                        input_bits[
                            (x + 2) % 5
                        ]
                    )
                )
            )

        return output_bits

    def active_positions(
        difference_state: np.ndarray,
        round_index: int,
    ) -> list[tuple[int, int, int]]:
        return [
            (
                round_index,
                y,
                k,
            )
            for y in range(5)
            for k in range(z)
            if np.any(
                difference_state[:, y, k]
            )
        ]

    # ============================================================
    # INVERSA LOCAL DE CHI
    # ============================================================

    chi_inverse: dict[int, int] = {}

    for input_integer in range(32):
        output_integer = five_bits_to_integer(
            chi_five_bits(
                integer_to_five_bits(
                    input_integer
                )
            )
        )

        if output_integer in chi_inverse:
            raise AssertionError(
                "Chi local no produjo una permutacion."
            )

        chi_inverse[
            output_integer
        ] = input_integer

    assert len(chi_inverse) == 32
    assert set(chi_inverse) == set(
        range(32)
    )

    def inverse_chi_state(
        output_state: np.ndarray,
    ) -> np.ndarray:
        input_state = np.zeros(
            (5, 5, z),
            dtype=np.int64,
        )

        for y in range(5):
            for k in range(z):
                output_integer = (
                    five_bits_to_integer(
                        output_state[:, y, k]
                    )
                )

                input_integer = chi_inverse[
                    output_integer
                ]

                input_state[
                    :,
                    y,
                    k,
                ] = integer_to_five_bits(
                    input_integer
                )

        np.testing.assert_array_equal(
            chi(input_state),
            output_state,
        )

        return input_state

    # ============================================================
    # MATRIZ DE L = rho_pi o theta
    # ============================================================

    linear_matrix = np.zeros(
        (
            state_bit_count,
            state_bit_count,
        ),
        dtype=np.uint8,
    )

    for input_bit_index in range(
        state_bit_count
    ):
        basis_state = np.zeros(
            (5, 5, z),
            dtype=np.int64,
        )

        basis_state.reshape(-1)[
            input_bit_index
        ] = 1

        transformed_basis = rho_pi(
            theta(basis_state)
        )

        linear_matrix[
            :,
            input_bit_index,
        ] = transformed_basis.reshape(
            -1
        ).astype(np.uint8)

    # ============================================================
    # DIFERENCIA DE SALIDA DE CHI EN LA RONDA 0
    #
    # Soporte:
    #     (y, k) = (2, 0)
    #     (y, k) = (4, 0)
    #
    # Beta:
    #     decimal 2 en ambas posiciones
    # ============================================================

    delta_a1 = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    beta_bits = integer_to_five_bits(
        2
    )

    delta_a1[
        :,
        2,
        0,
    ] = beta_bits

    delta_a1[
        :,
        4,
        0,
    ] = beta_bits

    delta_b1 = rho_pi(
        theta(delta_a1)
    )

    expected_active_round_1 = (
        active_positions(
            difference_state=delta_b1,
            round_index=1,
        )
    )

    assert expected_active_round_1 == [
        (1, 3, 2),
        (1, 4, 2),
    ]

    # ============================================================
    # REALIZACION CONCRETA EN LA ENTRADA DE CHI DE LA RONDA 1
    #
    # Valores izquierdos:
    #     posicion (3, 2): decimal 2
    #     posicion (4, 2): decimal 0
    # ============================================================

    left_b1 = np.zeros(
        (5, 5, z),
        dtype=np.int64,
    )

    left_b1[
        :,
        3,
        2,
    ] = integer_to_five_bits(
        2
    )

    left_b1[
        :,
        4,
        2,
    ] = integer_to_five_bits(
        0
    )

    right_b1 = np.bitwise_xor(
        left_b1,
        delta_b1,
    )

    np.testing.assert_array_equal(
        np.bitwise_xor(
            left_b1,
            right_b1,
        ),
        delta_b1,
    )

    delta_a2 = np.bitwise_xor(
        chi(left_b1),
        chi(right_b1),
    )

    delta_b2 = rho_pi(
        theta(delta_a2)
    )

    expected_active_round_2 = (
        active_positions(
            difference_state=delta_b2,
            round_index=2,
        )
    )

    assert expected_active_round_2 == [
        (2, 0, 0),
        (2, 0, 2),
        (2, 1, 5),
        (2, 1, 7),
        (2, 2, 4),
        (2, 3, 1),
        (2, 3, 3),
        (2, 3, 6),
        (2, 4, 2),
        (2, 4, 3),
    ]

    assert int(
        delta_b2.sum()
    ) == 13

    # ============================================================
    # RECUPERAR A_1 MEDIANTE L^{-1}
    # ============================================================

    left_a1_flat, left_a1_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                left_b1.reshape(-1)
            ),
        )
    )

    right_a1_flat, right_a1_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                right_b1.reshape(-1)
            ),
        )
    )

    assert left_a1_rank == state_bit_count
    assert right_a1_rank == state_bit_count

    left_a1 = left_a1_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    right_a1 = right_a1_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    np.testing.assert_array_equal(
        rho_pi(theta(left_a1)),
        left_b1,
    )

    np.testing.assert_array_equal(
        rho_pi(theta(right_a1)),
        right_b1,
    )

    # ============================================================
    # RECUPERAR LA SALIDA DE CHI DE LA RONDA 0
    #
    # A_1 = iota(C_0, 0).
    # Iota se invierte aplicando nuevamente la misma constante.
    # ============================================================

    left_c0 = iota(
        left_a1,
        round_index=0,
    )

    right_c0 = iota(
        right_a1,
        round_index=0,
    )

    np.testing.assert_array_equal(
        np.bitwise_xor(
            left_c0,
            right_c0,
        ),
        delta_a1,
    )

    # ============================================================
    # RECUPERAR B_0 Y SU DIFERENCIA
    # ============================================================

    left_b0 = inverse_chi_state(
        left_c0
    )

    right_b0 = inverse_chi_state(
        right_c0
    )

    delta_b0 = np.bitwise_xor(
        left_b0,
        right_b0,
    )

    expected_active_round_0 = (
        active_positions(
            difference_state=delta_b0,
            round_index=0,
        )
    )

    assert len(
        expected_active_round_0
    ) == 2

    # ============================================================
    # RECUPERAR A_0 MEDIANTE L^{-1}
    # ============================================================

    left_a0_flat, left_a0_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                left_b0.reshape(-1)
            ),
        )
    )

    right_a0_flat, right_a0_rank = (
        _solve_linear_system_gf2(
            matrix=linear_matrix,
            right_hand_side=(
                right_b0.reshape(-1)
            ),
        )
    )

    assert left_a0_rank == state_bit_count
    assert right_a0_rank == state_bit_count

    left_a0 = left_a0_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    right_a0 = right_a0_flat.reshape(
        (5, 5, z)
    ).astype(np.int64)

    np.testing.assert_array_equal(
        rho_pi(theta(left_a0)),
        left_b0,
    )

    np.testing.assert_array_equal(
        rho_pi(theta(right_a0)),
        right_b0,
    )

    # ============================================================
    # VALIDACION COMPLETA DE LAS RONDAS 0 Y 1
    # ============================================================

    np.testing.assert_array_equal(
        keccak_round(
            left_a0,
            round_index=0,
        ),
        left_a1,
    )

    np.testing.assert_array_equal(
        keccak_round(
            right_a0,
            round_index=0,
        ),
        right_a1,
    )

    left_a2 = keccak_round(
        left_a1,
        round_index=1,
    )

    right_a2 = keccak_round(
        right_a1,
        round_index=1,
    )

    recovered_delta_b2 = np.bitwise_xor(
        rho_pi(theta(left_a2)),
        rho_pi(theta(right_a2)),
    )

    np.testing.assert_array_equal(
        recovered_delta_b2,
        delta_b2,
    )

    # ============================================================
    # MODELO MILP DE TRES RONDAS
    # ============================================================

    config = ExperimentConfig(
        z=z,
        rounds=rounds,
        solver="cbc",
        time_limit_seconds=60,
        verbose=False,
    )

    model = ActiveSBoxPairedKeccakMILPModel(
        config=config,
        name=(
            "three_round_constructed_z8_"
            "two_two_ten"
        ),
    )

    model.build_paired_model()

    model.add_nonzero_input_difference_constraint()

    model.add_active_sbox_upper_bound(
        max_active_sboxes=14,
    )

    # Fijar completamente ambos estados iniciales.
    for x in range(5):
        for y in range(5):
            for k in range(z):
                model.problem += (
                    model.left.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        left_a0[x, y, k]
                    ),
                    (
                        f"three_round_z8_14_fix_left"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

                model.problem += (
                    model.right.state_variable(
                        round_index=0,
                        x=x,
                        y=y,
                        k=k,
                    )
                    == int(
                        right_a0[x, y, k]
                    ),
                    (
                        f"three_round_z8_14_fix_right"
                        f"_x{x}_y{y}_k{k}"
                    ),
                )

    model.set_active_sbox_objective()

    solve_status = model.solve()

    assert solve_status == "Optimal"

    assert (
        pulp.LpStatus[
            model.problem.status
        ]
        == "Optimal"
    )

    assert model.problem.valid(
        eps=1e-7
    )

    # ============================================================
    # RECUPERAR ACTIVIDAD DEL MILP
    # ============================================================

    recovered_active_by_round: dict[
        int,
        list[tuple[int, int, int]],
    ] = {
        0: [],
        1: [],
        2: [],
    }

    for round_index in range(rounds):
        for y in range(5):
            for k in range(z):
                variable = (
                    model.active_chi_variable(
                        round_index=round_index,
                        y=y,
                        k=k,
                    )
                )

                value = variable.value()

                assert value is not None

                if value > 0.5:
                    recovered_active_by_round[
                        round_index
                    ].append(
                        (
                            round_index,
                            y,
                            k,
                        )
                    )

    assert recovered_active_by_round[0] == (
        expected_active_round_0
    )

    assert recovered_active_by_round[1] == (
        expected_active_round_1
    )

    assert recovered_active_by_round[2] == (
        expected_active_round_2
    )

    assert len(
        recovered_active_by_round[0]
    ) == 2

    assert len(
        recovered_active_by_round[1]
    ) == 2

    assert len(
        recovered_active_by_round[2]
    ) == 10

    total_active = sum(
        len(positions)
        for positions
        in recovered_active_by_round.values()
    )

    assert total_active == 14

    assert model.objective_value() == pytest.approx(
        14.0
    )
