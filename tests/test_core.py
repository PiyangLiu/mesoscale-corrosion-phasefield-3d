import numpy as np
import pytest

from corrosion3d.core import generate_microstructure_3d
from corrosion3d.solver import (
    EngineeringParameters,
    _advance_unilateral_wedge_state,
    _solve_unilateral_wedge_contact,
    _unilateral_wedge_pressure,
    _update_rust_state,
    prepare_3d_solver,
    run_accelerated_corrosion_3d,
    run_full_3d,
)
from residual_opening_study import opening_summary


def test_residual_opening_summary_uses_final_instantaneous_width():
    history = {
        "surface_crack_width_mm": np.asarray([0.049, 0.0501, 0.0498]),
        "surface_crack_width_envelope_mm": np.asarray([0.049, 0.0501, 0.0501]),
    }
    result = opening_summary(history, opening_event=0.1)
    assert result["peak_opening_mm"] == pytest.approx(0.0501)
    assert result["opening_envelope_mm"] == pytest.approx(0.0501)
    assert result["final_current_opening_mm"] == pytest.approx(0.0498)
    assert result["unloaded_residual_opening_mm"] == pytest.approx(0.0498)
    assert result["unloading_closure_mm"] == pytest.approx(0.0003)


def test_unilateral_wedge_is_conservative_irreversible_and_compression_only():
    parameters = EngineeringParameters(
        aggregate_fraction=0.0,
        grid_spacing=0.01,
        domain_width=0.04,
        domain_height=0.06,
        domain_thickness=0.02,
        cover_depth=0.015,
        rebar_radius=0.005,
        seed=17,
    )
    simulation, _, _ = prepare_3d_solver(parameters, dt=0.01)
    generated = np.zeros(simulation.shape_3d)
    generated[simulation.rebar_bnd] = 4.0e-5
    result = _advance_unilateral_wedge_state(
        simulation,
        generated,
        0.01,
        np.zeros_like(generated),
        np.zeros_like(generated),
        4.0e6,
        5.0e-4,
        1.0,
        4.0,
        4.0,
        5.0e-5,
        0.75,
        1.0,
        0.0,
    )
    assert result["generated_volume"] > 0.0
    assert np.max(simulation.residual_wedge) > 0.0
    assert np.allclose(
        simulation.rust_mobile + simulation.rust_compacted + simulation.rust_lost,
        generated,
        rtol=0.0,
        atol=1.0e-15,
    )
    wedge_before = simulation.residual_wedge.copy()
    _advance_unilateral_wedge_state(
        simulation,
        np.zeros_like(generated),
        0.01,
        np.zeros_like(generated),
        np.zeros_like(generated),
        4.0e6,
        5.0e-4,
        1.0,
        4.0,
        4.0,
        5.0e-5,
        0.75,
        1.0,
        0.0,
    )
    assert np.all(simulation.residual_wedge >= wedge_before)
    pressure = _unilateral_wedge_pressure(
        simulation, 2.0 * simulation.residual_wedge, 4.0e11
    )
    assert np.max(pressure) == 0.0


def test_unilateral_wedge_does_not_instantly_discard_new_rust():
    parameters = EngineeringParameters(
        aggregate_fraction=0.0,
        grid_spacing=0.01,
        domain_width=0.04,
        domain_height=0.06,
        domain_thickness=0.02,
        cover_depth=0.015,
        rebar_radius=0.005,
        seed=18,
    )
    simulation, _, _ = prepare_3d_solver(parameters, dt=0.01)
    simulation.d[simulation.rebar_bnd] = 1.0
    generated = np.zeros(simulation.shape_3d)
    generated[simulation.rebar_bnd] = 4.0e-5
    _advance_unilateral_wedge_state(
        simulation,
        generated,
        0.01,
        np.zeros_like(generated),
        np.zeros_like(generated),
        4.0e6,
        5.0e-4,
        1.0,
        4.0,
        4.0,
        5.0e-5,
        0.75,
        1.0,
        0.0,
    )
    assert np.max(simulation.rust_lost) == 0.0
    np.testing.assert_allclose(
        simulation.rust_mobile + simulation.rust_compacted,
        generated,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_unilateral_contact_satisfies_complementarity():
    parameters = EngineeringParameters(
        aggregate_fraction=0.0,
        grid_spacing=0.01,
        domain_width=0.04,
        domain_height=0.06,
        domain_thickness=0.02,
        cover_depth=0.015,
        rebar_radius=0.005,
        seed=19,
    )
    simulation, _, _ = prepare_3d_solver(parameters, dt=0.01)
    simulation.residual_wedge[simulation.rebar_bnd] = 2.0e-5
    opening, residual = _solve_unilateral_wedge_contact(
        simulation,
        4.0e11,
        np.zeros(simulation.shape_3d),
    )
    assert np.all(opening >= 0.0)
    assert np.min(simulation.rust_pressure) >= 0.0
    assert residual <= 1.0e-4
    assert simulation.last_contact_min_separation_m >= -1.0e-8
    assert not simulation.last_contact_pdas_fallback_used


def test_matrix_free_contact_active_set_on_large_interface():
    class DiagonalComplianceSimulation:
        def __init__(self, size):
            self.rebar_bnd = np.ones(size, dtype=bool)
            self.rebar_area_weight = np.linspace(0.5, 1.5, size)
            self.residual_wedge = np.linspace(1.0e-5, 3.0e-5, size)
            self.rust_pressure = np.zeros(size)
            self.u_r = np.zeros(size)
            self.u = np.zeros(size)
            self.v = np.zeros(size)
            self.nx = np.ones(size)
            self.ny = np.zeros(size)

        def solve_mechanics_pseudo_transient(
            self, ur_map=None, pressure_map=None, update_history=True
        ):
            self.u[:] = 1.0e-12 * np.asarray(pressure_map)
            self.v[:] = 0.0

    simulation = DiagonalComplianceSimulation(300)
    opening, residual = _solve_unilateral_wedge_contact(
        simulation,
        4.0e11,
        np.zeros_like(simulation.rust_pressure),
    )
    expected_pressure = simulation.residual_wedge / (1.0e-12 + 1.0 / 4.0e11)
    np.testing.assert_allclose(
        simulation.rust_pressure, expected_pressure, rtol=2.0e-6, atol=1.0
    )
    assert np.all(opening >= 0.0)
    assert residual <= 1.0e-7
    assert simulation.last_contact_active_cells == 300


def test_matrix_free_contact_allows_nonmonotone_active_set_change():
    class GloballyCoupledComplianceSimulation:
        def __init__(self, size):
            self.rebar_bnd = np.ones(size, dtype=bool)
            self.rebar_area_weight = np.ones(size)
            self.residual_wedge = np.full(size, 1.0e-7)
            self.residual_wedge[::2] = 2.0e-5
            self.rust_pressure = np.zeros(size)
            self.u_r = np.zeros(size)
            self.u = np.zeros(size)
            self.v = np.zeros(size)
            self.nx = np.ones(size)
            self.ny = np.zeros(size)

        def solve_mechanics_pseudo_transient(
            self, ur_map=None, pressure_map=None, update_history=True
        ):
            pressure = np.asarray(pressure_map)
            self.u[:] = 2.0e-13 * pressure + 4.0e-12 * np.mean(pressure)
            self.v[:] = 0.0

    simulation = GloballyCoupledComplianceSimulation(300)
    opening, residual = _solve_unilateral_wedge_contact(
        simulation,
        4.0e11,
        np.zeros_like(simulation.rust_pressure),
    )
    assert np.all(opening >= 0.0)
    assert residual <= 1.0e-7
    assert simulation.last_contact_active_iterations >= 2
    assert np.all(simulation.rust_pressure[::2] > 0.0)
    assert np.max(simulation.rust_pressure[1::2]) <= 1.0


def test_microstructure_is_deterministic():
    arguments = dict(
        shape=(9, 13, 5),
        size=(0.04, 0.06, 0.02),
        rebar_radius=0.005,
        cover_depth=0.015,
        target_aggregate_fraction=0.10,
        itz_width=0.002,
        seed=7,
        max_attempts=1000,
    )
    first = generate_microstructure_3d(**arguments)
    second = generate_microstructure_3d(**arguments)
    assert np.array_equal(first.aggregate, second.aggregate)
    assert np.array_equal(first.itz_fraction, second.itz_fraction)


def test_short_three_dimensional_run():
    parameters = EngineeringParameters(
        case_id="ci_smoke_3d",
        cover_depth=0.015,
        rebar_radius=0.005,
        aggregate_fraction=0.0,
        grid_spacing=0.01,
        domain_width=0.04,
        domain_height=0.06,
        domain_thickness=0.02,
        seed=9,
        mechanics_max_iterations=20,
    )
    simulation, histories, summary = run_full_3d(
        total_years=0.5,
        dt=0.5,
        params=parameters,
        staggered_max_iterations=60,
        progress=False,
    )
    assert simulation.d.shape == simulation.shape_3d
    assert histories["times"].tolist() == [0.5]
    assert np.all(np.isfinite(simulation.C))
    assert abs(histories["rust_mass_balance_error_m"][-1]) <= 1.0e-12
    assert summary.max_damage >= 0.0


def test_depassivation_uses_end_of_step_time():
    parameters = EngineeringParameters(
        case_id="depassivation_clock",
        cover_depth=0.01,
        rebar_radius=0.004,
        critical_chloride=0.0,
        aggregate_fraction=0.0,
        grid_spacing=0.004,
        domain_width=0.04,
        domain_height=0.06,
        domain_thickness=0.016,
        mechanics_max_iterations=30,
    )
    simulation, histories, summary = run_full_3d(
        total_years=0.5,
        dt=0.5,
        params=parameters,
        staggered_max_iterations=10,
        progress=False,
    )
    assert summary.depassivation_year_min == 0.5
    assert np.min(simulation.t_ini_map[np.isfinite(simulation.t_ini_map)]) == 0.5
    assert histories["times"].tolist() == [0.5]


def test_axial_transport_control_removes_axial_coupling():
    parameters = EngineeringParameters(
        case_id="no_axial_transport",
        cover_depth=0.015,
        rebar_radius=0.005,
        aggregate_fraction=0.0,
        grid_spacing=0.005,
        domain_width=0.04,
        domain_height=0.06,
        domain_thickness=0.02,
        axial_transport_enabled=False,
        mechanics_max_iterations=20,
    )
    simulation, _, _ = run_full_3d(
        total_years=0.5,
        dt=0.5,
        params=parameters,
        progress=False,
    )
    np.testing.assert_allclose(simulation.C[:, :, 0], simulation.C[:, :, -1])
    assert np.max(np.abs(simulation.grad_phi_z)) == 0.0


def test_surface_opening_detector_does_not_require_localized_damage():
    parameters = EngineeringParameters(
        aggregate_fraction=0.0,
        grid_spacing=0.005,
        domain_width=0.04,
        domain_height=0.04,
        domain_thickness=0.02,
        cover_depth=0.025,
        rebar_radius=0.004,
    )
    simulation, _, _ = prepare_3d_solver(parameters, dt=0.5)
    simulation.d[:] = 0.0
    simulation.u[:] = 1.0e-3 * simulation.X
    opening_mm = simulation.estimate_surface_crack_width_mm()
    assert opening_mm > 0.015
    assert np.max(simulation.d) < 0.20


def test_pressure_overflow_is_separate_from_migration_loss():
    shape = (2, 1, 1)
    rust = np.full(shape, 2.0e-5)
    lost = np.zeros(shape)
    filled = np.zeros(shape)
    pressure = np.zeros(shape)
    active = np.ones(shape, dtype=bool)
    result = _update_rust_state(
        rust,
        lost,
        filled,
        pressure,
        np.zeros(shape),
        np.zeros(shape),
        np.ones(shape),
        1.0,
        active,
        8.0e-6,
        2.5,
        2.0e9,
        leakage_pressure=1.0e6,
        migration_base=0.1,
        migration_damage=0.0,
        new_product_loss_base=0.0,
        new_product_loss_damage=0.0,
    )
    assert np.sum(lost) > np.sum(result["overflow_removed_m"])
    assert result["overflow_equivalent_thickness_sum_m"] == np.sum(
        result["overflow_removed_m"]
    )


def test_spanning_damage_requires_one_interface_to_surface_component():
    parameters = EngineeringParameters(
        aggregate_fraction=0.0,
        grid_spacing=0.005,
        domain_width=0.04,
        domain_height=0.06,
        domain_thickness=0.02,
        cover_depth=0.025,
        rebar_radius=0.004,
    )
    simulation, _, _ = prepare_3d_solver(parameters, dt=0.5)
    simulation.d[:] = 0.0
    interface_cells = np.argwhere(simulation.rebar_bnd)
    ix, _, iz = map(int, interface_cells[0])
    same_column = interface_cells[
        (interface_cells[:, 0] == ix) & (interface_cells[:, 2] == iz)
    ]
    iy = int(np.max(same_column[:, 1]))
    simulation.d[ix, iy:, iz] = 0.25
    assert simulation.diagnostic_spanning_damage_component(0.20)[2]
    simulation.d[ix, iy + 1, iz] = 0.0
    assert not simulation.diagnostic_spanning_damage_component(0.20)[2]


def test_spanning_damage_rejects_corner_only_connectivity():
    parameters = EngineeringParameters(
        aggregate_fraction=0.0,
        grid_spacing=0.005,
        domain_width=0.04,
        domain_height=0.06,
        domain_thickness=0.02,
        cover_depth=0.025,
        rebar_radius=0.004,
    )
    simulation, _, _ = prepare_3d_solver(parameters, dt=0.5)
    simulation.d[:] = 0.0
    ix, iy, iz = map(int, np.argwhere(simulation.rebar_bnd)[0])
    while iy < simulation.Ny:
        simulation.d[ix, iy, iz] = 0.25
        ix = min(ix + 1, simulation.Nx - 1)
        iy += 1
    assert not simulation.diagnostic_spanning_damage_component(0.20)[2]


def test_accelerated_interface_stiffness_factor_is_applied():
    simulation, _, _ = run_accelerated_corrosion_3d(
        case_id="interface_stiffness",
        cover_depth_m=0.015,
        rebar_radius_m=0.005,
        tensile_strength_mpa=3.0,
        corrosion_current_ua_cm2=0.0,
        total_hours=1.0,
        grid_spacing=0.005,
        aggregate_fraction=0.0,
        domain_width_m=0.04,
        domain_height_m=0.06,
        domain_thickness_m=0.02,
        steel_interface_width_m=0.005,
        steel_interface_stiffness_factor=0.5,
    )
    np.testing.assert_allclose(
        simulation.E_map[simulation.rebar_bnd], 0.5 * simulation.E_mat_base
    )
