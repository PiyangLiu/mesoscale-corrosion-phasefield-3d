"""Coupled three-dimensional transport, corrosion-product, mechanics and phase-field solver."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pyamg
from scipy import ndimage, optimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .core import (
    FARADAY,
    M_FE,
    N_FE,
    RHO_FE,
    SECONDS_PER_YEAR,
    hoop_cracking_pressure,
)
from .simulation import CorrosionCracking3D


@dataclass
class SolverSummary:
    elapsed_s: float
    depassivation_year_min: float | None
    max_chloride: float
    max_electric_grad_v_m: float
    max_icorr_a_m2: float
    max_rust_pressure_mpa: float
    max_damage: float


@dataclass(frozen=True)
class EngineeringParameters:
    case_id: str = "baseline_3d"
    description: str = "baseline marine splash-zone reinforced concrete"
    cover_depth: float = 0.05
    rebar_radius: float = 0.01
    surface_chloride: float = 4.0
    critical_chloride: float = 1.2
    mortar_chloride_diffusivity: float = 0.0001
    itz_diffusivity_multiplier: float = 5.0
    reference_corrosion_current: float = 0.01
    rust_expansion_ratio: float = 2.5
    porous_gap: float = 8e-06
    crack_storage_coeff: float = 0.0012
    rust_bulk_modulus: float = 2000000000.0
    fracture_energy: float = 80.0
    phasefield_length: float = 0.005
    residual_stiffness: float = 1e-05
    fracture_heterogeneity_cv: float = 0.1
    fracture_heterogeneity_length: float = 0.01
    interface_flaw_factor: float = 1.0
    interface_flaw_radial_extent: float = 0.01
    interface_flaw_half_angle: float = 0.25
    oxygen_surface: float = 1.0
    oxygen_mode: str = "linear"
    oxygen_decay_alpha: float = 4.0
    oxygen_decay_center: float = 0.82
    aggregate_fraction: float = 0.3
    seed: int = 88
    grid_spacing: float | None = None
    domain_thickness: float | None = None
    domain_width: float | None = None
    domain_height: float | None = None
    itz_width: float = 0.003
    aggregate_records_path: str | None = None
    surface_gauge_length: float = 0.02
    steel_interface_width: float | None = None
    steel_interface_stiffness_factor: float = 1.0
    steel_interface_fracture_factor: float = 1.0
    surface_observation_depth: float | None = None
    mechanics_rtol: float = 1e-06
    mechanics_max_iterations: int = 1000
    transport_solver: str = "auto"
    transport_rtol: float = 1e-10
    transport_max_iterations: int = 1000
    pressure_limit_scale: float = 1.0
    pressure_limit_damage: float = 0.5
    leakage_law: str = "instantaneous_cap"
    leakage_rate: float = 0.25
    leakage_exponent: float = 1.0
    axial_transport_enabled: bool = True
    axial_symmetry_ends: bool = True
    notes: dict = field(default_factory=dict)


def _load_aggregate_records(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload["records"] if isinstance(payload, dict) else payload


def _fracture_heterogeneity_3d(sim, params: EngineeringParameters):
    cv = max(float(params.fracture_heterogeneity_cv), 0.0)
    if cv == 0.0:
        return np.ones(sim.shape_3d)
    correlation_length = max(
        float(params.fracture_heterogeneity_length), min(sim.dx, sim.dy, sim.dz)
    )
    rng = np.random.default_rng(int(params.seed) + 104729)
    field = np.zeros(sim.shape_3d)
    mode_count = 16
    for _ in range(mode_count):
        direction = rng.normal(size=3)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        wavelength = correlation_length * rng.uniform(0.75, 1.5)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        argument = (
            2.0
            * np.pi
            * (direction[0] * sim.X + direction[1] * sim.Y + direction[2] * sim.Z)
            / wavelength
            + phase
        )
        field += np.cos(argument)
    solid = ~sim.is_rebar
    values = field[solid]
    field = (field - float(np.mean(values))) / max(float(np.std(values)), 1e-12)
    log_sigma = np.sqrt(np.log1p(cv**2))
    factor = np.exp(log_sigma * field - 0.5 * log_sigma**2)
    return np.clip(factor, 0.7, 1.3)


class _DirectDirichletSolver:
    """Factorized transport solve with an explicit residual receipt."""

    def __init__(self, laplacian, n, dt, top_flat):
        self.penalty = 1.0e12
        matrix = sp.eye(n, format="csr") - dt * laplacian
        matrix = matrix.tolil()
        for row in np.flatnonzero(top_flat):
            matrix.rows[row] = [row]
            matrix.data[row] = [self.penalty]
        self.matrix = matrix.tocsr()
        self.factorized = spla.factorized(self.matrix.tocsc())
        self.last_relative_residual = np.inf
        self.last_iterations = 1

    def __call__(self, rhs):
        solution = self.factorized(np.asarray(rhs, dtype=float))
        residual = self.matrix @ solution - rhs
        self.last_relative_residual = float(
            np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0)
        )
        return solution


def _factorized_transport(laplacian, n, dt, top_flat, locked_flat):
    solver = _DirectDirichletSolver(laplacian, n, dt, top_flat)
    return (solver, solver.penalty)


class _IterativeDirichletSolver:
    """SPD elimination and AMG-preconditioned CG for large transport systems."""

    def __init__(self, laplacian, n, dt, top_flat, rtol, maxiter):
        self.penalty = 1.0e12
        self.boundary = np.asarray(top_flat, dtype=bool)
        self.rtol = float(rtol)
        self.maxiter = int(maxiter)
        base = sp.eye(n, format="csr") - dt * laplacian
        boundary_float = self.boundary.astype(float)
        self.boundary_coupling = np.asarray(base @ boundary_float).ravel()
        free = sp.diags((~self.boundary).astype(float), format="csr")
        self.matrix = (
            free @ base @ free + sp.diags(boundary_float, format="csr")
        ).tocsr()
        hierarchy = pyamg.smoothed_aggregation_solver(
            self.matrix, symmetry="symmetric", max_coarse=300
        )
        self.preconditioner = hierarchy.aspreconditioner(cycle="V")
        self.solution = np.zeros(n, dtype=float)
        self.last_relative_residual = np.inf
        self.last_iterations = 0

    def __call__(self, rhs):
        rhs = np.asarray(rhs, dtype=float)
        boundary_values = rhs[self.boundary] / self.penalty
        boundary_value = (
            float(np.mean(boundary_values)) if boundary_values.size else 0.0
        )
        reduced_rhs = rhs.copy()
        reduced_rhs[self.boundary] = 0.0
        reduced_rhs -= boundary_value * self.boundary_coupling
        reduced_rhs[self.boundary] = boundary_value
        self.solution[self.boundary] = boundary_value
        iteration_count = [0]

        def count_iteration(_):
            iteration_count[0] += 1

        solution, info = spla.cg(
            self.matrix,
            reduced_rhs,
            x0=self.solution,
            rtol=self.rtol,
            atol=0.0,
            maxiter=self.maxiter,
            M=self.preconditioner,
            callback=count_iteration,
        )
        residual = self.matrix @ solution - reduced_rhs
        relative_residual = float(
            np.linalg.norm(residual) / max(np.linalg.norm(reduced_rhs), 1.0)
        )
        if info != 0 or relative_residual > 5.0 * self.rtol:
            solution, info = spla.lgmres(
                self.matrix,
                reduced_rhs,
                x0=solution,
                rtol=self.rtol,
                atol=0.0,
                maxiter=min(self.maxiter, 100),
                M=self.preconditioner,
                inner_m=40,
                outer_k=4,
            )
            residual = self.matrix @ solution - reduced_rhs
            relative_residual = float(
                np.linalg.norm(residual) / max(np.linalg.norm(reduced_rhs), 1.0)
            )
        if info != 0 or relative_residual > 5.0 * self.rtol:
            raise RuntimeError(
                "transport equilibrium failed: "
                f"info={info}, relative_residual={relative_residual:.3e}"
            )
        self.solution = solution
        self.last_relative_residual = relative_residual
        self.last_iterations = iteration_count[0]
        return solution


def _prepare_transport_solver(laplacian, n, dt, top_flat, locked_flat, params):
    mode = str(params.transport_solver).lower()
    if mode == "auto":
        mode = "iterative" if n >= 100000 else "direct"
    if mode == "direct":
        return _factorized_transport(laplacian, n, dt, top_flat, locked_flat)
    if mode != "iterative":
        raise ValueError(f"unknown transport solver mode: {params.transport_solver}")
    solver = _IterativeDirichletSolver(
        laplacian,
        n,
        dt,
        top_flat,
        params.transport_rtol,
        params.transport_max_iterations,
    )
    return (solver, solver.penalty)


def _transport_residual(solver) -> float:
    return float(getattr(solver, "last_relative_residual", 0.0))


def _solve_dirichlet(
    solve, penalty, field, shape, top_flat, top_value, locked_flat, locked_value=0.0
):
    rhs = field.ravel().copy()
    rhs[top_flat] = penalty * top_value
    return solve(rhs).reshape(shape)


def _oxygen_availability(saturation, mode="linear", alpha=4.0, center=0.82):
    if mode == "constant":
        return np.ones_like(saturation, dtype=float)
    if mode == "nonmonotonic":
        decay = np.exp(-alpha * np.maximum(saturation - center, 0.0))
        return np.clip(decay, 0.0, 1.0)
    return np.clip(0.35 + 0.65 * saturation, 0.0, 1.0)


def _corrosion_current(
    c_kg,
    oh,
    saturation,
    damage,
    active,
    i0,
    oxygen_mode="linear",
    oxygen_alpha=4.0,
    oxygen_center=0.82,
):
    cl_mol = 1000.0 * np.maximum(c_kg, 0.0) / 35.45
    ratio = cl_mol / np.maximum(oh, 1e-12)
    chloride = 1.0 / (1.0 + np.exp(-10.0 * (np.log10(np.maximum(ratio, 1e-18)) + 1.35)))
    oxygen = _oxygen_availability(saturation, oxygen_mode, oxygen_alpha, oxygen_center)
    moisture = np.clip((saturation - 0.35) / 0.45, 0.0, 1.0)
    damage_factor = 1.0 + 1.5 * np.clip(damage, 0.0, 1.0)
    current = i0 * chloride * oxygen * moisture * damage_factor
    return np.where(active, np.minimum(current, 0.05), 0.0)


def _update_rust_state(
    rust,
    lost,
    filled,
    pressure,
    i_corr,
    damage,
    saturation,
    dt,
    rebar_bnd,
    gap,
    gamma,
    bulk,
    leakage_pressure=None,
    migration_base=0.05,
    migration_damage=0.85,
    new_product_loss_base=0.05,
    new_product_loss_damage=0.85,
    crack_storage_coeff=0.0012,
    leakage_damage_factor=0.5,
    leakage_pressure_scale=1.0,
    pressure_leakage_rate=0.0,
    mechanical_opening=None,
    leakage_driving_pressure=None,
    leakage_law="instantaneous_cap",
    leakage_rate=0.25,
    leakage_exponent=1.0,
):
    steel_loss = i_corr * M_FE * SECONDS_PER_YEAR * dt / (N_FE * FARADAY * RHO_FE)
    generated = gamma * steel_loss
    migration_rate = migration_base + migration_damage * np.clip(damage, 0.0, 1.0)
    new_product_loss_fraction = np.clip(
        new_product_loss_base + new_product_loss_damage * np.clip(damage, 0.0, 1.0),
        0.0,
        1.0,
    )
    migrated_existing = np.minimum(rust, migration_rate * dt * rust)
    migrated_new = new_product_loss_fraction * generated
    rust += np.maximum(generated - migrated_new, 0.0) - migrated_existing
    lost += migrated_existing + migrated_new
    d_clip = np.clip(damage, 0.0, 1.0)
    crack_capacity = crack_storage_coeff * d_clip**1.5
    capacity = gap + crack_capacity
    filled[:] = np.minimum(rust, capacity)
    excess = np.maximum(rust - capacity, 0.0)
    sat = np.clip((saturation - 0.35) / 0.55, 0.0, 1.0)
    effective_bulk = bulk * (1.0 - d_clip) ** 2
    stiffness = effective_bulk * sat
    pressure_raw = stiffness * excess / np.maximum(capacity, 1e-12)
    cap_triggered = 0
    overflow_amount_sum = 0.0
    overflow_removed = np.zeros_like(rust)
    if leakage_pressure is not None:
        threshold = (
            leakage_pressure_scale
            * leakage_pressure
            * (1.0 + leakage_damage_factor * (1.0 - d_clip))
        )
        allowed_excess = threshold * capacity / np.maximum(stiffness, 1.0e-18)
        overflow = np.maximum(excess - allowed_excess, 0.0)
        overflow[~rebar_bnd] = 0.0
        if leakage_law == "instantaneous_cap":
            removed = overflow
        elif leakage_law == "pressure_dependent":
            over_ratio = np.maximum(pressure_raw / np.maximum(threshold, 1.0) - 1.0, 0.0)
            fraction = 1.0 - np.exp(
                -max(float(leakage_rate), 0.0)
                * max(float(dt), 0.0)
                * over_ratio ** max(float(leakage_exponent), 0.0)
            )
            removed = overflow * np.clip(fraction, 0.0, 1.0)
        else:
            raise ValueError(f"unknown leakage_law: {leakage_law}")
        overflow_removed[:] = removed
        cap_triggered = int(np.count_nonzero(overflow > 0.0))
        overflow_amount_sum = float(np.sum(removed[rebar_bnd]))
        rust -= removed
        lost += removed
        filled[:] = np.minimum(rust, capacity)
        excess = np.maximum(rust - capacity, 0.0)
        pressure_raw = stiffness * excess / np.maximum(capacity, 1e-12)
    pressure[:] = pressure_raw
    pressure[~rebar_bnd] = 0.0
    rust[~rebar_bnd] = 0.0
    lost[~rebar_bnd] = 0.0
    filled[~rebar_bnd] = 0.0
    return {
        "cap_triggered_cells": cap_triggered,
        "overflow_equivalent_thickness_sum_m": overflow_amount_sum,
        "overflow_removed_m": overflow_removed,
        "generated_volume": float(np.sum(generated[rebar_bnd])),
        "retained_volume": float(np.sum(rust[rebar_bnd])),
        "lost_volume": float(np.sum(lost[rebar_bnd])),
    }


def _kinetic_leakage_equilibrium(
    retained_before_leakage,
    damage,
    saturation,
    dt,
    rebar_bnd,
    gap,
    bulk,
    leakage_pressure,
    mechanical_opening,
    leakage_permeability,
    pressure_scale,
    leakage_exponent,
    leakage_damage_factor,
    leakage_opening_factor,
    leakage_opening_reference,
):
    d_clip = np.clip(damage, 0.0, 1.0)
    opening = np.maximum(np.asarray(mechanical_opening, dtype=float), 0.0)
    capacity = gap + opening
    saturation_factor = np.clip((saturation - 0.35) / 0.55, 0.0, 1.0)
    damage_multiplier = 1.0 + max(float(leakage_damage_factor), 0.0) * d_clip
    opening_multiplier = 1.0 + max(float(leakage_opening_factor), 0.0) * (
        opening / max(float(leakage_opening_reference), 1.0e-12)
    )
    trial_pressure = (
        bulk
        * saturation_factor
        * np.maximum(retained_before_leakage - capacity, 0.0)
        / np.maximum(capacity, 1.0e-12)
    )
    lower = np.minimum(retained_before_leakage, capacity)
    upper = retained_before_leakage.copy()
    coefficient = (
        max(float(dt), 0.0)
        * max(float(leakage_permeability), 0.0)
        * damage_multiplier
        * opening_multiplier
    )
    exponent = max(float(leakage_exponent), 0.0)
    pressure_denominator = np.maximum(capacity, 1.0e-12)
    for _ in range(40):
        midpoint = 0.5 * (lower + upper)
        midpoint_pressure = (
            bulk
            * saturation_factor
            * np.maximum(midpoint - capacity, 0.0)
            / pressure_denominator
        )
        midpoint_overpressure = np.maximum(
            (midpoint_pressure - leakage_pressure)
            / max(float(pressure_scale), 1.0),
            0.0,
        )
        residual = (
            midpoint
            + coefficient * midpoint_overpressure**exponent
            - retained_before_leakage
        )
        move_lower = residual < 0.0
        lower = np.where(move_lower, midpoint, lower)
        upper = np.where(move_lower, upper, midpoint)
    retained_after = 0.5 * (lower + upper)
    removed = np.maximum(retained_before_leakage - retained_after, 0.0)
    removed[~rebar_bnd] = 0.0
    filled = np.minimum(retained_after, capacity)
    pressure = (
        bulk
        * saturation_factor
        * np.maximum(retained_after - capacity, 0.0)
        / pressure_denominator
    )
    pressure[~rebar_bnd] = 0.0
    return retained_after, removed, filled, pressure, trial_pressure


def _update_rust_state_kinetic(
    rust,
    lost,
    filled,
    pressure,
    i_corr,
    damage,
    saturation,
    dt,
    rebar_bnd,
    gap,
    gamma,
    bulk,
    leakage_pressure,
    mechanical_opening,
    leakage_permeability,
    pressure_scale,
    leakage_exponent=1.0,
    leakage_damage_factor=0.0,
    leakage_opening_factor=0.0,
    leakage_opening_reference=5.0e-5,
    migration_base=0.05,
    migration_damage=0.85,
    new_product_loss_base=0.05,
    new_product_loss_damage=0.85,
):
    """Advance retained product with a finite-rate pressure-driven leakage law."""
    steel_loss = i_corr * M_FE * SECONDS_PER_YEAR * dt / (N_FE * FARADAY * RHO_FE)
    generated = gamma * steel_loss
    d_clip = np.clip(damage, 0.0, 1.0)
    migrated_existing = np.minimum(
        rust, (migration_base + migration_damage * d_clip) * dt * rust
    )
    new_loss_fraction = np.clip(
        new_product_loss_base + new_product_loss_damage * d_clip, 0.0, 1.0
    )
    migrated_new = new_loss_fraction * generated
    rust += np.maximum(generated - migrated_new, 0.0) - migrated_existing
    lost += migrated_existing + migrated_new

    retained_before_leakage = rust.copy()
    retained_after, removed, filled_after, pressure_after, trial_pressure = (
        _kinetic_leakage_equilibrium(
            retained_before_leakage,
            d_clip,
            saturation,
            dt,
            rebar_bnd,
            gap,
            bulk,
            leakage_pressure,
            mechanical_opening,
            leakage_permeability,
            pressure_scale,
            leakage_exponent,
            leakage_damage_factor,
            leakage_opening_factor,
            leakage_opening_reference,
        )
    )
    rust[:] = retained_after
    lost += removed
    filled[:] = filled_after
    pressure[:] = pressure_after
    rust[~rebar_bnd] = 0.0
    lost[~rebar_bnd] = 0.0
    filled[~rebar_bnd] = 0.0
    return {
        "cap_triggered_cells": int(
            np.count_nonzero((trial_pressure > leakage_pressure) & rebar_bnd)
        ),
        "overflow_removed_m": removed,
        "overflow_equivalent_thickness_sum_m": float(np.sum(removed[rebar_bnd])),
        "generated_volume": float(np.sum(generated[rebar_bnd])),
        "retained_volume": float(np.sum(rust[rebar_bnd])),
        "lost_volume": float(np.sum(lost[rebar_bnd])),
    }


def _solve_kinetic_contact(
    sim,
    retained_before_leakage,
    dt,
    leakage_pressure,
    leakage_permeability,
    pressure_scale,
    leakage_exponent,
    leakage_damage_factor,
    leakage_opening_factor,
    leakage_opening_reference,
    initial_pressure,
):
    interface = sim.rebar_bnd
    initial = np.maximum(initial_pressure[interface], 0.0)

    def evaluate(vector, update_history=False):
        sim.rust_pressure[:] = 0.0
        sim.rust_pressure[interface] = np.maximum(vector, 0.0)
        sim.solve_mechanics_pseudo_transient(
            sim.u_r, sim.rust_pressure, update_history=update_history
        )
        opening = np.maximum(sim.u * sim.nx + sim.v * sim.ny, 0.0)
        result = _kinetic_leakage_equilibrium(
            retained_before_leakage,
            sim.d,
            sim.S,
            dt,
            interface,
            sim.delta_0_gap,
            sim.rust_bulk_modulus,
            leakage_pressure,
            opening,
            leakage_permeability,
            pressure_scale,
            leakage_exponent,
            leakage_damage_factor,
            leakage_opening_factor,
            leakage_opening_reference,
        )
        return result, opening

    def residual(vector):
        result, _ = evaluate(vector, update_history=False)
        target_pressure = result[3][interface]
        return np.maximum(vector, 0.0) - target_pressure

    try:
        solution = optimize.anderson(
            residual,
            initial,
            M=8,
            alpha=0.1,
            w0=0.01,
            maxiter=80,
            f_tol=100.0,
        )
    except optimize.NoConvergence as error:
        solution = np.asarray(error.args[0], dtype=float)
    solution = np.maximum(solution, 0.0)
    result, opening = evaluate(solution, update_history=False)
    target = result[3][interface]
    relative_residual = float(
        np.max(np.abs(solution - target))
        / max(float(np.max(np.abs(target))), float(np.max(np.abs(solution))), 1.0)
    )
    sim.rust_pressure[:] = 0.0
    sim.rust_pressure[interface] = solution
    return result, opening, relative_residual


def _advance_unilateral_wedge_state(
    sim,
    generated,
    dt_year,
    previous_pressure,
    normal_opening,
    leakage_pressure,
    leakage_permeability,
    leakage_exponent,
    leakage_damage_factor,
    leakage_opening_factor,
    leakage_opening_reference,
    compacted_fraction,
    wedge_conversion_fraction,
    compacted_dissolution_rate,
):
    """Advance a conservative mobile/compacted rust inventory.

    Compacted rust is an immobile load-bearing solid. Newly generated product
    first enters the compacted/mobile inventory and only mobile product can
    leave through the finite-rate leakage law. This prevents damage itself
    from instantaneously deleting newly generated mass.
    """
    interface = sim.rebar_bnd
    damage = _project_damage_to_interface(sim.d, interface, sim.is_rebar)
    generated = np.maximum(np.asarray(generated, dtype=float), 0.0)
    retained_new = generated

    compact_fraction = np.clip(float(compacted_fraction), 0.0, 1.0)
    sim.rust_compacted += compact_fraction * retained_new
    sim.rust_mobile += (1.0 - compact_fraction) * retained_new

    opening = np.maximum(np.asarray(normal_opening, dtype=float), 0.0)
    overpressure = np.maximum(
        (np.asarray(previous_pressure, dtype=float) - float(leakage_pressure))
        / max(float(leakage_pressure), 1.0),
        0.0,
    )
    multiplier = (1.0 + max(float(leakage_damage_factor), 0.0) * damage) * (
        1.0
        + max(float(leakage_opening_factor), 0.0)
        * opening
        / max(float(leakage_opening_reference), 1.0e-12)
    )
    mobile_leak = (
        max(float(dt_year), 0.0)
        * max(float(leakage_permeability), 0.0)
        * multiplier
        * overpressure ** max(float(leakage_exponent), 0.0)
    )
    mobile_leak = np.minimum(sim.rust_mobile, mobile_leak)
    mobile_leak[~interface] = 0.0
    sim.rust_mobile -= mobile_leak
    sim.rust_lost += mobile_leak

    dissolution_fraction = 1.0 - np.exp(
        -max(float(compacted_dissolution_rate), 0.0) * max(float(dt_year), 0.0)
    )
    dissolved = dissolution_fraction * sim.rust_compacted
    dissolved[~interface] = 0.0
    sim.rust_compacted -= dissolved
    sim.rust_lost += dissolved

    sim.rust_volume[:] = sim.rust_mobile + sim.rust_compacted
    candidate_wedge = np.clip(float(wedge_conversion_fraction), 0.0, 1.0) * np.maximum(
        sim.rust_compacted - sim.delta_0_gap, 0.0
    )
    if compacted_dissolution_rate > 0.0:
        sim.residual_wedge[:] = np.minimum(
            sim.residual_wedge, np.maximum(sim.rust_compacted - sim.delta_0_gap, 0.0)
        )
    sim.residual_wedge[:] = np.maximum(sim.residual_wedge, candidate_wedge)
    sim.rust_filled[:] = np.minimum(sim.rust_volume, sim.delta_0_gap)
    for field in (
        sim.rust_mobile,
        sim.rust_compacted,
        sim.rust_volume,
        sim.rust_lost,
        sim.residual_wedge,
        sim.rust_filled,
    ):
        field[~interface] = 0.0
    return {
        "cap_triggered_cells": int(np.count_nonzero((overpressure > 0.0) & interface)),
        "overflow_removed_m": mobile_leak + dissolved,
        "overflow_equivalent_thickness_sum_m": float(
            np.sum((mobile_leak + dissolved)[interface])
        ),
        "generated_volume": float(np.sum(generated[interface])),
        "retained_volume": float(np.sum(sim.rust_volume[interface])),
        "lost_volume": float(np.sum(sim.rust_lost[interface])),
    }


def _unilateral_wedge_pressure(sim, normal_opening, contact_stiffness_pa_per_m):
    """Compression-only contact pressure; an open interface carries no tension."""
    opening = np.maximum(np.asarray(normal_opening, dtype=float), 0.0)
    pressure = max(float(contact_stiffness_pa_per_m), 0.0) * np.maximum(
        sim.residual_wedge - opening, 0.0
    )
    pressure[~sim.rebar_bnd] = 0.0
    return pressure


def _solve_unilateral_wedge_contact(sim, contact_stiffness_pa_per_m, initial_pressure):
    """Solve the unilateral rust-wedge contact complementarity problem.

    With separation ``y = g_open - g_wedge + p/k_n``, contact requires
    ``p >= 0``, ``y >= 0`` and ``p*y = 0``. A matrix-free primal-dual
    active-set solve is used first. At fixed damage, mechanical opening is
    linear in contact pressure, so active-set systems are solved through
    compliance actions without assembling a dense interface matrix. On small
    interfaces, a stalled solve falls back to an explicitly sampled matrix.
    """
    interface = sim.rebar_bnd
    stiffness = max(float(contact_stiffness_pa_per_m), 1.0)
    pressure_scale = max(
        float(np.max(np.asarray(initial_pressure, dtype=float)[interface])),
        stiffness * float(np.max(sim.residual_wedge[interface])),
        1.0e6,
    )
    initial = (
        np.maximum(np.asarray(initial_pressure, dtype=float)[interface], 0.0)
        / pressure_scale
    )

    # Primary solve: impose the compliant contact law directly in the global
    # mechanical system. This avoids one full mechanical solve per interface
    # basis vector and therefore remains usable on fine grids.
    if hasattr(sim, "_mechanics_stiffness"):
        active = (initial > 1.0e-12) | (sim.residual_wedge[interface] > 0.0)
        seen_active_sets = set()
        monolithic_iterations = 0
        monolithic_linear_iterations = 0
        monolithic_smoothing = 1.0e-12
        for _ in range(50):
            monolithic_iterations += 1
            signature = np.packbits(active).tobytes()
            if signature in seen_active_sets:
                break
            seen_active_sets.add(signature)

            active_field = np.zeros_like(interface, dtype=bool)
            active_field[interface] = active
            sim.solve_mechanics_pseudo_transient(
                sim.u_r,
                sim.rust_pressure,
                update_history=False,
                unilateral_contact_active=active_field,
                unilateral_contact_wedge=sim.residual_wedge,
                unilateral_contact_stiffness=stiffness,
            )
            monolithic_linear_iterations += sim.last_mechanics_iterations
            raw_opening = sim.u * sim.nx + sim.v * sim.ny
            raw_pressure = np.zeros_like(initial)
            raw_pressure[active] = stiffness * (
                sim.residual_wedge[interface][active]
                - raw_opening[interface][active]
            )
            separation = raw_opening[interface] - sim.residual_wedge[interface]
            separation[active] += raw_pressure[active] / stiffness
            pressure_hat = np.maximum(raw_pressure, 0.0) / pressure_scale
            separation_hat = stiffness * separation / pressure_scale
            complementarity = (
                np.sqrt(
                    pressure_hat**2
                    + separation_hat**2
                    + monolithic_smoothing**2
                )
                - pressure_hat
                - separation_hat
            )
            residual = float(np.max(np.abs(complementarity)))
            switching_function = raw_pressure / pressure_scale - separation_hat
            new_active = active.copy()
            new_active[switching_function > 1.0e-7] = True
            new_active[switching_function < -1.0e-7] = False
            if np.array_equal(new_active, active):
                pressure_tolerance = 1.0e-7 * pressure_scale
                separation_tolerance = 1.0e-7 * pressure_scale / stiffness
                if (
                    residual <= 1.0e-4
                    and (
                        not np.any(active)
                        or float(np.min(raw_pressure[active]))
                        >= -pressure_tolerance
                    )
                    and (
                        not np.any(~active)
                        or float(np.min(separation[~active]))
                        >= -separation_tolerance
                    )
                ):
                    pressure_values = np.maximum(raw_pressure, 0.0)
                    sim.last_contact_min_separation_m = float(
                        np.min(separation)
                    )
                    sim.last_contact_complementarity_residual = residual
                    sim.last_contact_linear_info = 0
                    sim.last_contact_linear_iterations = (
                        monolithic_linear_iterations
                    )
                    sim.last_contact_active_iterations = monolithic_iterations
                    sim.last_contact_active_cells = int(
                        np.count_nonzero(
                            pressure_values > 1.0e-12 * pressure_scale
                        )
                    )
                    sim.last_contact_pdas_fallback_used = False
                    sim.rust_pressure[:] = 0.0
                    sim.rust_pressure[interface] = pressure_values
                    return np.maximum(raw_opening, 0.0), residual
                break
            active = new_active

    def evaluate_affine(dimensionless_pressure, update_history=False):
        pressure_values = pressure_scale * np.asarray(
            dimensionless_pressure, dtype=float
        )
        sim.rust_pressure[:] = 0.0
        sim.rust_pressure[interface] = pressure_values
        sim.solve_mechanics_pseudo_transient(
            sim.u_r, sim.rust_pressure, update_history=update_history
        )
        raw_opening = sim.u * sim.nx + sim.v * sim.ny
        separation = (
            raw_opening[interface]
            - sim.residual_wedge[interface]
            + pressure_values / stiffness
        )
        return pressure_values, separation, np.maximum(raw_opening, 0.0)

    def evaluate(dimensionless_pressure, update_history=False):
        return evaluate_affine(
            np.maximum(np.asarray(dimensionless_pressure, dtype=float), 0.0),
            update_history=update_history,
        )

    smoothing = 1.0e-12

    zero = np.zeros_like(initial)
    _, zero_separation, _ = evaluate_affine(zero, update_history=False)
    affine_offset = stiffness * zero_separation / pressure_scale

    def compliance_action(vector):
        vector = np.asarray(vector, dtype=float)
        if not np.any(vector):
            return np.zeros_like(vector)
        _, separation, _ = evaluate_affine(vector, update_history=False)
        return stiffness * separation / pressure_scale - affine_offset

    def fb_residual(pressure_hat, separation_hat):
        scale = np.sqrt(
            pressure_hat**2 + separation_hat**2 + smoothing**2
        )
        return float(
            np.max(np.abs(scale - pressure_hat - separation_hat))
        )

    def complementarity_state(vector):
        pressure_hat = np.maximum(np.asarray(vector, dtype=float), 0.0)
        separation_hat = compliance_action(pressure_hat) + affine_offset
        return fb_residual(pressure_hat, separation_hat), separation_hat

    solution = np.maximum(initial.copy(), 0.0)
    best_solution = solution.copy()
    best_residual = np.inf
    active_iteration_count = 0
    linear_info = 0
    linear_iteration_count = 0
    area_weight = np.asarray(
        getattr(sim, "rebar_area_weight", np.ones_like(sim.rust_pressure)),
        dtype=float,
    )[interface]
    positive_area = area_weight[area_weight > 0.0]
    reference_area = float(np.max(positive_area)) if positive_area.size else 1.0
    sqrt_area = np.sqrt(np.maximum(area_weight / reference_area, 1.0e-12))
    seen_active_sets = set()
    initial_residual, initial_separation = complementarity_state(solution)
    best_residual = initial_residual
    best_solution = solution.copy()
    active = solution - initial_separation > 0.0
    if not np.any(active):
        active = affine_offset < 0.0

    for _ in range(30):
        active_iteration_count += 1
        if best_residual <= 1.0e-7:
            break
        signature = np.packbits(active).tobytes()
        if signature in seen_active_sets:
            break
        seen_active_sets.add(signature)

        candidate = np.zeros_like(solution)
        active_indices = np.flatnonzero(active)
        if active_indices.size:

            def active_compliance_action(weighted_pressure):
                pressure = np.zeros_like(solution)
                pressure[active] = weighted_pressure / sqrt_area[active]
                separation = compliance_action(pressure)
                return sqrt_area[active] * separation[active]

            active_operator = spla.LinearOperator(
                (active_indices.size, active_indices.size),
                matvec=active_compliance_action,
                dtype=float,
            )
            active_rhs = -sqrt_area[active] * affine_offset[active]
            active_initial = sqrt_area[active] * solution[active]
            max_linear_iterations = min(
                max(30, int(4.0 * np.sqrt(active_indices.size))), 200
            )
            def count_linear_iteration(_):
                nonlocal linear_iteration_count
                linear_iteration_count += 1

            weighted_solution, linear_info = spla.cg(
                active_operator,
                active_rhs,
                x0=active_initial,
                rtol=1.0e-8,
                atol=0.0,
                maxiter=max_linear_iterations,
                callback=count_linear_iteration,
            )
            candidate[active] = weighted_solution / sqrt_area[active]

        candidate_separation = compliance_action(candidate) + affine_offset
        new_active = candidate - candidate_separation > 0.0
        feasible_candidate = np.maximum(candidate, 0.0)
        feasible_separation = (
            compliance_action(feasible_candidate) + affine_offset
        )
        candidate_residual = fb_residual(
            feasible_candidate, feasible_separation
        )
        if candidate_residual < best_residual:
            best_residual = candidate_residual
            best_solution = feasible_candidate.copy()
        solution = feasible_candidate
        if candidate_residual <= 1.0e-7:
            break
        if np.array_equal(new_active, active):
            break
        active = new_active

    final_pdas_residual, _ = complementarity_state(solution)
    if final_pdas_residual < best_residual:
        best_residual = final_pdas_residual
        best_solution = solution.copy()
    solution = best_solution
    pdas_fallback_used = False
    projected_residual, _ = complementarity_state(solution)
    if projected_residual > 1.0e-4 and solution.size <= 1024:
        pdas_fallback_used = True

        def solve_explicit_pdas(compliance, starting_solution, starting_residual):
            pdas_solution = np.maximum(starting_solution, 0.0)
            pdas_best = pdas_solution.copy()
            pdas_best_residual = starting_residual
            pdas_iterations = 0
            active = (
                pdas_solution
                - (compliance @ pdas_solution + affine_offset)
                > 0.0
            )
            for _ in range(50):
                pdas_iterations += 1
                candidate = np.zeros_like(pdas_solution)
                if np.any(active):
                    active_matrix = compliance[np.ix_(active, active)]
                    active_rhs = -affine_offset[active]
                    try:
                        candidate[active] = np.linalg.solve(
                            active_matrix, active_rhs
                        )
                    except np.linalg.LinAlgError:
                        candidate[active] = np.linalg.lstsq(
                            active_matrix, active_rhs, rcond=1.0e-10
                        )[0]
                feasible_candidate = np.maximum(candidate, 0.0)
                separation_candidate = (
                    compliance @ feasible_candidate + affine_offset
                )
                candidate_residual = fb_residual(
                    feasible_candidate, separation_candidate
                )
                if candidate_residual < pdas_best_residual:
                    pdas_best_residual = candidate_residual
                    pdas_best = feasible_candidate.copy()
                new_active = (
                    candidate - (compliance @ candidate + affine_offset) > 0.0
                )
                pdas_solution = candidate
                if np.array_equal(new_active, active):
                    break
                active = new_active
            return pdas_best, pdas_best_residual, pdas_iterations

        cached_compliance = getattr(sim, "_contact_compliance_cache", None)
        cached_damage = getattr(sim, "_contact_compliance_damage", None)
        cached_stiffness = getattr(sim, "_contact_compliance_stiffness", None)
        cache_valid = bool(
            cached_compliance is not None
            and cached_compliance.shape == (solution.size, solution.size)
            and cached_damage is not None
            and np.shape(cached_damage) == np.shape(sim.d)
            and cached_stiffness == stiffness
            and float(np.max(np.abs(sim.d - cached_damage))) <= 1.0e-2
        )

        for use_cache in ((True, False) if cache_valid else (False,)):
            if use_cache:
                compliance = cached_compliance
            else:
                sim._mechanics_direct_preconditioner = None
                sim._mechanics_direct_preconditioner_damage = None
                sim._mechanics_direct_factor = None
                compliance = np.empty(
                    (solution.size, solution.size), dtype=float
                )
                for column in range(solution.size):
                    basis = np.zeros_like(solution)
                    basis[column] = 1.0
                    compliance[:, column] = compliance_action(basis)
                sim._contact_compliance_cache = compliance.copy()
                sim._contact_compliance_damage = sim.d.copy()
                sim._contact_compliance_stiffness = stiffness

            pdas_best, _, pdas_iterations = solve_explicit_pdas(
                compliance, solution, projected_residual
            )
            active_iteration_count += pdas_iterations
            actual_residual, _ = complementarity_state(pdas_best)
            if actual_residual < projected_residual:
                solution = pdas_best
                projected_residual = actual_residual
            if actual_residual <= 1.0e-4 or not use_cache:
                break

    pressure_values, separation, opening = evaluate(solution, update_history=False)
    pressure_hat = pressure_values / pressure_scale
    separation_hat = stiffness * separation / pressure_scale
    complementarity = (
        np.sqrt(pressure_hat**2 + separation_hat**2 + smoothing**2)
        - pressure_hat
        - separation_hat
    )
    relative_residual = float(np.max(np.abs(complementarity)))
    sim.last_contact_min_separation_m = float(np.min(separation))
    sim.last_contact_complementarity_residual = relative_residual
    sim.last_contact_linear_info = int(linear_info)
    sim.last_contact_linear_iterations = linear_iteration_count
    sim.last_contact_active_iterations = active_iteration_count
    sim.last_contact_active_cells = int(np.count_nonzero(pressure_values > 1.0e-12 * pressure_scale))
    sim.last_contact_pdas_fallback_used = pdas_fallback_used
    sim.rust_pressure[:] = 0.0
    sim.rust_pressure[interface] = pressure_values
    return opening, relative_residual


def _project_damage_to_interface(damage, rebar_bnd, is_rebar=None):
    source = np.clip(np.asarray(damage, dtype=float), 0.0, 1.0).copy()
    if is_rebar is not None:
        source[np.asarray(is_rebar, dtype=bool)] = 0.0
    adjacent_damage = ndimage.maximum_filter(source, size=3, mode="nearest")
    projected = source.copy()
    projected[rebar_bnd] = adjacent_damage[rebar_bnd]
    return projected


def _enforce_zero_dirichlet_spd(matrix, rhs, locked):
    locked = np.asarray(locked, dtype=bool).ravel()
    free_diagonal = sp.diags((~locked).astype(float), format="csr")
    constrained = free_diagonal @ matrix @ free_diagonal + sp.diags(
        locked.astype(float), format="csr"
    )
    constrained_rhs = np.asarray(rhs, dtype=float).copy()
    constrained_rhs[locked] = 0.0
    return (constrained.tocsr(), constrained_rhs)


def _contact_pressure_from_retained_rust(
    sim,
    mode="local",
    normal_opening=None,
    update_filled=True,
    pressure_limit=None,
    pressure_limit_scale=1.0,
    pressure_limit_damage=0.5,
    leakage_law="instantaneous_cap",
):
    damage = _project_damage_to_interface(sim.d, sim.rebar_bnd, sim.is_rebar)
    sim.interface_crack_damage = damage
    capacity = sim.delta_0_gap + sim.crack_storage_coeff * damage**1.5
    if update_filled:
        sim.rust_filled[:] = np.minimum(sim.rust_volume, capacity)
    excess = np.maximum(sim.rust_volume - capacity, 0.0)
    saturation = np.clip((sim.S - 0.35) / 0.55, 0.0, 1.0)
    stiffness = sim.rust_bulk_modulus * (1.0 - damage) ** 2 * saturation
    pressure = stiffness * excess / np.maximum(capacity, 1e-12)
    if pressure_limit is not None and leakage_law == "instantaneous_cap":
        limit = (
            pressure_limit_scale
            * pressure_limit
            * (1.0 + pressure_limit_damage * (1.0 - damage))
        )
        allowed_excess = limit * capacity / np.maximum(stiffness, 1.0e-18)
        overflow = np.maximum(excess - allowed_excess, 0.0)
        overflow[~sim.rebar_bnd] = 0.0
        if update_filled:
            sim.rust_volume -= overflow
            sim.rust_lost += overflow
            sim.rust_filled[:] = np.minimum(sim.rust_volume, capacity)
            excess = np.maximum(sim.rust_volume - capacity, 0.0)
            pressure = stiffness * excess / np.maximum(capacity, 1e-12)
        else:
            pressure = np.minimum(pressure, limit)
    pressure[~sim.rebar_bnd] = 0.0
    if mode == "redistributed":
        weights = sim.rebar_area_weight[sim.rebar_bnd]
        weight_sum = max(float(np.sum(weights)), 1e-30)
        retained_volume = float(np.sum(weights * sim.rust_volume[sim.rebar_bnd]))
        capacity_volume = max(float(np.sum(weights * capacity[sim.rebar_bnd])), 1e-30)
        pooled_excess = max(retained_volume - capacity_volume, 0.0)
        effective_bulk = float(
            np.sum(weights * sim.rust_bulk_modulus * saturation[sim.rebar_bnd])
            / weight_sum
        )
        pooled_pressure = effective_bulk * pooled_excess / capacity_volume
        pressure[:] = 0.0
        pressure[sim.rebar_bnd] = pooled_pressure
    elif mode != "local":
        raise ValueError(f"unknown contact pressure mode: {mode}")
    return pressure


def _apply_engineering_parameters_3d(sim, params: EngineeringParameters):
    sim.cover_depth = params.cover_depth
    sim.R_bar = params.rebar_radius
    sim.cy = sim.H - sim.cover_depth - sim.R_bar
    sim.C_max_surf = params.surface_chloride
    sim.C_critical = params.critical_chloride
    sim.Dc_mat_base = params.mortar_chloride_diffusivity
    sim.i_corr_ref = params.reference_corrosion_current
    sim.delta_0_gap = params.porous_gap
    sim.rust_expansion_ratio = params.rust_expansion_ratio
    sim.crack_storage_coeff = params.crack_storage_coeff
    sim.rust_bulk_modulus = params.rust_bulk_modulus
    sim.Gc_mat_base = params.fracture_energy
    sim.l0 = params.phasefield_length
    sim.k_res = params.residual_stiffness
    sim.agg_vol_frac = params.aggregate_fraction
    sim.itz_width = params.itz_width
    sim.aggregate_records = _load_aggregate_records(params.aggregate_records_path)
    sim.surface_gauge_length = params.surface_gauge_length
    sim.steel_interface_width = params.steel_interface_width
    sim.surface_observation_depth = params.surface_observation_depth
    sim.mechanics_rtol = params.mechanics_rtol
    sim.mechanics_max_iterations = params.mechanics_max_iterations
    sim.axial_symmetry_ends = params.axial_symmetry_ends
    sim._generate_meso_structure_3d()
    sim.Dc_map = np.where(
        sim.is_agg,
        1e-12,
        sim.Dc_mat_base
        * (1.0 + (params.itz_diffusivity_multiplier - 1.0) * sim.itz_fraction),
    )
    sim.Dw_map = np.where(
        sim.is_agg, 1e-12, sim.Dw_mat + (sim.Dw_itz - sim.Dw_mat) * sim.itz_fraction
    )
    base_gc_map = np.where(
        sim.is_agg,
        sim.Gc_mat_base * 10.0,
        sim.Gc_mat_base * (1.0 - 0.6 * sim.itz_fraction),
    )
    heterogeneity = _fracture_heterogeneity_3d(sim, params)
    sim.fracture_heterogeneity_factor = heterogeneity
    sim.Gc_map = np.where(sim.is_agg, base_gc_map, base_gc_map * heterogeneity)
    interface_material = sim.rebar_bnd & ~sim.is_agg
    interface_gc = (
        sim.Gc_mat_base * params.steel_interface_fracture_factor * heterogeneity
    )
    sim.Gc_map[interface_material] = np.minimum(
        sim.Gc_map[interface_material], interface_gc[interface_material]
    )
    if params.interface_flaw_factor < 1.0:
        radius = np.sqrt((sim.X - sim.cx) ** 2 + (sim.Y - sim.cy) ** 2)
        angle_from_cover = np.arctan2(sim.X - sim.cx, sim.Y - sim.cy)
        flaw = (
            (radius > sim.R_bar)
            & (radius <= sim.R_bar + params.interface_flaw_radial_extent)
            & (np.abs(angle_from_cover) <= params.interface_flaw_half_angle)
            & ~sim.is_agg
            & ~sim.is_rebar
            & ~sim.rebar_bnd
        )
        sim.Gc_map[flaw] *= max(float(params.interface_flaw_factor), 0.05)
        sim.interface_flaw_mask = flaw
    else:
        sim.interface_flaw_mask = np.zeros(sim.shape_3d, dtype=bool)
    sim.E_map = np.where(
        sim.is_agg, 60000000000.0, sim.E_mat_base * (1.0 - 0.4 * sim.itz_fraction)
    )
    sim.E_map[interface_material] = np.minimum(
        sim.E_map[interface_material],
        sim.E_mat_base * params.steel_interface_stiffness_factor,
    )
    sim.K_bulk = sim.E_map / (3 * (1 - 2 * sim.nu))
    sim.mu_map = sim.E_map / (2 * (1 + sim.nu))
    sim.lam_map = sim.K_bulk - 2.0 / 3.0 * sim.mu_map
    sim.S[:] = sim.S_init
    sim.C[:] = 0.0
    sim.d[:] = 0.0
    sim.H_hist[:] = 0.0
    sim.Na[:] = 60.0
    sim.OH[:] = 80.0
    sim.electric_grad[:] = 0.0
    sim.grad_phi_x[:] = 0.0
    sim.grad_phi_y[:] = 0.0
    sim.grad_phi_z[:] = 0.0
    sim.rust_generated[:] = 0.0
    sim.rust_volume[:] = 0.0
    sim.rust_filled[:] = 0.0
    sim.rust_lost[:] = 0.0
    sim.rust_pressure[:] = 0.0
    sim.x_corr[:] = 0.0
    sim.i_corr[:] = 0.0
    sim.u_r[:] = 0.0
    sim.u[:] = 0.0
    sim.v[:] = 0.0
    sim.w[:] = 0.0
    sim.t_ini_map[:] = np.inf
    sim.elastic_energy_j = 0.0
    sim.fracture_energy_j = 0.0
    sim.pressure_work_j = 0.0
    sim.surface_crack_width_mm = 0.0
    sim.surface_crack_width_displacement_mm = 0.0
    sim.surface_crack_width_envelope_mm = 0.0
    sim.surface_gauge_locked = False
    sim.surface_gauge_center_x = np.nan
    sim.surface_gauge_center_z = np.nan
    sim.surface_gauge_normal_x = np.nan
    sim.surface_gauge_normal_z = np.nan


def prepare_3d_solver(params: EngineeringParameters | None = None, dt: float = 0.5):
    params = params or EngineeringParameters()
    sim = CorrosionCracking3D(
        seed=params.seed,
        grid_spacing=params.grid_spacing,
        domain_thickness=params.domain_thickness,
        domain_width=params.domain_width,
        domain_height=params.domain_height,
        itz_width=params.itz_width,
        aggregate_records=_load_aggregate_records(params.aggregate_records_path),
        surface_gauge_length=params.surface_gauge_length,
        steel_interface_width=params.steel_interface_width,
        surface_observation_depth=params.surface_observation_depth,
        rebar_radius=params.rebar_radius,
        cover_depth=params.cover_depth,
        axial_symmetry_ends=params.axial_symmetry_ends,
    )
    _apply_engineering_parameters_3d(sim, params)
    locked = sim.is_agg | sim.is_rebar
    top = np.zeros(sim.shape_3d, dtype=bool)
    top[:, -1, :] = True
    top &= ~locked
    n = sim.N
    sim._solve_s, sim._penalty_s = _prepare_transport_solver(
        sim.build_harmonic_laplacian_3d(
            sim.Dw_map, axial_transport=params.axial_transport_enabled
        ),
        n,
        dt,
        top.ravel(),
        locked.ravel(),
        params,
    )
    sim._solve_c, sim._penalty_c = _prepare_transport_solver(
        sim.build_harmonic_laplacian_3d(
            sim.Dc_map / 1.6, axial_transport=params.axial_transport_enabled
        ),
        n,
        dt,
        top.ravel(),
        locked.ravel(),
        params,
    )
    sim._solve_na, sim._penalty_na = _prepare_transport_solver(
        sim.build_harmonic_laplacian_3d(
            0.65 * sim.Dc_map, axial_transport=params.axial_transport_enabled
        ),
        n,
        dt,
        top.ravel(),
        locked.ravel(),
        params,
    )
    sim._solve_oh, sim._penalty_oh = _prepare_transport_solver(
        sim.build_harmonic_laplacian_3d(
            2.6 * sim.Dc_map, axial_transport=params.axial_transport_enabled
        ),
        n,
        dt,
        top.ravel(),
        locked.ravel(),
        params,
    )
    sim._pressure_solve = None
    sim._phase_lap = sim.build_harmonic_laplacian_3d(sim.Gc_map * sim.l0)
    return (sim, top, locked)


def solve_phase_field_cg_3d(sim, maxiter=450):
    diag = sim.Gc_map / sim.l0 + 2.0 * sim.H_hist
    a = sp.diags(diag.ravel(), 0, format="csr") - sim._phase_lap
    rhs = (2.0 * sim.H_hist).ravel()
    locked_mask = sim.is_rebar | getattr(sim, "mechanics_anchor_mask", False)
    if sim.axial_symmetry_ends:
        locked_mask[:, :, 0] = True
        locked_mask[:, :, -1] = True
    locked = locked_mask.ravel()
    a, rhs = _enforce_zero_dirichlet_spd(a, rhs, locked)
    iteration_count = [0]

    def count_iteration(_):
        iteration_count[0] += 1

    d, info = spla.cg(
        a,
        rhs,
        x0=sim.d.ravel(),
        rtol=1e-06,
        atol=0.0,
        maxiter=maxiter,
        callback=count_iteration,
    )
    if info != 0:
        d = spla.spsolve(a, rhs)
    residual = a @ d - rhs
    free = ~locked
    relative_residual = float(
        np.linalg.norm(residual[free]) / max(np.linalg.norm(rhs[free]), 1.0)
    )
    if not np.all(np.isfinite(d)) or relative_residual > 5e-06:
        raise RuntimeError(
            f"phase-field equilibrium failed: cg_info={info}, relative_residual={relative_residual:.3e}"
        )
    sim.last_phasefield_residual = relative_residual
    sim.last_phasefield_iterations = iteration_count[0]
    sim.last_phasefield_converged = True
    sim.d = np.maximum(sim.d, np.clip(d.reshape(sim.shape_3d), 0.0, 1.0))
    sim.d[locked_mask] = 0.0
    sim.H_hist[locked_mask] = 0.0


def run_full_3d(
    total_years=50.0,
    dt=0.5,
    params: EngineeringParameters | None = None,
    staggered_max_iterations: int = 60,
    progress=True,
    step_callback=None,
    termination_condition=None,
):
    params = params or EngineeringParameters()
    sim, top, locked = prepare_3d_solver(params, dt=dt)
    times, max_c, max_p, max_d, max_i = ([], [], [], [], [])
    elastic_e, fracture_e, pressure_work, energy_error = ([], [], [], [])
    rust_gen, rust_ret, rust_loss, rust_err, crack_w, crack_w_disp, crack_w_envelope = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    rust_gen_volume, rust_ret_volume, rust_loss_volume, rust_err_volume = (
        [],
        [],
        [],
        [],
    )
    cap_trigger_counts, overflow_volumes = [], []
    retained_loss_ratios = []
    tensile_strain_maxima, driving_energy_maxima = [], []
    trial_pressure_maxima = []
    component_cells_0p05, component_cells_0p20 = [], []
    component_touches_surface_0p05, component_touches_surface_0p20 = [], []
    surface_damage_maxima, damage_area_fractions = [], []
    damage_event_times = {0.05: None, 0.10: None, 0.20: None, 0.35: None}
    opening_event_times = {0.005: None, 0.01: None, 0.05: None}
    cap_trigger_total = 0
    overflow_volume_total = 0.0
    (
        mechanics_residuals,
        mechanics_calls,
        staggered_counts,
        coupling_pressure_residuals,
        transport_residuals,
        phasefield_residuals,
    ) = ([], [], [], [], [], [])
    peak_pressure_field = np.zeros_like(sim.rust_pressure)
    peak_icorr_field = np.zeros_like(sim.i_corr)
    peak_electric_field = np.zeros_like(sim.electric_grad)
    p_cr = hoop_cracking_pressure(sim.R_bar, sim.R_bar + sim.cover_depth, 3000000.0)
    start = time.time()
    t = 0.0
    while t < total_years - 1e-12:
        actual_dt = min(dt, total_years - t)
        sim.S = np.clip(
            _solve_dirichlet(
                sim._solve_s,
                sim._penalty_s,
                sim.S,
                sim.shape_3d,
                top.ravel(),
                1.0,
                locked.ravel(),
            ),
            0.0,
            1.0,
        )
        sim.C = np.clip(
            _solve_dirichlet(
                sim._solve_c,
                sim._penalty_c,
                sim.C,
                sim.shape_3d,
                top.ravel(),
                sim.C_max_surf,
                locked.ravel(),
            ),
            0.0,
            sim.C_max_surf,
        )
        sim.Na = np.clip(
            _solve_dirichlet(
                sim._solve_na,
                sim._penalty_na,
                sim.Na,
                sim.shape_3d,
                top.ravel(),
                560.0,
                locked.ravel(),
                60.0,
            ),
            1.0,
            700.0,
        )
        sim.OH = np.clip(
            _solve_dirichlet(
                sim._solve_oh,
                sim._penalty_oh,
                sim.OH,
                sim.shape_3d,
                top.ravel(),
                25.0,
                locked.ravel(),
                80.0,
            ),
            5.0,
            120.0,
        )
        sim.update_electric_field()
        if not params.axial_transport_enabled:
            sim.grad_phi_z[:] = 0.0
            sim.electric_grad = np.sqrt(sim.grad_phi_x**2 + sim.grad_phi_y**2)
        active = sim.rebar_bnd & (sim.C >= sim.C_critical)
        sim.t_ini_map[active & np.isinf(sim.t_ini_map)] = t + actual_dt
        sim.i_corr = _corrosion_current(
            sim.C,
            sim.OH,
            sim.S,
            sim.d,
            active,
            sim.i_corr_ref,
            params.oxygen_mode,
            params.oxygen_decay_alpha,
            params.oxygen_decay_center,
        )
        step_residual = 0.0
        step_operator_calls = 0
        step_staggered_count = 0
        step_coupling_residual = 0.0
        step_phasefield_residual = 0.0
        step_phasefield_iterations = 0
        step_trial_pressure_mpa = 0.0
        step_cap_trigger_count = 0
        step_overflow_volume = 0.0
        substep_dt = actual_dt / 3.0
        for _substep in range(3):
            steel_loss = (
                sim.i_corr
                * M_FE
                * SECONDS_PER_YEAR
                * substep_dt
                / (N_FE * FARADAY * RHO_FE)
            )
            sim.rust_generated += sim.rust_expansion_ratio * steel_loss
            interface_damage = _project_damage_to_interface(
                sim.d, sim.rebar_bnd, sim.is_rebar
            )
            normal_opening = np.maximum(sim.u * sim.nx + sim.v * sim.ny, 0.0)
            rust_update = _update_rust_state(
                sim.rust_volume,
                sim.rust_lost,
                sim.rust_filled,
                sim.rust_pressure,
                sim.i_corr,
                interface_damage,
                sim.S,
                substep_dt,
                sim.rebar_bnd,
                sim.delta_0_gap,
                sim.rust_expansion_ratio,
                sim.rust_bulk_modulus,
                p_cr,
                crack_storage_coeff=sim.crack_storage_coeff,
                mechanical_opening=normal_opening,
                leakage_pressure_scale=params.pressure_limit_scale,
                leakage_damage_factor=params.pressure_limit_damage,
                leakage_law=params.leakage_law,
                leakage_rate=params.leakage_rate,
                leakage_exponent=params.leakage_exponent,
            )
            step_cap_trigger_count += rust_update["cap_triggered_cells"]
            step_overflow_volume += float(
                np.sum(sim.rebar_area_weight * rust_update["overflow_removed_m"])
            )
            step_trial_pressure_mpa = max(
                step_trial_pressure_mpa, float(np.max(sim.rust_pressure) / 1.0e6)
            )
            sim.rust_pressure[:] = _contact_pressure_from_retained_rust(
                sim,
                mode="local",
                normal_opening=normal_opening,
                update_filled=False,
                pressure_limit=p_cr,
                pressure_limit_scale=params.pressure_limit_scale,
                pressure_limit_damage=params.pressure_limit_damage,
                leakage_law=params.leakage_law,
            )
            sim.u_r = np.maximum(sim.rust_volume - sim.rust_filled, 0.0)
            sim.u_r[~sim.rebar_bnd] = 0.0
            if np.max(sim.rust_pressure) > 0.01 * p_cr:
                converged = False
                for staggered_index in range(max(int(staggered_max_iterations), 1)):
                    damage_before = sim.d.copy()
                    pressure_before = sim.rust_pressure.copy()
                    sim.solve_mechanics_pseudo_transient(sim.u_r, sim.rust_pressure)
                    solve_phase_field_cg_3d(sim)
                    step_phasefield_residual = max(
                        step_phasefield_residual, sim.last_phasefield_residual
                    )
                    step_phasefield_iterations += sim.last_phasefield_iterations
                    damage_change = float(np.max(np.abs(sim.d - damage_before)))
                    target_pressure = _contact_pressure_from_retained_rust(
                        sim,
                        mode="local",
                        update_filled=False,
                        pressure_limit=p_cr,
                        pressure_limit_scale=params.pressure_limit_scale,
                        pressure_limit_damage=params.pressure_limit_damage,
                        leakage_law=params.leakage_law,
                    )
                    pressure_scale = max(
                        float(np.max(np.abs(target_pressure))),
                        float(np.max(np.abs(pressure_before))),
                        1.0,
                    )
                    pressure_compatibility = float(
                        np.max(np.abs(target_pressure - pressure_before))
                        / pressure_scale
                    )
                    sim.rust_pressure[:] = target_pressure
                    sim.rust_pressure[~sim.rebar_bnd] = 0.0
                    step_residual = max(step_residual, sim.last_mechanics_residual)
                    step_coupling_residual = pressure_compatibility
                    step_operator_calls += sim.last_mechanics_iterations
                    step_staggered_count += 1
                    if (
                        damage_change <= 0.0001
                        and pressure_compatibility <= 0.0001
                        and sim.last_mechanics_converged
                        and (
                            sim.last_mechanics_residual
                            <= max(5e-06, sim.mechanics_rtol)
                        )
                    ):
                        converged = True
                        break
                if not converged:
                    raise RuntimeError(
                        f"staggered mechanics/phase-field iteration failed at t={t:.6g} yr; last damage increment={damage_change:.3e}, pressure compatibility residual={pressure_compatibility:.3e}, mechanics residual={sim.last_mechanics_residual:.3e}"
                    )
            sim.rust_pressure[:] = _contact_pressure_from_retained_rust(
                sim,
                mode="local",
                update_filled=True,
                pressure_limit=p_cr,
                pressure_limit_scale=params.pressure_limit_scale,
                pressure_limit_damage=params.pressure_limit_damage,
                leakage_law=params.leakage_law,
            )
        sim.update_energy_diagnostics()
        tensile_strain, driving_energy = sim.diagnostic_tensile_strain_and_driving_energy()
        surface_damage = float(np.max(sim.d[np.asarray(sim.surface_rows, dtype=int)])) if hasattr(sim, "surface_rows") else float(np.max(sim.d[:, -1, :]))
        surface_area_fraction = float(np.mean(sim.d[:, -1, :] >= 0.05))
        step = {
            "max_chloride": float(np.nanmax(sim.C)),
            "max_pressure_mpa": float(np.nanmax(sim.rust_pressure) / 1000000.0),
            "max_damage": float(np.nanmax(sim.d)),
            "max_icorr": float(np.nanmax(sim.i_corr)),
        }
        peak_pressure_field = np.maximum(peak_pressure_field, sim.rust_pressure)
        peak_icorr_field = np.maximum(peak_icorr_field, sim.i_corr)
        peak_electric_field = np.maximum(peak_electric_field, sim.electric_grad)
        t += actual_dt
        times.append(t)
        max_c.append(step["max_chloride"])
        max_p.append(step["max_pressure_mpa"])
        max_d.append(step["max_damage"])
        max_i.append(step["max_icorr"])
        elastic_e.append(sim.elastic_energy_j)
        fracture_e.append(sim.fracture_energy_j)
        pressure_work.append(sim.pressure_work_j)
        energy_error.append(sim.mechanical_energy_relative_error)
        interface_area = max(float(np.sum(sim.rebar_area_weight[sim.rebar_bnd])), 1e-30)
        gen_volume = float(np.sum(sim.rebar_area_weight * sim.rust_generated))
        ret_volume = float(np.sum(sim.rebar_area_weight * sim.rust_volume))
        lost_volume = float(np.sum(sim.rebar_area_weight * sim.rust_lost))
        gen = gen_volume / interface_area
        ret = ret_volume / interface_area
        lost = lost_volume / interface_area
        rust_gen.append(gen)
        rust_ret.append(ret)
        rust_loss.append(lost)
        rust_err.append(gen - ret - lost)
        rust_gen_volume.append(gen_volume)
        rust_ret_volume.append(ret_volume)
        rust_loss_volume.append(lost_volume)
        rust_err_volume.append(gen_volume - ret_volume - lost_volume)
        cap_trigger_counts.append(step_cap_trigger_count)
        overflow_volumes.append(step_overflow_volume)
        retained_loss_ratios.append(lost / max(ret + lost, 1e-30))
        tensile_strain_maxima.append(tensile_strain)
        driving_energy_maxima.append(driving_energy)
        trial_pressure_maxima.append(step_trial_pressure_mpa)
        comp05 = sim.diagnostic_damage_component(0.05)
        comp20 = sim.diagnostic_damage_component(0.20)
        component_cells_0p05.append(comp05[0])
        component_cells_0p20.append(comp20[0])
        component_touches_surface_0p05.append(float(comp05[2]))
        component_touches_surface_0p20.append(float(comp20[2]))
        surface_damage_maxima.append(surface_damage)
        damage_area_fractions.append(surface_area_fraction)
        cap_trigger_total += step_cap_trigger_count
        overflow_volume_total += step_overflow_volume
        for threshold in damage_event_times:
            if damage_event_times[threshold] is None and float(np.max(sim.d)) >= threshold:
                damage_event_times[threshold] = float(t)
        for threshold in opening_event_times:
            if opening_event_times[threshold] is None and float(sim.surface_crack_width_displacement_mm) >= threshold:
                opening_event_times[threshold] = float(t)
        crack_w.append(sim.surface_crack_width_mm)
        crack_w_disp.append(sim.surface_crack_width_displacement_mm)
        crack_w_envelope.append(sim.surface_crack_width_envelope_mm)
        mechanics_residuals.append(step_residual)
        mechanics_calls.append(step_operator_calls)
        staggered_counts.append(step_staggered_count)
        coupling_pressure_residuals.append(step_coupling_residual)
        transport_residual = max(
            _transport_residual(sim._solve_s),
            _transport_residual(sim._solve_c),
            _transport_residual(sim._solve_na),
            _transport_residual(sim._solve_oh),
        )
        transport_residuals.append(transport_residual)
        sim.last_step_phasefield_residual = step_phasefield_residual
        sim.last_step_phasefield_iterations = step_phasefield_iterations
        phasefield_residuals.append(step_phasefield_residual)
        if step_callback is not None:
            step_callback(sim, t)
        if progress and len(times) % 10 == 0:
            print(
                f"full 3D {params.case_id}: t={t:.1f} yr, p={max_p[-1]:.3f} MPa, d={max_d[-1]:.3f}, w={crack_w[-1]:.4f} mm",
                flush=True,
            )
        if termination_condition is not None and termination_condition(sim, t):
            break
    elapsed = time.time() - start
    dep = sim.t_ini_map[np.isfinite(sim.t_ini_map)]
    sim.peak_rust_pressure = peak_pressure_field
    sim.peak_icorr = peak_icorr_field
    sim.peak_electric_grad = peak_electric_field
    summary = SolverSummary(
        elapsed,
        float(np.min(dep)) if dep.size else None,
        float(np.max(sim.C)),
        float(np.max(sim.electric_grad)),
        float(np.max(sim.i_corr)),
        float(np.max(sim.rust_pressure) / 1000000.0),
        float(np.max(sim.d)),
    )
    histories = {
        "times": np.asarray(times),
        "max_chloride": np.asarray(max_c),
        "max_pressure_mpa": np.asarray(max_p),
        "max_damage": np.asarray(max_d),
        "max_icorr": np.asarray(max_i),
        "elastic_energy_j": np.asarray(elastic_e),
        "fracture_energy_j": np.asarray(fracture_e),
        "pressure_work_j": np.asarray(pressure_work),
        "mechanical_energy_relative_error": np.asarray(energy_error),
        "rust_generated_m": np.asarray(rust_gen),
        "rust_retained_m": np.asarray(rust_ret),
        "rust_lost_m": np.asarray(rust_loss),
        "rust_mass_balance_error_m": np.asarray(rust_err),
        "rust_generated_volume_m3": np.asarray(rust_gen_volume),
        "rust_retained_volume_m3": np.asarray(rust_ret_volume),
        "rust_lost_volume_m3": np.asarray(rust_loss_volume),
        "rust_mass_balance_error_m3": np.asarray(rust_err_volume),
        "pressure_cap_trigger_count": np.asarray(cap_trigger_counts),
        "pressure_overflow_volume_m3": np.asarray(overflow_volumes),
        "rust_lost_fraction": np.asarray(retained_loss_ratios),
        "max_tensile_strain": np.asarray(tensile_strain_maxima),
        "max_phasefield_driving_energy": np.asarray(driving_energy_maxima),
        "trial_pressure_max_mpa": np.asarray(trial_pressure_maxima),
        "surface_max_damage": np.asarray(surface_damage_maxima),
        "surface_damage_area_fraction_0p05": np.asarray(damage_area_fractions),
        "damage_component_cells_0p05": np.asarray(component_cells_0p05),
        "damage_component_cells_0p20": np.asarray(component_cells_0p20),
        "damage_component_touches_surface_0p05": np.asarray(component_touches_surface_0p05),
        "damage_component_touches_surface_0p20": np.asarray(component_touches_surface_0p20),
        "damage_event_time_0p05": np.asarray([damage_event_times[0.05] or np.nan] * len(times)),
        "damage_event_time_0p10": np.asarray([damage_event_times[0.10] or np.nan] * len(times)),
        "damage_event_time_0p20": np.asarray([damage_event_times[0.20] or np.nan] * len(times)),
        "damage_event_time_0p35": np.asarray([damage_event_times[0.35] or np.nan] * len(times)),
        "opening_event_time_0p005mm": np.asarray([opening_event_times[0.005] or np.nan] * len(times)),
        "opening_event_time_0p01mm": np.asarray([opening_event_times[0.01] or np.nan] * len(times)),
        "opening_event_time_0p05mm": np.asarray([opening_event_times[0.05] or np.nan] * len(times)),
        "surface_crack_width_mm": np.asarray(crack_w),
        "surface_crack_width_displacement_mm": np.asarray(crack_w_disp),
        "surface_crack_width_envelope_mm": np.asarray(crack_w_envelope),
        "mechanics_relative_residual": np.asarray(mechanics_residuals),
        "mechanics_operator_calls": np.asarray(mechanics_calls),
        "staggered_iterations": np.asarray(staggered_counts),
        "coupling_pressure_relative_residual": np.asarray(coupling_pressure_residuals),
        "transport_relative_residual": np.asarray(transport_residuals),
        "phasefield_relative_residual": np.asarray(phasefield_residuals),
        "pressure_cap_trigger_total": np.asarray([cap_trigger_total]),
        "pressure_overflow_volume_total_m3": np.asarray([overflow_volume_total]),
    }
    return (sim, histories, summary)


def phasefield_gc_from_tensile_strength(
    tensile_strength_pa: float, elastic_modulus_pa: float, length_scale_m: float
) -> float:
    """Return the AT2 fracture energy associated with a tensile scale."""
    return float(
        256.0
        * length_scale_m
        * tensile_strength_pa**2
        / (27.0 * elastic_modulus_pa)
    )


def run_accelerated_corrosion_3d(
    case_id: str,
    cover_depth_m: float,
    rebar_radius_m: float,
    tensile_strength_mpa: float,
    corrosion_current_ua_cm2: float,
    total_hours: float = 3600.0,
    dt_hours: float = 25.0,
    seed: int = 201,
    grid_spacing: float | None = 0.0025,
    mechanics_iterations: int = 60,
    rust_expansion_ratio: float = 2.5,
    porous_gap_m: float = 8.0e-6,
    rust_bulk_modulus_pa: float = 2.0e9,
    crack_storage_coeff_m: float = 1.2e-3,
    leakage_damage_factor: float = 0.50,
    leakage_pressure_scale: float = 1.0,
    aggregate_fraction: float = 0.30,
    fracture_energy_n_m: float | None = None,
    elastic_modulus_scale: float = 1.0,
    corrosion_current_factor: float = 1.0,
    corrosion_angular_concentration: float = 0.0,
    corrosion_profile_mode: str = "uniform_von_mises",
    leakage_law: str = "instantaneous_cap",
    leakage_rate: float = 0.25,
    leakage_exponent: float = 1.0,
    leakage_permeability_m_per_year: float = 5.0e-4,
    leakage_damage_factor_kinetic: float = 4.0,
    leakage_opening_factor: float = 4.0,
    leakage_opening_reference_m: float = 5.0e-5,
    wedge_compacted_fraction: float = 0.75,
    wedge_conversion_fraction: float = 1.0,
    wedge_contact_stiffness_pa_per_m: float = 4.0e11,
    compacted_dissolution_rate_per_year: float = 0.0,
    damage_coupling_tolerance: float = 1.0e-4,
    interface_flaw_factor: float = 1.0,
    steel_interface_fracture_factor: float = 1.0,
    steel_interface_stiffness_factor: float = 1.0,
    surface_crack_width_scale: float = 1.0,
    phasefield_length_m: float = 0.005,
    residual_stiffness: float = 1.0e-5,
    steel_interface_width_m: float | None = None,
    surface_observation_depth_m: float | None = 0.01,
    domain_width_m: float | None = None,
    domain_height_m: float | None = None,
    domain_thickness_m: float | None = None,
    itz_width_m: float = 0.003,
    surface_gauge_length_m: float = 0.02,
    axial_symmetry_ends: bool = True,
    stop_after_spanning_threshold: float | None = None,
    stop_after_opening_threshold_mm: float | None = None,
    unload_after_opening_event: bool = False,
    post_event_hours: float = 0.0,
    progress: bool = False,
    progress_interval_steps: int = 20,
    adaptive_time_stepping: bool = False,
    minimum_dt_hours: float | None = None,
    checkpoint_path: str | None = None,
    resume_from_checkpoint: bool = False,
):
    """Run a prescribed-current accelerated test with the residual-controlled core."""
    ft_pa = float(tensile_strength_mpa) * 1.0e6
    elastic_modulus = max(
        4.0e9,
        elastic_modulus_scale
        * max(18.0e9, 22.0e9 + 2.0e9 * float(tensile_strength_mpa) / 3.0),
    )
    length_scale = float(phasefield_length_m)
    gc = (
        float(fracture_energy_n_m)
        if fracture_energy_n_m is not None
        else phasefield_gc_from_tensile_strength(ft_pa, elastic_modulus, length_scale)
    )
    params = EngineeringParameters(
        case_id=case_id,
        cover_depth=cover_depth_m,
        rebar_radius=rebar_radius_m,
        surface_chloride=4.0,
        critical_chloride=0.0,
        mortar_chloride_diffusivity=1.0e-4,
        itz_diffusivity_multiplier=5.0,
        reference_corrosion_current=corrosion_current_ua_cm2 * 0.01,
        rust_expansion_ratio=rust_expansion_ratio,
        porous_gap=porous_gap_m,
        crack_storage_coeff=crack_storage_coeff_m,
        rust_bulk_modulus=rust_bulk_modulus_pa,
        fracture_energy=gc,
        aggregate_fraction=aggregate_fraction,
        seed=seed,
        grid_spacing=grid_spacing,
        phasefield_length=length_scale,
        residual_stiffness=residual_stiffness,
        steel_interface_width=steel_interface_width_m,
        surface_observation_depth=surface_observation_depth_m,
        domain_width=domain_width_m,
        domain_height=domain_height_m,
        domain_thickness=domain_thickness_m,
        itz_width=itz_width_m,
        surface_gauge_length=surface_gauge_length_m,
        interface_flaw_factor=interface_flaw_factor,
        steel_interface_fracture_factor=steel_interface_fracture_factor,
        steel_interface_stiffness_factor=steel_interface_stiffness_factor,
        axial_transport_enabled=True,
        axial_symmetry_ends=axial_symmetry_ends,
    )
    sim, _, _ = prepare_3d_solver(params, dt=dt_hours / SECONDS_PER_YEAR)
    sim.E_mat_base = elastic_modulus
    sim.E_map = np.where(
        sim.is_agg, 60.0e9, np.where(sim.is_itz, 0.6 * elastic_modulus, elastic_modulus)
    )
    interface_material = sim.rebar_bnd & ~sim.is_agg
    sim.E_map[interface_material] = np.minimum(
        sim.E_map[interface_material],
        elastic_modulus * steel_interface_stiffness_factor,
    )
    sim.K_bulk = sim.E_map / (3.0 * (1.0 - 2.0 * sim.nu))
    sim.mu_map = sim.E_map / (2.0 * (1.0 + sim.nu))
    sim.lam_map = sim.K_bulk - 2.0 / 3.0 * sim.mu_map
    sim._mechanics_stiffness = None
    sim._mechanics_stiffness_damage = None
    sim._mechanics_preconditioners = None
    sim._mechanics_direct_preconditioner = None
    sim._mechanics_direct_factor = None
    sim.surface_crack_width_scale = surface_crack_width_scale
    sim.S[:] = 1.0
    sim.C[:] = 4.0
    sim.Na[:] = 560.0
    sim.OH[:] = 80.0

    dt_year = float(dt_hours) / (365.25 * 24.0)
    current_a_m2 = float(corrosion_current_ua_cm2) * 0.01 * float(corrosion_current_factor)
    angle_from_cover = np.arctan2(sim.X - sim.cx, sim.Y - sim.cy)
    if corrosion_profile_mode == "xu2024_fig8":
        # Xu et al. (2024), Fig. 8a reports three separate parallel rebars,
        # not three axial stations of one bar. Use their pooled angular fit.
        # Figure 3 defines theta=180 degrees on the cover-facing side, whereas
        # angle_from_cover=0 here, hence the 180-degree coordinate rotation.
        baseline = 0.0
        amplitude = 1.3803690337
        kappa = 0.1814114004
        direction = np.deg2rad(323.0123442 - 180.0)
        current_profile = baseline + amplitude * np.exp(
            kappa * (np.cos(angle_from_cover - direction) - 1.0)
        )
    elif corrosion_profile_mode == "uniform_von_mises":
        current_profile = np.exp(
            np.clip(float(corrosion_angular_concentration), 0.0, 8.0)
            * np.cos(angle_from_cover)
        )
    else:
        raise ValueError(f"unknown corrosion_profile_mode: {corrosion_profile_mode}")
    current_profile[~sim.rebar_bnd] = 0.0
    area_weights = sim.rebar_area_weight[sim.rebar_bnd]
    weighted_mean = float(
        np.sum(area_weights * current_profile[sim.rebar_bnd])
        / max(float(np.sum(area_weights)), 1.0e-30)
    )
    current_profile /= max(weighted_mean, 1.0e-30)
    p_cr = hoop_cracking_pressure(sim.R_bar, sim.R_bar + sim.cover_depth, ft_pa)
    times_h, max_p, max_d, max_i, crack_w = [], [], [], [], []
    elastic_e, fracture_e, pressure_work, energy_error = [], [], [], []
    rust_gen, rust_ret, rust_loss, rust_err = [], [], [], []
    rust_mobile_hist, rust_compacted_hist, wedge_hist = [], [], []
    mechanics_residuals, coupling_residuals, phasefield_residuals = [], [], []
    contact_min_separations, contact_active_iterations, contact_active_cells = [], [], []
    contact_linear_iterations = []
    contact_pdas_fallback_used = []
    cap_trigger_counts, overflow_volumes = [], []
    tensile_strain_maxima, driving_energy_maxima = [], []
    component_cells_0p05, component_cells_0p20 = [], []
    component_touches_surface_0p05, component_touches_surface_0p20 = [], []
    trial_pressure_maxima = []
    damage_event_times = {0.05: None, 0.10: None, 0.20: None, 0.35: None}
    spanning_event_times = {0.05: None, 0.20: None}
    opening_event_times = {0.005: None, 0.01: None, 0.05: None}
    cap_trigger_total = 0
    overflow_volume_total = 0.0
    stop_time_h = None
    corrosion_active = True
    peak_pressure_field = np.zeros_like(sim.rust_pressure)
    peak_current_field = np.zeros_like(sim.i_corr)
    peak_electric_field = np.zeros_like(sim.electric_grad)
    t_h = 0.0
    nominal_dt_hours = float(dt_hours)
    current_dt_hours = nominal_dt_hours
    minimum_step_hours = (
        max(float(minimum_dt_hours), 1.0e-9)
        if minimum_dt_hours is not None
        else max(nominal_dt_hours / 8.0, 1.0e-9)
    )
    step_sizes_h, damage_coupling_iterations = [], []
    rollback_fields = (
        "i_corr",
        "rust_generated",
        "rust_volume",
        "rust_lost",
        "rust_filled",
        "rust_mobile",
        "rust_compacted",
        "residual_wedge",
        "rust_pressure",
        "u_r",
        "u",
        "v",
        "w",
        "d",
        "H_hist",
    )
    history_targets = {
        "times_h": times_h,
        "max_p": max_p,
        "max_d": max_d,
        "max_i": max_i,
        "crack_w": crack_w,
        "elastic_e": elastic_e,
        "fracture_e": fracture_e,
        "pressure_work": pressure_work,
        "energy_error": energy_error,
        "rust_gen": rust_gen,
        "rust_ret": rust_ret,
        "rust_loss": rust_loss,
        "rust_err": rust_err,
        "rust_mobile": rust_mobile_hist,
        "rust_compacted": rust_compacted_hist,
        "wedge": wedge_hist,
        "mechanics_residuals": mechanics_residuals,
        "coupling_residuals": coupling_residuals,
        "phasefield_residuals": phasefield_residuals,
        "contact_min_separations": contact_min_separations,
        "contact_active_iterations": contact_active_iterations,
        "contact_active_cells": contact_active_cells,
        "contact_linear_iterations": contact_linear_iterations,
        "contact_fallback": contact_pdas_fallback_used,
        "cap_trigger_counts": cap_trigger_counts,
        "overflow_volumes": overflow_volumes,
        "tensile_strain": tensile_strain_maxima,
        "driving_energy": driving_energy_maxima,
        "component_cells_0p05": component_cells_0p05,
        "component_cells_0p20": component_cells_0p20,
        "component_surface_0p05": component_touches_surface_0p05,
        "component_surface_0p20": component_touches_surface_0p20,
        "trial_pressure": trial_pressure_maxima,
        "step_sizes_h": step_sizes_h,
        "damage_coupling_iterations": damage_coupling_iterations,
    }
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path else None
    interface_area = max(
        float(np.sum(sim.rebar_area_weight[sim.rebar_bnd])), 1.0e-30
    )

    if resume_from_checkpoint and checkpoint is not None and checkpoint.exists():
        with checkpoint.open("rb") as stream:
            saved = pickle.load(stream)
        if saved.get("case_id") != case_id:
            raise ValueError("checkpoint case_id does not match requested case")
        for name in rollback_fields:
            getattr(sim, name)[:] = saved["state"][name]
        (
            sim.surface_crack_width_mm,
            sim.surface_crack_width_displacement_mm,
            sim.surface_crack_width_envelope_mm,
        ) = saved["surface_widths"]
        for name, target in history_targets.items():
            target.extend(saved["histories"][name])
        t_h = float(saved["t_h"])
        current_dt_hours = float(saved["current_dt_hours"])
        cap_trigger_total = int(saved["cap_trigger_total"])
        overflow_volume_total = float(saved["overflow_volume_total"])
        stop_time_h = saved["stop_time_h"]
        corrosion_active = bool(saved["corrosion_active"])
        peak_pressure_field[:] = saved["peak_pressure_field"]
        peak_current_field[:] = saved["peak_current_field"]
        peak_electric_field[:] = saved["peak_electric_field"]
        damage_event_times.update(saved["damage_event_times"])
        spanning_event_times.update(saved["spanning_event_times"])
        opening_event_times.update(saved["opening_event_times"])
        sim._mechanics_stiffness = None
        sim._mechanics_stiffness_damage = None
        sim._mechanics_preconditioners = None
        sim._mechanics_direct_preconditioner = None
        sim._mechanics_direct_factor = None
        if progress:
            print(
                f"{case_id}: resumed from t={t_h:.3f} h, "
                f"dt={current_dt_hours:.3f} h",
                flush=True,
            )

    def save_checkpoint():
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_id": case_id,
            "t_h": t_h,
            "current_dt_hours": current_dt_hours,
            "state": {
                name: np.asarray(getattr(sim, name)).copy()
                for name in rollback_fields
            },
            "surface_widths": (
                sim.surface_crack_width_mm,
                sim.surface_crack_width_displacement_mm,
                sim.surface_crack_width_envelope_mm,
            ),
            "histories": {
                name: list(values) for name, values in history_targets.items()
            },
            "cap_trigger_total": cap_trigger_total,
            "overflow_volume_total": overflow_volume_total,
            "stop_time_h": stop_time_h,
            "corrosion_active": corrosion_active,
            "peak_pressure_field": peak_pressure_field.copy(),
            "peak_current_field": peak_current_field.copy(),
            "peak_electric_field": peak_electric_field.copy(),
            "damage_event_times": damage_event_times.copy(),
            "spanning_event_times": spanning_event_times.copy(),
            "opening_event_times": opening_event_times.copy(),
        }
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(checkpoint)

    while t_h < float(total_hours) - 1.0e-12:
        actual_h = min(current_dt_hours, float(total_hours) - t_h)
        step_snapshot = {
            name: np.asarray(getattr(sim, name)).copy()
            for name in rollback_fields
        }
        width_snapshot = (
            sim.surface_crack_width_mm,
            sim.surface_crack_width_displacement_mm,
            sim.surface_crack_width_envelope_mm,
        )
        cap_trigger_snapshot = cap_trigger_total
        actual_year = actual_h / (365.25 * 24.0)
        sim.i_corr[:] = 0.0
        if corrosion_active:
            sim.i_corr[sim.rebar_bnd] = (
                current_a_m2 * current_profile[sim.rebar_bnd]
            )
        normal_opening = np.maximum(sim.u * sim.nx + sim.v * sim.ny, 0.0)
        steel_loss = (
            sim.i_corr
            * M_FE
            * SECONDS_PER_YEAR
            * actual_year
            / (N_FE * FARADAY * RHO_FE)
        )
        sim.rust_generated += sim.rust_expansion_ratio * steel_loss
        generated_step = sim.rust_expansion_ratio * steel_loss
        if leakage_law == "unilateral_wedge":
            rust_update = _advance_unilateral_wedge_state(
                sim,
                generated_step,
                actual_year,
                sim.rust_pressure,
                normal_opening,
                leakage_pressure_scale * p_cr,
                leakage_permeability_m_per_year,
                leakage_exponent,
                leakage_damage_factor_kinetic,
                leakage_opening_factor,
                leakage_opening_reference_m,
                wedge_compacted_fraction,
                wedge_conversion_fraction,
                compacted_dissolution_rate_per_year,
            )
            sim.rust_pressure[:] = _unilateral_wedge_pressure(
                sim, normal_opening, wedge_contact_stiffness_pa_per_m
            )
        elif leakage_law == "kinetic_permeability":
            rust_update = _update_rust_state_kinetic(
                sim.rust_volume,
                sim.rust_lost,
                sim.rust_filled,
                sim.rust_pressure,
                sim.i_corr,
                sim.d,
                sim.S,
                actual_year,
                sim.rebar_bnd,
                sim.delta_0_gap,
                sim.rust_expansion_ratio,
                sim.rust_bulk_modulus,
                leakage_pressure_scale * p_cr,
                normal_opening,
                0.0,
                p_cr,
                leakage_exponent=leakage_exponent,
                leakage_damage_factor=leakage_damage_factor_kinetic,
                leakage_opening_factor=leakage_opening_factor,
                leakage_opening_reference=leakage_opening_reference_m,
            )
            kinetic_inventory = sim.rust_volume.copy()
            kinetic_lost_base = sim.rust_lost.copy()
            (
                retained_after,
                kinetic_removed,
                filled_after,
                pressure_after,
                kinetic_trial_pressure,
            ) = _kinetic_leakage_equilibrium(
                kinetic_inventory,
                sim.d,
                sim.S,
                actual_year,
                sim.rebar_bnd,
                sim.delta_0_gap,
                sim.rust_bulk_modulus,
                leakage_pressure_scale * p_cr,
                normal_opening,
                leakage_permeability_m_per_year,
                p_cr,
                leakage_exponent,
                leakage_damage_factor_kinetic,
                leakage_opening_factor,
                leakage_opening_reference_m,
            )
            sim.rust_volume[:] = retained_after
            sim.rust_lost[:] = kinetic_lost_base + kinetic_removed
            sim.rust_filled[:] = filled_after
            sim.rust_pressure[:] = pressure_after
            rust_update["cap_triggered_cells"] = int(
                np.count_nonzero(
                    (kinetic_trial_pressure > leakage_pressure_scale * p_cr)
                    & sim.rebar_bnd
                )
            )
            rust_update["overflow_removed_m"] = kinetic_removed
        else:
            rust_update = _update_rust_state(
                sim.rust_volume,
                sim.rust_lost,
                sim.rust_filled,
                sim.rust_pressure,
                sim.i_corr,
                sim.d,
                sim.S,
                actual_year,
                sim.rebar_bnd,
                sim.delta_0_gap,
                sim.rust_expansion_ratio,
                sim.rust_bulk_modulus,
                p_cr,
                crack_storage_coeff=sim.crack_storage_coeff,
                leakage_pressure_scale=leakage_pressure_scale,
                leakage_damage_factor=leakage_damage_factor,
                leakage_law=leakage_law,
                leakage_rate=leakage_rate,
                leakage_exponent=leakage_exponent,
                mechanical_opening=normal_opening,
            )
        trial_pressure_mpa = float(np.max(sim.rust_pressure) / 1.0e6)
        cap_trigger_total += rust_update["cap_triggered_cells"]
        step_overflow_volume = float(
            np.sum(sim.rebar_area_weight * rust_update["overflow_removed_m"])
        )
        sim.u_r = np.maximum(sim.rust_volume - sim.rust_filled, 0.0)
        sim.u_r[~sim.rebar_bnd] = 0.0
        step_mechanics = 0.0
        step_coupling = 0.0
        step_phasefield = 0.0
        step_coupling_iteration_count = 0
        if np.max(sim.rust_pressure) > 0.01 * p_cr:
            converged = False
            if leakage_law == "unilateral_wedge":
                for _ in range(max(int(mechanics_iterations), 1)):
                    step_coupling_iteration_count += 1
                    damage_before = sim.d.copy()
                    normal_opening, step_coupling = _solve_unilateral_wedge_contact(
                        sim,
                        wedge_contact_stiffness_pa_per_m,
                        sim.rust_pressure,
                    )
                    sim.solve_mechanics_pseudo_transient(
                        sim.u_r, sim.rust_pressure, update_history=True
                    )
                    solve_phase_field_cg_3d(sim)
                    damage_change = float(np.max(np.abs(sim.d - damage_before)))
                    step_mechanics = max(step_mechanics, sim.last_mechanics_residual)
                    step_phasefield = max(step_phasefield, sim.last_phasefield_residual)
                    if (
                        damage_change <= max(float(damage_coupling_tolerance), 0.0)
                        and step_coupling <= 1.0e-4
                        and sim.last_mechanics_converged
                    ):
                        converged = True
                        break
            elif leakage_law == "kinetic_permeability":
                for _ in range(max(min(int(mechanics_iterations), 50), 1)):
                    step_coupling_iteration_count += 1
                    damage_before = sim.d.copy()
                    (
                        kinetic_result,
                        normal_opening,
                        step_coupling,
                    ) = _solve_kinetic_contact(
                        sim,
                        kinetic_inventory,
                        actual_year,
                        leakage_pressure_scale * p_cr,
                        leakage_permeability_m_per_year,
                        p_cr,
                        leakage_exponent,
                        leakage_damage_factor_kinetic,
                        leakage_opening_factor,
                        leakage_opening_reference_m,
                        sim.rust_pressure,
                    )
                    (
                        retained_after,
                        kinetic_removed,
                        filled_after,
                        _,
                        kinetic_trial_pressure,
                    ) = kinetic_result
                    sim.rust_volume[:] = retained_after
                    sim.rust_lost[:] = kinetic_lost_base + kinetic_removed
                    sim.rust_filled[:] = filled_after
                    sim.u_r[:] = np.maximum(
                        sim.rust_volume - sim.rust_filled, 0.0
                    )
                    sim.u_r[~sim.rebar_bnd] = 0.0
                    sim.solve_mechanics_pseudo_transient(
                        sim.u_r, sim.rust_pressure, update_history=True
                    )
                    solve_phase_field_cg_3d(sim)
                    damage_change = float(np.max(np.abs(sim.d - damage_before)))
                    step_mechanics = max(step_mechanics, sim.last_mechanics_residual)
                    step_phasefield = max(step_phasefield, sim.last_phasefield_residual)
                    if (
                        damage_change <= 1.0e-4
                        and step_coupling <= 1.0e-4
                        and sim.last_mechanics_converged
                    ):
                        converged = True
                        break
            else:
                for _ in range(max(int(mechanics_iterations), 1)):
                    step_coupling_iteration_count += 1
                    damage_before = sim.d.copy()
                    pressure_before = sim.rust_pressure.copy()
                    sim.solve_mechanics_pseudo_transient(sim.u_r, sim.rust_pressure)
                    solve_phase_field_cg_3d(sim)
                    target_pressure = _contact_pressure_from_retained_rust(
                        sim,
                        update_filled=False,
                        pressure_limit=p_cr,
                        pressure_limit_scale=leakage_pressure_scale,
                        pressure_limit_damage=leakage_damage_factor,
                        leakage_law=leakage_law,
                    )
                    scale = max(float(np.max(np.abs(target_pressure))), 1.0)
                    step_coupling = float(
                        np.max(np.abs(target_pressure - pressure_before)) / scale
                    )
                    damage_change = float(np.max(np.abs(sim.d - damage_before)))
                    sim.rust_pressure[:] = target_pressure
                    step_mechanics = max(step_mechanics, sim.last_mechanics_residual)
                    step_phasefield = max(step_phasefield, sim.last_phasefield_residual)
                    if (
                        damage_change <= 1.0e-4
                        and step_coupling <= 1.0e-4
                        and sim.last_mechanics_converged
                    ):
                        converged = True
                        break
            if not converged:
                if (
                    adaptive_time_stepping
                    and actual_h > minimum_step_hours + 1.0e-12
                ):
                    for name, values in step_snapshot.items():
                        getattr(sim, name)[:] = values
                    (
                        sim.surface_crack_width_mm,
                        sim.surface_crack_width_displacement_mm,
                        sim.surface_crack_width_envelope_mm,
                    ) = width_snapshot
                    cap_trigger_total = cap_trigger_snapshot
                    current_dt_hours = max(
                        0.5 * actual_h, minimum_step_hours
                    )
                    sim._mechanics_stiffness = None
                    sim._mechanics_stiffness_damage = None
                    sim._mechanics_preconditioners = None
                    sim._mechanics_direct_preconditioner = None
                    sim._mechanics_direct_factor = None
                    if progress:
                        print(
                            f"{case_id}: retry t={t_h:.3f} h with "
                            f"dt={current_dt_hours:.3f} h after "
                            f"{step_coupling_iteration_count} coupling iterations",
                            flush=True,
                        )
                    continue
                raise RuntimeError(
                    f"accelerated coupled solve failed at {t_h:.3f} h: "
                    f"damage increment={damage_change:.3e}, pressure residual={step_coupling:.3e}, "
                    f"contact linear info={sim.last_contact_linear_info}, "
                    f"active iterations={sim.last_contact_active_iterations}"
                )
        if leakage_law == "kinetic_permeability":
            rust_update["overflow_removed_m"] = kinetic_removed
            step_overflow_volume = float(
                np.sum(sim.rebar_area_weight * kinetic_removed)
            )
            trial_pressure_mpa = max(
                trial_pressure_mpa,
                float(np.max(kinetic_trial_pressure) / 1.0e6),
            )
        overflow_volume_total += step_overflow_volume
        sim.update_energy_diagnostics()
        tensile_strain, driving_energy = sim.diagnostic_tensile_strain_and_driving_energy()
        peak_pressure_field = np.maximum(peak_pressure_field, sim.rust_pressure)
        peak_current_field = np.maximum(peak_current_field, sim.i_corr)
        sim.update_electric_field()
        peak_electric_field = np.maximum(peak_electric_field, sim.electric_grad)
        t_h += actual_h
        step_sizes_h.append(actual_h)
        damage_coupling_iterations.append(step_coupling_iteration_count)
        if adaptive_time_stepping:
            if step_coupling_iteration_count >= 100:
                current_dt_hours = max(
                    0.5 * actual_h, minimum_step_hours
                )
            elif step_coupling_iteration_count <= 12:
                current_dt_hours = min(
                    2.0 * actual_h, nominal_dt_hours
                )
        times_h.append(t_h)
        max_p.append(float(np.max(sim.rust_pressure) / 1.0e6))
        max_d.append(float(np.max(sim.d)))
        max_i.append(float(np.max(sim.i_corr)))
        crack_w.append(float(sim.surface_crack_width_mm))
        elastic_e.append(sim.elastic_energy_j)
        fracture_e.append(sim.fracture_energy_j)
        pressure_work.append(sim.pressure_work_j)
        energy_error.append(sim.mechanical_energy_relative_error)
        generated_volume = float(np.sum(sim.rebar_area_weight * sim.rust_generated))
        retained_volume = float(np.sum(sim.rebar_area_weight * sim.rust_volume))
        lost_volume = float(np.sum(sim.rebar_area_weight * sim.rust_lost))
        gen = generated_volume / interface_area
        ret = retained_volume / interface_area
        lost = lost_volume / interface_area
        rust_gen.append(gen)
        rust_ret.append(ret)
        rust_loss.append(lost)
        rust_err.append(gen - ret - lost)
        rust_mobile_hist.append(
            float(np.sum(sim.rebar_area_weight * sim.rust_mobile)) / interface_area
        )
        rust_compacted_hist.append(
            float(np.sum(sim.rebar_area_weight * sim.rust_compacted)) / interface_area
        )
        wedge_hist.append(
            float(np.sum(sim.rebar_area_weight * sim.residual_wedge)) / interface_area
        )
        cap_trigger_counts.append(int(rust_update["cap_triggered_cells"]))
        overflow_volumes.append(step_overflow_volume)
        tensile_strain_maxima.append(tensile_strain)
        driving_energy_maxima.append(driving_energy)
        trial_pressure_maxima.append(trial_pressure_mpa)
        comp05 = sim.diagnostic_damage_component(0.05)
        comp20 = sim.diagnostic_damage_component(0.20)
        span05 = sim.diagnostic_spanning_damage_component(0.05)
        span20 = sim.diagnostic_spanning_damage_component(0.20)
        component_cells_0p05.append(comp05[0])
        component_cells_0p20.append(comp20[0])
        component_touches_surface_0p05.append(float(comp05[2]))
        component_touches_surface_0p20.append(float(comp20[2]))
        mechanics_residuals.append(step_mechanics)
        coupling_residuals.append(step_coupling)
        phasefield_residuals.append(step_phasefield)
        contact_min_separations.append(sim.last_contact_min_separation_m)
        contact_active_iterations.append(sim.last_contact_active_iterations)
        contact_active_cells.append(sim.last_contact_active_cells)
        contact_linear_iterations.append(sim.last_contact_linear_iterations)
        contact_pdas_fallback_used.append(float(sim.last_contact_pdas_fallback_used))
        for threshold in damage_event_times:
            if damage_event_times[threshold] is None and float(np.max(sim.d)) >= threshold:
                damage_event_times[threshold] = float(t_h / (365.25 * 24.0))
        for threshold, spanning in ((0.05, span05), (0.20, span20)):
            if spanning_event_times[threshold] is None and spanning[2]:
                spanning_event_times[threshold] = float(t_h / (365.25 * 24.0))
                if (
                    stop_after_spanning_threshold is not None
                    and abs(float(stop_after_spanning_threshold) - threshold) <= 1.0e-12
                ):
                    stop_time_h = t_h + max(float(post_event_hours), 0.0)
        for threshold in opening_event_times:
            if opening_event_times[threshold] is None and float(sim.surface_crack_width_displacement_mm) >= threshold:
                opening_event_times[threshold] = float(t_h / (365.25 * 24.0))
        if (
            stop_after_opening_threshold_mm is not None
            and stop_time_h is None
            and float(sim.surface_crack_width_displacement_mm)
            >= float(stop_after_opening_threshold_mm)
        ):
            if unload_after_opening_event:
                corrosion_active = False
            stop_time_h = t_h + max(float(post_event_hours), 0.0)
        progress_interval = max(int(progress_interval_steps), 1)
        if progress and len(times_h) % progress_interval == 0:
            print(
                f"{case_id}: t={t_h:.3f} h, dt={actual_h:.3f} h, "
                f"outer_it={step_coupling_iteration_count}, "
                f"p={max_p[-1]:.3f} MPa, "
                f"d={max_d[-1]:.3f}, w={crack_w[-1]:.3f} mm, "
                f"contact_it={sim.last_contact_active_iterations}, "
                f"linear_it={sim.last_contact_linear_iterations}, "
                f"fallback={int(sim.last_contact_pdas_fallback_used)}",
                flush=True,
            )
        save_checkpoint()
        if stop_time_h is not None and t_h >= stop_time_h - 1.0e-12:
            break
    sim.peak_rust_pressure = peak_pressure_field
    sim.peak_icorr = peak_current_field
    sim.peak_electric_grad = peak_electric_field
    histories = {
        "times_h": np.asarray(times_h),
        "times": np.asarray(times_h) / (365.25 * 24.0),
        "max_chloride": np.full(len(times_h), 4.0),
        "max_pressure_mpa": np.asarray(max_p),
        "max_damage": np.asarray(max_d),
        "max_icorr": np.asarray(max_i),
        "surface_crack_width_mm": np.asarray(crack_w),
        "surface_crack_width_envelope_mm": np.maximum.accumulate(crack_w),
        "elastic_energy_j": np.asarray(elastic_e),
        "fracture_energy_j": np.asarray(fracture_e),
        "pressure_work_j": np.asarray(pressure_work),
        "mechanical_energy_relative_error": np.asarray(energy_error),
        "rust_generated_m": np.asarray(rust_gen),
        "rust_retained_m": np.asarray(rust_ret),
        "rust_lost_m": np.asarray(rust_loss),
        "rust_mobile_m": np.asarray(rust_mobile_hist),
        "rust_compacted_m": np.asarray(rust_compacted_hist),
        "residual_wedge_m": np.asarray(wedge_hist),
        "rust_mass_balance_error_m": np.asarray(rust_err),
        "rust_generated_volume_m3": np.asarray(rust_gen) * interface_area,
        "rust_retained_volume_m3": np.asarray(rust_ret) * interface_area,
        "rust_lost_volume_m3": np.asarray(rust_loss) * interface_area,
        "rust_mass_balance_error_m3": np.asarray(rust_err) * interface_area,
        "rust_lost_fraction": np.asarray(rust_loss)
        / np.maximum(np.asarray(rust_ret) + np.asarray(rust_loss), 1e-30),
        "mechanics_relative_residual": np.asarray(mechanics_residuals),
        "coupling_pressure_relative_residual": np.asarray(coupling_residuals),
        "contact_min_separation_m": np.asarray(contact_min_separations),
        "contact_active_iterations": np.asarray(contact_active_iterations),
        "contact_active_cells": np.asarray(contact_active_cells),
        "contact_linear_iterations": np.asarray(contact_linear_iterations),
        "contact_pdas_fallback_used": np.asarray(contact_pdas_fallback_used),
        "time_step_hours": np.asarray(step_sizes_h),
        "damage_coupling_iterations": np.asarray(damage_coupling_iterations),
        "phasefield_relative_residual": np.asarray(phasefield_residuals),
        "transport_relative_residual": np.zeros(len(times_h)),
        "pressure_cap_trigger_count": np.asarray(cap_trigger_counts),
        "pressure_overflow_volume_m3": np.asarray(overflow_volumes),
        "max_tensile_strain": np.asarray(tensile_strain_maxima),
        "max_phasefield_driving_energy": np.asarray(driving_energy_maxima),
        "trial_pressure_max_mpa": np.asarray(trial_pressure_maxima),
        "damage_component_cells_0p05": np.asarray(component_cells_0p05),
        "damage_component_cells_0p20": np.asarray(component_cells_0p20),
        "damage_component_touches_surface_0p05": np.asarray(component_touches_surface_0p05),
        "damage_component_touches_surface_0p20": np.asarray(component_touches_surface_0p20),
        "damage_event_time_0p05": np.asarray([damage_event_times[0.05] or np.nan] * len(times_h)),
        "damage_event_time_0p10": np.asarray([damage_event_times[0.10] or np.nan] * len(times_h)),
        "damage_event_time_0p20": np.asarray([damage_event_times[0.20] or np.nan] * len(times_h)),
        "damage_event_time_0p35": np.asarray([damage_event_times[0.35] or np.nan] * len(times_h)),
        "spanning_event_time_0p05": np.asarray([spanning_event_times[0.05] or np.nan] * len(times_h)),
        "spanning_event_time_0p20": np.asarray([spanning_event_times[0.20] or np.nan] * len(times_h)),
        "opening_event_time_0p005mm": np.asarray([opening_event_times[0.005] or np.nan] * len(times_h)),
        "opening_event_time_0p01mm": np.asarray([opening_event_times[0.01] or np.nan] * len(times_h)),
        "opening_event_time_0p05mm": np.asarray([opening_event_times[0.05] or np.nan] * len(times_h)),
        "pressure_cap_trigger_total": np.asarray([cap_trigger_total]),
        "pressure_overflow_volume_total_m3": np.asarray([overflow_volume_total]),
    }
    summary = SolverSummary(
        0.0,
        0.0,
        4.0,
        float(np.max(sim.electric_grad)),
        float(np.max(sim.i_corr)),
        float(np.max(sim.rust_pressure) / 1.0e6),
        float(np.max(sim.d)),
    )
    return sim, histories, summary
