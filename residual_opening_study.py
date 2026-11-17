"""Run a corrosion-rust-wedge residual-opening calculation."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from corrosion3d.solver import phasefield_gc_from_tensile_strength, run_accelerated_corrosion_3d


def scalar_event(history, name):
    values = np.asarray(history[name], dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else None


def opening_summary(history, opening_event):
    """Keep instantaneous opening distinct from its irreversible envelope."""
    current = np.asarray(history["surface_crack_width_mm"], dtype=float)
    envelope = np.asarray(
        history.get("surface_crack_width_envelope_mm", np.maximum.accumulate(current)),
        dtype=float,
    )
    final_current = float(current[-1])
    final_envelope = float(envelope[-1])
    return {
        "peak_opening_mm": float(np.max(current)),
        "final_opening_mm": final_current,
        "final_current_opening_mm": final_current,
        "opening_envelope_mm": final_envelope,
        "unloaded_residual_opening_mm": (
            final_current if opening_event is not None else None
        ),
        "unloading_closure_mm": (
            max(final_envelope - final_current, 0.0)
            if opening_event is not None
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", default="residual_opening_case")
    parser.add_argument("--grid-mm", type=float, default=5.0)
    parser.add_argument("--dt-hours", type=float, default=5.0)
    parser.add_argument("--total-hours", type=float, default=500.0)
    parser.add_argument("--post-opening-hours", type=float, default=50.0)
    parser.add_argument("--opening-threshold-mm", type=float, default=0.05)
    parser.add_argument(
        "--unload-trigger-mm",
        type=float,
        default=None,
        help="Loading opening that triggers current interruption; defaults to the opening threshold.",
    )
    parser.add_argument("--seed", type=int, default=880)
    parser.add_argument("--rust-expansion", type=float, default=4.4)
    parser.add_argument("--compacted-fraction", type=float, default=0.9)
    parser.add_argument("--wedge-fraction", type=float, default=1.0)
    parser.add_argument("--contact-stiffness", type=float, default=4.0e11)
    parser.add_argument("--leakage-permeability", type=float, default=5.0e-4)
    parser.add_argument("--damage-coupling-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--adaptive-time-stepping", action="store_true")
    parser.add_argument("--minimum-dt-hours", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--residual-stiffness", type=float, default=1.0e-5)
    parser.add_argument("--angular-concentration", type=float, default=0.0)
    parser.add_argument(
        "--corrosion-profile-mode",
        choices=("uniform_von_mises", "xu2024_fig8"),
        default="uniform_von_mises",
    )
    args = parser.parse_args()
    unload_trigger_mm = (
        args.opening_threshold_mm
        if args.unload_trigger_mm is None
        else args.unload_trigger_mm
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ft_mpa = 3.79
    e_pa = max(18.0e9, 22.0e9 + 2.0e9 * ft_mpa / 3.0)
    ell_m = 0.005
    gc = phasefield_gc_from_tensile_strength(ft_mpa * 1.0e6, e_pa, ell_m)
    started = time.perf_counter()
    try:
        sim, history, summary = run_accelerated_corrosion_3d(
            case_id=args.case_id,
            cover_depth_m=0.020,
            rebar_radius_m=0.0135,
            tensile_strength_mpa=ft_mpa,
            corrosion_current_ua_cm2=100.0,
            total_hours=args.total_hours,
            dt_hours=args.dt_hours,
            seed=args.seed,
            grid_spacing=args.grid_mm / 1000.0,
            mechanics_iterations=240,
            rust_expansion_ratio=args.rust_expansion,
            leakage_pressure_scale=1.1,
            leakage_law="unilateral_wedge",
            leakage_permeability_m_per_year=args.leakage_permeability,
            wedge_compacted_fraction=args.compacted_fraction,
            wedge_conversion_fraction=args.wedge_fraction,
            wedge_contact_stiffness_pa_per_m=args.contact_stiffness,
            damage_coupling_tolerance=args.damage_coupling_tolerance,
            fracture_energy_n_m=gc,
            phasefield_length_m=ell_m,
            residual_stiffness=args.residual_stiffness,
            corrosion_angular_concentration=args.angular_concentration,
            corrosion_profile_mode=args.corrosion_profile_mode,
            steel_interface_width_m=0.0025,
            surface_observation_depth_m=0.010,
            domain_width_m=0.080,
            domain_height_m=0.080,
            domain_thickness_m=0.020,
            itz_width_m=0.003,
            surface_gauge_length_m=0.020,
            stop_after_opening_threshold_mm=unload_trigger_mm,
            unload_after_opening_event=True,
            post_event_hours=args.post_opening_hours,
            progress=True,
            progress_interval_steps=1,
            adaptive_time_stepping=args.adaptive_time_stepping,
            minimum_dt_hours=args.minimum_dt_hours,
            checkpoint_path=str(output / "restart_state.pkl"),
            resume_from_checkpoint=args.resume,
        )
    except Exception as error:
        failure = {
            "case_id": args.case_id,
            "status": "failed",
            "runtime_s": time.perf_counter() - started,
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "grid_mm": args.grid_mm,
            "dt_hours": args.dt_hours,
            "seed": args.seed,
        }
        (output / "failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise
    elapsed = time.perf_counter() - started
    n = len(history["times_h"])
    pd.DataFrame({key: value for key, value in history.items() if len(value) == n}).to_csv(
        output / "history.csv", index=False
    )
    np.savez_compressed(
        output / "fields.npz",
        damage=sim.d,
        rust_pressure=sim.rust_pressure,
        rust_generated=sim.rust_generated,
        rust_mobile=sim.rust_mobile,
        rust_compacted=sim.rust_compacted,
        rust_retained=sim.rust_volume,
        rust_lost=sim.rust_lost,
        residual_wedge=sim.residual_wedge,
        displacement_u=sim.u,
        displacement_v=sim.v,
        displacement_w=sim.w,
        rebar_interface=sim.rebar_bnd,
        aggregates=sim.is_agg,
        itz_fraction=sim.itz_fraction,
    )
    event_names = {
        0.005: "opening_event_time_0p005mm",
        0.01: "opening_event_time_0p01mm",
        0.05: "opening_event_time_0p05mm",
    }
    event_name = event_names.get(round(args.opening_threshold_mm, 6))
    opening_event = scalar_event(history, event_name) if event_name else None
    width_summary = opening_summary(history, opening_event)
    summary_payload = {
        "case_id": args.case_id,
        "status": (
            "opening_threshold_reached" if opening_event is not None else "right_censored"
        ),
        "grid_mm": args.grid_mm,
        "dt_hours": args.dt_hours,
        "seed": args.seed,
        "runtime_s": elapsed,
        "completed_hours": float(history["times_h"][-1]),
        "opening_event_year": opening_event,
        "opening_threshold_mm": args.opening_threshold_mm,
        "unload_trigger_mm": unload_trigger_mm,
        **width_summary,
        "residual_target_met": bool(
            width_summary["unloaded_residual_opening_mm"] is not None
            and width_summary["unloaded_residual_opening_mm"]
            >= args.opening_threshold_mm
        ),
        "peak_pressure_mpa": float(np.max(history["max_pressure_mpa"])),
        "final_pressure_mpa": float(history["max_pressure_mpa"][-1]),
        "max_damage": summary.max_damage,
        "spanning_d0p20": bool(sim.diagnostic_spanning_damage_component(0.20)[2]),
        "max_mechanics_residual": float(np.max(history["mechanics_relative_residual"])),
        "max_contact_residual": float(np.max(history["coupling_pressure_relative_residual"])),
        "min_contact_separation_m": float(
            np.min(history.get("contact_min_separation_m", np.asarray([0.0])))
        ),
        "max_contact_active_iterations": int(
            np.max(history.get("contact_active_iterations", np.asarray([0])))
        ),
        "max_contact_active_cells": int(
            np.max(history.get("contact_active_cells", np.asarray([0])))
        ),
        "max_contact_linear_iterations": int(
            np.max(history.get("contact_linear_iterations", np.asarray([0])))
        ),
        "pdas_fallback_step_count": int(
            np.count_nonzero(history.get("contact_pdas_fallback_used", np.asarray([0])))
        ),
        "max_phasefield_residual": float(np.max(history["phasefield_relative_residual"])),
        "max_mass_balance_error_m": float(np.max(np.abs(history["rust_mass_balance_error_m"]))),
        "rust_expansion_ratio": args.rust_expansion,
        "compacted_fraction": args.compacted_fraction,
        "wedge_conversion_fraction": args.wedge_fraction,
        "contact_stiffness_pa_per_m": args.contact_stiffness,
        "leakage_permeability_m_per_year": args.leakage_permeability,
        "damage_coupling_tolerance": args.damage_coupling_tolerance,
        "adaptive_time_stepping": args.adaptive_time_stepping,
        "minimum_dt_hours": args.minimum_dt_hours,
        "minimum_used_dt_hours": float(
            np.min(history.get("time_step_hours", np.asarray([args.dt_hours])))
        ),
        "max_damage_coupling_iterations": int(
            np.max(history.get("damage_coupling_iterations", np.asarray([0])))
        ),
        "residual_stiffness": args.residual_stiffness,
        "corrosion_angular_concentration": args.angular_concentration,
        "corrosion_profile_mode": args.corrosion_profile_mode,
        "fracture_energy_n_m": gc,
        "model_equation": "finite-rate mobile leakage; no instantaneous damage-dependent loss of new rust",
        "interpretation": "accelerated mechanism test; not a natural-service lifetime prediction",
    }
    (output / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
