"""Run the conditional accelerated localization study on a compact 3-D domain."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from corrosion3d.solver import (
    phasefield_gc_from_tensile_strength,
    run_accelerated_corrosion_3d,
)


def _last_or_none(values):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite[-1]) if finite.size else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-mm", type=float, default=5.0)
    parser.add_argument("--total-hours", type=float, default=1000.0)
    parser.add_argument("--dt-hours", type=float, default=25.0)
    parser.add_argument("--pressure-scale", type=float, default=2.5)
    parser.add_argument("--leakage-rate", type=float, default=500.0)
    parser.add_argument("--post-event-hours", type=float, default=100.0)
    parser.add_argument("--continue-after-event", action="store_true")
    parser.add_argument("--localized-flaw-factor", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[880])
    parser.add_argument(
        "--interface-factors", type=float, nargs="+", default=[1.0, 0.7, 0.5]
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tensile_strength_mpa = 3.79
    elastic_modulus_pa = max(18.0e9, 22.0e9 + 2.0e9 * tensile_strength_mpa / 3.0)
    length_scale_m = 0.005
    fracture_energy = phasefield_gc_from_tensile_strength(
        tensile_strength_mpa * 1.0e6, elastic_modulus_pa, length_scale_m
    )
    rows = []
    for seed in args.seeds:
        for factor in args.interface_factors:
            case_id = f"accelerated_h{args.grid_mm:g}_seed{seed}_interface{factor:.2f}"
            started = time.perf_counter()
            sim, histories, summary = run_accelerated_corrosion_3d(
                case_id=case_id,
                cover_depth_m=0.020,
                rebar_radius_m=0.0135,
                tensile_strength_mpa=tensile_strength_mpa,
                corrosion_current_ua_cm2=100.0,
                total_hours=args.total_hours,
                dt_hours=args.dt_hours,
                seed=seed,
                grid_spacing=args.grid_mm / 1000.0,
                mechanics_iterations=120,
                leakage_pressure_scale=args.pressure_scale,
                leakage_law="pressure_dependent",
                leakage_rate=args.leakage_rate,
                leakage_exponent=1.0,
                fracture_energy_n_m=fracture_energy,
                interface_flaw_factor=args.localized_flaw_factor,
                steel_interface_fracture_factor=factor,
                steel_interface_stiffness_factor=factor,
                phasefield_length_m=length_scale_m,
                steel_interface_width_m=0.0025,
                surface_observation_depth_m=0.010,
                domain_width_m=0.080,
                domain_height_m=0.080,
                domain_thickness_m=0.020,
                itz_width_m=0.003,
                surface_gauge_length_m=0.020,
                axial_symmetry_ends=True,
                stop_after_spanning_threshold=(
                    None if args.continue_after_event else 0.20
                ),
                post_event_hours=args.post_event_hours,
                progress=True,
            )
            runtime_s = time.perf_counter() - started
            case_dir = output / case_id
            case_dir.mkdir(exist_ok=True)
            n = len(histories["times_h"])
            pd.DataFrame(
                {key: value for key, value in histories.items() if len(value) == n}
            ).to_csv(case_dir / "history.csv", index=False)
            np.savez_compressed(
                case_dir / "fields.npz",
                damage=sim.d,
                rust_pressure=sim.rust_pressure,
                rust_generated=sim.rust_generated,
                rust_retained=sim.rust_volume,
                rust_lost=sim.rust_lost,
                displacement_u=sim.u,
                displacement_v=sim.v,
                displacement_w=sim.w,
                rebar_interface=sim.rebar_bnd,
                aggregates=sim.is_agg,
                itz_fraction=sim.itz_fraction,
            )
            span05 = sim.diagnostic_spanning_damage_component(0.05)
            span20 = sim.diagnostic_spanning_damage_component(0.20)
            row = {
                "case_id": case_id,
                "grid_mm": args.grid_mm,
                "seed": seed,
                "interface_factor": factor,
                "fracture_energy_n_m": fracture_energy,
                "runtime_s": runtime_s,
                "completed_hours": _last_or_none(histories["times_h"]),
                "peak_pressure_mpa": float(np.max(histories["max_pressure_mpa"])),
                "max_damage": summary.max_damage,
                "spanning_d0p05": bool(span05[2]),
                "spanning_d0p05_volume_m3": span05[1],
                "spanning_d0p20": bool(span20[2]),
                "spanning_d0p20_volume_m3": span20[1],
                "spanning_d0p05_year": _last_or_none(
                    histories["spanning_event_time_0p05"]
                ),
                "spanning_d0p20_year": _last_or_none(
                    histories["spanning_event_time_0p20"]
                ),
                "opening_envelope_mm": float(
                    np.max(histories["surface_crack_width_envelope_mm"])
                ),
                "opening_0p05mm_year": _last_or_none(
                    histories["opening_event_time_0p05mm"]
                ),
                "max_mechanics_residual": float(
                    np.max(histories["mechanics_relative_residual"])
                ),
                "max_coupling_residual": float(
                    np.max(histories["coupling_pressure_relative_residual"])
                ),
                "max_phasefield_residual": float(
                    np.max(histories["phasefield_relative_residual"])
                ),
                "lost_rust_fraction": _last_or_none(histories["rust_lost_fraction"]),
            }
            rows.append(row)
            (case_dir / "summary.json").write_text(
                json.dumps(row, indent=2, allow_nan=False), encoding="utf-8"
            )
    pd.DataFrame(rows).to_csv(output / "case_summary.csv", index=False)
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "purpose": "Conditional accelerated localization study",
                "accelerated_current_ua_cm2": 100.0,
                "natural_service_life_interpretation": False,
                "tensile_strength_mpa": tensile_strength_mpa,
                "elastic_modulus_pa": elastic_modulus_pa,
                "phasefield_length_m": length_scale_m,
                "fracture_energy_n_m": fracture_energy,
                "leakage_law": "pressure-dependent finite-rate leakage law",
                "pressure_scale": args.pressure_scale,
                "leakage_rate_per_year": args.leakage_rate,
                "connectivity": "6-neighbour face connectivity to the true top surface",
                "domain_m": [0.080, 0.080, 0.020],
                "axial_boundary": "zero axial displacement and phase-field drive on symmetry end planes",
                "steel_interface_width_m": 0.0025,
                "itz_width_m": 0.003,
                "surface_gauge_length_m": 0.020,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
