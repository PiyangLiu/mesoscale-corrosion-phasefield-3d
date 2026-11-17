"""Run the eight Mullard-Stewart Series SV accelerated-current cases."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from corrosion3d.solver import run_accelerated_corrosion_3d


CASES = [
    ("SV_I_2i", 10.0, 16.0),
    ("SV_I_1i", 20.0, 16.0),
    ("SV_II_1i", 30.0, 16.0),
    ("SV_II_2i", 40.0, 16.0),
    ("SV_III_1i", 10.0, 27.0),
    ("SV_III_2i", 20.0, 27.0),
    ("SV_IV_2i", 30.0, 27.0),
    ("SV_IV_1i", 40.0, 27.0),
]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-spacing-mm", type=float, default=5.0)
    parser.add_argument("--total-hours", type=float, default=3500.0)
    parser.add_argument("--dt-hours", type=float, default=25.0)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (case_id, cover_mm, diameter_mm) in enumerate(CASES):
        sim, histories, summary = run_accelerated_corrosion_3d(
            case_id=case_id,
            cover_depth_m=cover_mm / 1000.0,
            rebar_radius_m=diameter_mm / 2000.0,
            tensile_strength_mpa=3.79,
            corrosion_current_ua_cm2=100.0,
            total_hours=args.total_hours,
            dt_hours=args.dt_hours,
            seed=610 + index,
            grid_spacing=args.grid_spacing_mm / 1000.0,
            mechanics_iterations=60,
            progress=True,
        )
        case_dir = out / case_id
        case_dir.mkdir(exist_ok=True)
        pd.DataFrame(histories).to_csv(case_dir / "history.csv", index=False)
        rows.append(
            {
                "case_id": case_id,
                "cover_mm": cover_mm,
                "bar_diameter_mm": diameter_mm,
                "grid_spacing_mm": args.grid_spacing_mm,
                "total_hours": args.total_hours,
                "dt_hours": args.dt_hours,
                "peak_pressure_mpa": summary.max_rust_pressure_mpa,
                "final_max_damage": summary.max_damage,
                "final_surface_width_mm": float(histories["surface_crack_width_envelope_mm"][-1]),
                "maximum_mechanics_residual": float(max(histories["mechanics_relative_residual"])),
                "maximum_pressure_coupling_residual": float(max(histories["coupling_pressure_relative_residual"])),
                "maximum_phasefield_residual": float(max(histories["phasefield_relative_residual"])),
                "maximum_mass_balance_error_m": float(max(abs(histories["rust_mass_balance_error_m"]))),
            }
        )
    pd.DataFrame(rows).to_csv(out / "case_summary.csv", index=False)
    (out / "study_metadata.json").write_text(
        json.dumps(
            {
                "solver": "corrosion3d.solver.run_accelerated_corrosion_3d",
                "grid_spacing_mm": args.grid_spacing_mm,
                "total_hours": args.total_hours,
                "dt_hours": args.dt_hours,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
