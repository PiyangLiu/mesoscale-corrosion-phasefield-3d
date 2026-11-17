"""Reproduce the article's service-exposure studies with the coupled solver.

Each command writes a complete result set to the selected output directory.
The generated metadata records source hashes and configuration data alongside
the numerical histories and fields.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from corrosion3d.cli import save_results
from corrosion3d.solver import EngineeringParameters, run_full_3d


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src" / "corrosion3d"
DEFAULT_OUTPUT = ROOT / "outputs" / "service_exposure_studies"
GEOMETRY = ROOT / "data" / "mesh_seed88_cover50mm" / "continuous_geometry_seed88_cover50mm.json"

BASELINE = EngineeringParameters(
    case_id="paper_baseline_3d",
    description="residual-controlled marine exposure baseline",
    cover_depth=0.05,
    rebar_radius=0.01,
    surface_chloride=4.0,
    critical_chloride=1.2,
    mortar_chloride_diffusivity=1.0e-4,
    itz_diffusivity_multiplier=5.0,
    reference_corrosion_current=0.01,
    rust_expansion_ratio=2.5,
    porous_gap=8.0e-6,
    crack_storage_coeff=1.2e-3,
    rust_bulk_modulus=2.0e9,
    fracture_energy=80.0,
    phasefield_length=0.005,
    residual_stiffness=1.0e-4,
    fracture_heterogeneity_cv=0.10,
    fracture_heterogeneity_length=0.01,
    aggregate_fraction=0.30,
    seed=88,
    grid_spacing=0.0025,
    domain_width=0.12,
    domain_height=0.20,
    domain_thickness=0.04,
    itz_width=0.003,
    steel_interface_width=0.0025,
    surface_observation_depth=0.01,
    surface_gauge_length=0.02,
    mechanics_rtol=1.0e-6,
    mechanics_max_iterations=1200,
    transport_solver="iterative",
    transport_rtol=1.0e-10,
    transport_max_iterations=1000,
    pressure_limit_scale=1.0,
    pressure_limit_damage=0.5,
    axial_transport_enabled=True,
    axial_symmetry_ends=True,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    files = [
        ROOT / name
        for name in (
            "paper_studies.py",
            "paper_figures.py",
            "service_exposure_horizon.py",
        )
    ] + [SOURCE / name for name in ("core.py", "simulation.py", "solver.py", "cli.py")]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}


def first_time(times, values, threshold):
    hit = np.flatnonzero(np.asarray(values) >= threshold)
    return None if hit.size == 0 else float(np.asarray(times)[hit[0]])


def summary_row(case_id, params, histories, summary):
    generated = np.maximum(np.abs(histories["rust_generated_m"]), 1.0e-30)
    return {
        "case_id": case_id,
        "depassivation_year": summary.depassivation_year_min,
        "damage_0p20_year": first_time(histories["times"], histories["max_damage"], 0.20),
        "surface_width_0p05_year": first_time(
            histories["times"], histories["surface_crack_width_envelope_mm"], 0.05
        ),
        "peak_pressure_mpa": float(np.max(histories["max_pressure_mpa"])),
        "final_max_damage": float(histories["max_damage"][-1]),
        "final_surface_width_mm": float(histories["surface_crack_width_envelope_mm"][-1]),
        "final_elastic_energy_j": float(histories["elastic_energy_j"][-1]),
        "final_fracture_energy_j": float(histories["fracture_energy_j"][-1]),
        "final_pressure_work_j": float(histories["pressure_work_j"][-1]),
        "maximum_mechanical_energy_relative_error": float(
            np.max(histories["mechanical_energy_relative_error"])
        ),
        "maximum_mechanics_relative_residual": float(
            np.max(histories["mechanics_relative_residual"])
        ),
        "maximum_phasefield_relative_residual": float(
            np.max(histories["phasefield_relative_residual"])
        ),
        "maximum_pressure_coupling_relative_residual": float(
            np.max(histories["coupling_pressure_relative_residual"])
        ),
        "maximum_transport_relative_residual": float(
            np.max(histories["transport_relative_residual"])
        ),
        "maximum_relative_mass_balance_error": float(
            np.max(np.abs(histories["rust_mass_balance_error_m"]) / generated)
        ),
        "grid_spacing_mm": 1000.0 * float(params.grid_spacing),
        "dt_year": float(histories["times"][0]),
        "parameters": json.dumps(asdict(params), sort_keys=True),
    }


def run_case(output_root: Path, params, total_years, dt, group, *, stop_at_depassivation=False):
    case_dir = output_root / group / params.case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "simulation": {
            "total_years": total_years,
            "dt_years": dt,
            "staggered_max_iterations": 600,
        },
        "parameters": asdict(params),
    }
    config_path = case_dir / "configuration.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    termination = None
    if stop_at_depassivation:
        termination = lambda sim, _time: bool(np.any(np.isfinite(sim.t_ini_map)))
    simulation, histories, summary = run_full_3d(
        total_years=total_years,
        dt=dt,
        params=params,
        staggered_max_iterations=600,
        progress=True,
        termination_condition=termination,
    )
    save_results(case_dir, config_path, params, simulation, histories, summary)
    return simulation, histories, summary


def run_baseline(output_root: Path):
    sim, histories, summary = run_case(output_root, BASELINE, 22.0, 0.5, "baseline")
    pd.DataFrame([summary_row(BASELINE.case_id, BASELINE, histories, summary)]).to_csv(
        output_root / "baseline_summary.csv", index=False
    )


def run_time_steps(output_root: Path):
    rows = []
    baseline_summary = output_root / "baseline_summary.csv"
    if baseline_summary.exists():
        baseline_row = pd.read_csv(baseline_summary).iloc[0].to_dict()
        baseline_row["case_id"] = "time_step_0p5"
        rows.append(baseline_row)
        source_case = output_root / "baseline" / "paper_baseline_3d"
        target_case = output_root / "time_step" / "time_step_0p5"
        if source_case.exists() and not target_case.exists():
            target_case.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_case, target_case)
    for dt in (1.0, 0.25):
        params = replace(BASELINE, case_id=f"time_step_{str(dt).replace('.', 'p')}")
        _, histories, summary = run_case(output_root, params, 22.0, dt, "time_step")
        rows.append(summary_row(params.case_id, params, histories, summary))
        pd.DataFrame(rows).to_csv(output_root / "time_step_summary.csv", index=False)


def run_parameters(output_root: Path):
    coarse = replace(BASELINE, grid_spacing=0.005, transport_solver="direct")
    cases = [
        replace(coarse, case_id="parameter_baseline"),
        replace(coarse, case_id="cover_40mm", cover_depth=0.04),
        replace(coarse, case_id="cover_60mm", cover_depth=0.06),
        replace(coarse, case_id="chloride_2p5", surface_chloride=2.5),
        replace(coarse, case_id="chloride_5p5", surface_chloride=5.5),
        replace(coarse, case_id="diffusivity_0p5", mortar_chloride_diffusivity=5.0e-5),
        replace(coarse, case_id="diffusivity_2p0", mortar_chloride_diffusivity=2.0e-4),
        replace(coarse, case_id="current_0p003", reference_corrosion_current=0.003),
        replace(coarse, case_id="current_0p020", reference_corrosion_current=0.020),
        replace(coarse, case_id="storage_4um", porous_gap=4.0e-6),
        replace(coarse, case_id="storage_25um", porous_gap=25.0e-6),
        replace(coarse, case_id="expansion_2p0", rust_expansion_ratio=2.0),
        replace(coarse, case_id="expansion_3p5", rust_expansion_ratio=3.5),
        replace(coarse, case_id="fracture_160", fracture_energy=160.0),
        replace(coarse, case_id="fracture_240", fracture_energy=240.0),
    ]
    rows = []
    for params in cases:
        _, histories, summary = run_case(output_root, params, 35.0, 0.5, "parameters")
        rows.append(summary_row(params.case_id, params, histories, summary))
        pd.DataFrame(rows).to_csv(output_root / "parameter_summary.csv", index=False)


def run_axial_control(output_root: Path):
    rows = []
    for seed in range(88, 93):
        common = replace(
            BASELINE,
            seed=seed,
            grid_spacing=0.0025,
            domain_width=0.08,
            domain_height=0.12,
            domain_thickness=0.04,
            transport_solver="direct",
        )
        for enabled in (True, False):
            label = "enabled" if enabled else "disabled"
            params = replace(
                common,
                case_id=f"axial_{label}_seed{seed}",
                axial_transport_enabled=enabled,
            )
            _, histories, summary = run_case(
                output_root,
                params,
                25.0,
                0.25,
                "axial_control",
                stop_at_depassivation=True,
            )
            rows.append(
                {
                    "case_id": params.case_id,
                    "seed": seed,
                    "axial_transport": label,
                    "depassivation_year": summary.depassivation_year_min,
                    "end_year": float(histories["times"][-1]),
                }
            )
            pd.DataFrame(rows).to_csv(output_root / "axial_control_summary.csv", index=False)


def run_mesh(output_root: Path):
    rows = []
    for h_mm in (5.0, 2.5, 1.25):
        params = replace(
            BASELINE,
            case_id=f"mesh_h_{str(h_mm).replace('.', 'p')}mm",
            grid_spacing=h_mm / 1000.0,
            domain_width=0.08,
            domain_height=0.12,
            domain_thickness=0.04,
            aggregate_records_path=str(GEOMETRY),
            transport_solver="iterative" if h_mm <= 2.5 else "direct",
        )
        sim, histories, summary = run_case(output_root, params, 35.0, 0.5, "mesh")
        row = summary_row(params.case_id, params, histories, summary)
        row.update(
            actual_aggregate_fraction=float(sim.actual_aggregate_fraction),
            actual_itz_fraction=float(sim.actual_itz_fraction),
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_root / "mesh_summary.csv", index=False)


def write_receipt(output_root: Path, commands):
    receipt = {
        "source_hashes": source_hashes(),
        "commands": commands,
        "result_root": str(output_root.resolve()),
        "depassivation_clock": "end of first accepted transport step satisfying C >= C_critical",
        "primary_solver": "corrosion3d.solver.run_full_3d",
    }
    (output_root / "RUN_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )


def make_figures(output_root: Path):
    figure_dir = output_root / "figures"
    figure_dir.mkdir(exist_ok=True)
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0})

    if (output_root / "axial_control_summary.csv").exists():
        frame = pd.read_csv(output_root / "axial_control_summary.csv")
        pivot = frame.pivot(index="seed", columns="axial_transport", values="depassivation_year")
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)
        x = np.arange(len(pivot))
        axes[0].plot(x, pivot["enabled"], "o-", label="Axial transport enabled")
        axes[0].plot(x, pivot["disabled"], "s--", label="Axial transport disabled")
        axes[0].set(xticks=x, xticklabels=pivot.index, xlabel="Mesostructure seed", ylabel="First-local-depassivation indicator (yr)")
        axes[0].legend(frameon=False)
        delta = pivot["disabled"] - pivot["enabled"]
        axes[1].bar(x, delta, color="#0072B2")
        axes[1].axhline(0.0, color="black", lw=0.8)
        axes[1].set(xticks=x, xticklabels=pivot.index, xlabel="Mesostructure seed", ylabel="Disabled minus enabled (yr)")
        fig.savefig(figure_dir / "Figure_10_axial.png", dpi=400)
        plt.close(fig)

    if (output_root / "time_step_summary.csv").exists() and (output_root / "mesh_summary.csv").exists():
        dt = pd.read_csv(output_root / "time_step_summary.csv").sort_values("dt_year")
        mesh = pd.read_csv(output_root / "mesh_summary.csv").sort_values("grid_spacing_mm")
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), constrained_layout=True)
        axes[0, 0].plot(dt["dt_year"], dt["depassivation_year"], "o-")
        axes[0, 0].set(xlabel="Time step (yr)", ylabel="First-local-depassivation indicator (yr)")
        axes[0, 1].plot(dt["dt_year"], dt["maximum_mechanical_energy_relative_error"], "s-")
        axes[0, 1].set(xlabel="Time step (yr)", ylabel="Maximum |Wp - 2Eel| / max(Wp, 2Eel)")
        axes[1, 0].plot(mesh["grid_spacing_mm"], mesh["depassivation_year"], "o-")
        axes[1, 0].set(xlabel="Grid spacing (mm)", ylabel="First-local-depassivation indicator (yr)")
        axes[1, 1].plot(mesh["grid_spacing_mm"], mesh["final_max_damage"], "s-")
        axes[1, 1].axhline(0.20, color="#666666", ls=":")
        axes[1, 1].set(xlabel="Grid spacing (mm)", ylabel="Final maximum damage")
        fig.savefig(figure_dir / "Figure_16.png", dpi=400)
        plt.close(fig)


def main():
    parser = ArgumentParser()
    parser.add_argument("commands", nargs="+", choices=("baseline", "time-step", "parameters", "axial", "mesh", "figures"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "baseline": run_baseline,
        "time-step": run_time_steps,
        "parameters": run_parameters,
        "axial": run_axial_control,
        "mesh": run_mesh,
        "figures": make_figures,
    }
    for command in args.commands:
        dispatch[command](output)
    write_receipt(output, args.commands)


if __name__ == "__main__":
    main()
