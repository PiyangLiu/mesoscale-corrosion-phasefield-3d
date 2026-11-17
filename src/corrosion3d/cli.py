"""Command-line interface for one complete three-dimensional simulation."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, fields
import csv
import json
from pathlib import Path

import numpy as np

from .solver import EngineeringParameters, run_full_3d


def _first_time(
    times: np.ndarray, values: np.ndarray, threshold: float
) -> float | None:
    indices = np.flatnonzero(values >= threshold)
    return None if indices.size == 0 else float(times[indices[0]])


def load_configuration(path: Path) -> tuple[EngineeringParameters, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameter_data = dict(payload.get("parameters", {}))
    valid_names = {item.name for item in fields(EngineeringParameters)}
    unknown = sorted(set(parameter_data) - valid_names)
    if unknown:
        raise ValueError(f"Unknown engineering parameters: {unknown}")
    records_path = parameter_data.get("aggregate_records_path")
    if records_path:
        candidate = Path(records_path)
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        parameter_data["aggregate_records_path"] = str(candidate)
    return EngineeringParameters(**parameter_data), dict(payload.get("simulation", {}))


def save_results(
    output: Path, configuration: Path, params, simulation, histories, summary
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    times = histories["times"]
    columns = [
        name
        for name, values in histories.items()
        if np.asarray(values).shape == times.shape
    ]
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for index in range(times.size):
            writer.writerow(
                [float(np.asarray(histories[name])[index]) for name in columns]
            )
    np.savez_compressed(
        output / "fields.npz",
        X=simulation.X,
        Y=simulation.Y,
        Z=simulation.Z,
        chloride=simulation.C,
        saturation=simulation.S,
        sodium=simulation.Na,
        hydroxyl=simulation.OH,
        electric_gradient=simulation.electric_grad,
        corrosion_current=simulation.i_corr,
        rust_pressure=simulation.rust_pressure,
        rust_generated=simulation.rust_generated,
        rust_retained=simulation.rust_volume,
        rust_filled=simulation.rust_filled,
        rust_lost=simulation.rust_lost,
        damage=simulation.d,
        displacement_u=simulation.u,
        displacement_v=simulation.v,
        displacement_w=simulation.w,
        aggregate=simulation.is_agg,
        itz_fraction=simulation.itz_fraction,
        rebar=simulation.is_rebar,
        rebar_interface=simulation.rebar_bnd,
    )
    geometry = {
        "seed": params.seed,
        "domain_m": [simulation.W, simulation.H, simulation.D],
        "grid_shape": list(simulation.shape_3d),
        "actual_aggregate_fraction": simulation.actual_aggregate_fraction,
        "actual_itz_fraction": simulation.actual_itz_fraction,
        "records": simulation.agg_list,
    }
    (output / "aggregate_geometry.json").write_text(
        json.dumps(geometry, indent=2), encoding="utf-8"
    )
    generated = np.maximum(np.abs(histories["rust_generated_m"]), 1.0e-30)
    report = {
        "configuration": configuration.name,
        "parameters": asdict(params),
        "elapsed_s": summary.elapsed_s,
        "depassivation_year": summary.depassivation_year_min,
        "damage_0p20_year": _first_time(
            histories["times"], histories["max_damage"], 0.20
        ),
        "surface_width_0p05_year": _first_time(
            histories["times"], histories["surface_crack_width_envelope_mm"], 0.05
        ),
        "peak_pressure_mpa": float(np.max(histories["max_pressure_mpa"])),
        "final_max_damage": float(histories["max_damage"][-1]),
        "final_surface_width_mm": float(
            histories["surface_crack_width_envelope_mm"][-1]
        ),
        "pressure_cap_trigger_total": int(np.sum(histories["pressure_cap_trigger_count"])),
        "pressure_overflow_volume_total_m3": float(
            np.sum(histories["pressure_overflow_volume_m3"])
        ),
        "retained_rust_final_m3": float(histories["rust_retained_volume_m3"][-1]),
        "lost_rust_final_m3": float(histories["rust_lost_volume_m3"][-1]),
        "lost_rust_fraction_final": float(histories["rust_lost_fraction"][-1]),
        "max_tensile_strain": float(np.max(histories["max_tensile_strain"])),
        "max_phasefield_driving_energy": float(
            np.max(histories["max_phasefield_driving_energy"])
        ),
        "max_trial_pressure_mpa": float(np.max(histories["trial_pressure_max_mpa"])),
        "damage_event_times": {
            "d_0.05": _first_time(histories["times"], histories["max_damage"], 0.05),
            "d_0.10": _first_time(histories["times"], histories["max_damage"], 0.10),
            "d_0.20": _first_time(histories["times"], histories["max_damage"], 0.20),
            "d_0.35": _first_time(histories["times"], histories["max_damage"], 0.35),
        },
        "opening_event_times_mm": {
            "0.005": _first_time(
                histories["times"], histories["surface_crack_width_displacement_mm"], 0.005
            ),
            "0.01": _first_time(
                histories["times"], histories["surface_crack_width_displacement_mm"], 0.01
            ),
            "0.05": _first_time(
                histories["times"], histories["surface_crack_width_displacement_mm"], 0.05
            ),
        },
        "maximum_mechanics_relative_residual": float(
            np.max(histories["mechanics_relative_residual"])
        ),
        "maximum_coupling_pressure_relative_residual": float(
            np.max(histories["coupling_pressure_relative_residual"])
        ),
        "maximum_transport_relative_residual": float(
            np.max(histories["transport_relative_residual"])
        ),
        "maximum_mechanical_energy_relative_error": float(
            np.max(histories["mechanical_energy_relative_error"])
        ),
        "maximum_staggered_iterations": int(np.max(histories["staggered_iterations"])),
        "maximum_relative_mass_balance_error": float(
            np.max(np.abs(histories["rust_mass_balance_error_m"]) / generated)
        ),
        "grid_shape": list(simulation.shape_3d),
        "actual_aggregate_fraction": simulation.actual_aggregate_fraction,
        "actual_itz_fraction": simulation.actual_itz_fraction,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(
        description="Run the coupled three-dimensional corrosion-cracking model."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    params, controls = load_configuration(config_path)
    simulation, histories, summary = run_full_3d(
        total_years=float(controls.get("total_years", 3.0)),
        dt=float(controls.get("dt_years", 1.0)),
        params=params,
        staggered_max_iterations=int(
            controls.get("staggered_max_iterations", 60)
        ),
        progress=args.progress,
    )
    save_results(
        args.output.resolve(), config_path, params, simulation, histories, summary
    )
    print((args.output.resolve() / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
