"""Run an extended service-exposure mechanical-event calculation.

The calculation distinguishes the earliest-cell chloride crossing from the
mechanical event indicators. It advances the residual-controlled solver until
localization, the surface-opening threshold, or the requested time horizon.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from corrosion3d.cli import load_configuration, save_results
from corrosion3d.solver import run_full_3d


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-years", type=float, default=None)
    parser.add_argument("--dt-years", type=float, default=None)
    parser.add_argument("--damage-threshold", type=float, default=0.20)
    parser.add_argument("--surface-width-threshold-mm", type=float, default=0.05)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    params, controls = load_configuration(config_path)
    total_years = float(
        args.total_years
        if args.total_years is not None
        else controls.get("total_years", 100.0)
    )
    dt_years = float(
        args.dt_years if args.dt_years is not None else controls.get("dt_years", 0.5)
    )
    stop = {"reason": None}

    def termination_condition(sim, _time):
        if float(np.max(sim.d)) >= args.damage_threshold:
            stop["reason"] = "damage_threshold"
            return True
        if float(sim.surface_crack_width_envelope_mm) >= args.surface_width_threshold_mm:
            stop["reason"] = "surface_width_threshold"
            return True
        return False

    simulation, histories, summary = run_full_3d(
        total_years=total_years,
        dt=dt_years,
        params=params,
        staggered_max_iterations=int(controls.get("staggered_max_iterations", 60)),
        progress=args.progress,
        termination_condition=termination_condition,
    )
    output = args.output.resolve()
    save_results(output, config_path, params, simulation, histories, summary)

    report_path = output / "summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reached = stop["reason"] is not None
    report["event_horizon"] = {
        "damage_threshold": args.damage_threshold,
        "surface_width_threshold_mm": args.surface_width_threshold_mm,
        "requested_horizon_years": total_years,
        "time_step_years": dt_years,
        "termination": stop["reason"] or "horizon_exhausted",
        "threshold_reached": reached,
        "final_time_years": float(histories["times"][-1]),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
