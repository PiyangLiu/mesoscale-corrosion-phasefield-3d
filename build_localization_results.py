"""Assemble the grid-seed localization matrix and conditional event-time ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "early_localization"
OUT = RESULTS / "grid_seed_summary"


def read_table(relative: str, source_label: str) -> pd.DataFrame:
    path = RESULTS / relative / "case_summary.csv"
    frame = pd.read_csv(path)
    frame["source_result"] = str(path.relative_to(ROOT))
    frame["source_label"] = source_label
    return frame


def km_table(times_h: np.ndarray, observed: np.ndarray) -> pd.DataFrame:
    rows = [{"time_h": 0.0, "at_risk": len(times_h), "events": 0, "censored": 0, "survival": 1.0}]
    survival = 1.0
    for time_h in sorted(np.unique(times_h)):
        at_risk = int(np.count_nonzero(times_h >= time_h))
        events = int(np.count_nonzero((times_h == time_h) & observed))
        censored = int(np.count_nonzero((times_h == time_h) & ~observed))
        if events:
            survival *= 1.0 - events / at_risk
        rows.append(
            {
                "time_h": float(time_h),
                "at_risk": at_risk,
                "events": events,
                "censored": censored,
                "survival": survival,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h5 = read_table("grid_h5_seeds880_882", "h5")
    h5 = h5[np.isclose(h5.interface_factor, 1.0)].copy()
    grid = pd.concat(
        [
            h5,
            read_table("grid_h2p5_seeds880_882", "h2p5"),
            read_table("grid_h1p25_seeds880_882", "h1p25"),
        ],
        ignore_index=True,
    ).sort_values(["grid_mm", "seed"], ascending=[False, True])
    grid["spanning_d0p20_hours"] = grid["spanning_d0p20_year"] * 365.25 * 24.0
    grid["spanning_volume_fraction"] = grid["spanning_d0p20_volume_m3"] / (0.08 * 0.08 * 0.02)
    grid["opening_right_censored"] = grid["opening_0p05mm_year"].isna()
    grid.to_csv(OUT / "grid_seed_cases.csv", index=False)

    stats = (
        grid.groupby("grid_mm")
        .agg(
            seeds=("seed", "count"),
            event_time_mean_h=("spanning_d0p20_hours", "mean"),
            event_time_std_h=("spanning_d0p20_hours", "std"),
            spanning_volume_mean_m3=("spanning_d0p20_volume_m3", "mean"),
            spanning_volume_std_m3=("spanning_d0p20_volume_m3", "std"),
            spanning_volume_fraction_mean=("spanning_volume_fraction", "mean"),
            max_damage_mean=("max_damage", "mean"),
            max_damage_std=("max_damage", "std"),
            max_opening_mm=("opening_envelope_mm", "max"),
            max_mechanics_residual=("max_mechanics_residual", "max"),
            max_phasefield_residual=("max_phasefield_residual", "max"),
        )
        .reset_index()
        .sort_values("grid_mm", ascending=False)
    )
    stats.to_csv(OUT / "grid_statistics.csv", index=False)

    comparisons = []
    indexed = grid.set_index(["grid_mm", "seed"])
    for seed in (880, 881, 882):
        for coarse, fine in ((5.0, 2.5), (2.5, 1.25)):
            a = indexed.loc[(coarse, seed)]
            b = indexed.loc[(fine, seed)]
            comparisons.append(
                {
                    "seed": seed,
                    "coarse_grid_mm": coarse,
                    "fine_grid_mm": fine,
                    "event_time_difference_h": b["spanning_d0p20_hours"]
                    - a["spanning_d0p20_hours"],
                    "spanning_volume_relative_difference": abs(
                        b["spanning_d0p20_volume_m3"]
                        - a["spanning_d0p20_volume_m3"]
                    )
                    / max(abs(b["spanning_d0p20_volume_m3"]), 1.0e-30),
                    "opening_difference_mm": b["opening_envelope_mm"]
                    - a["opening_envelope_mm"],
                }
            )
    adjacent = pd.DataFrame(comparisons)
    adjacent.to_csv(OUT / "adjacent_grid_comparison.csv", index=False)

    ensemble_parts = []
    for scale, directory in (
        (1.0, "conditional_ensemble_scale1"),
        (1.1, "conditional_ensemble_scale1p1"),
        (2.5, "conditional_ensemble_scale2p5"),
        (5.0, "conditional_ensemble_scale5"),
    ):
        part = read_table(directory, f"pressure_scale_{scale:g}")
        part["pressure_scale"] = scale
        ensemble_parts.append(part)
    ensemble = pd.concat(ensemble_parts, ignore_index=True)
    ensemble["damage_event_observed"] = ensemble["spanning_d0p20_year"].notna()
    ensemble["damage_event_or_censor_h"] = np.where(
        ensemble["damage_event_observed"],
        ensemble["spanning_d0p20_year"] * 365.25 * 24.0,
        ensemble["completed_hours"],
    )
    ensemble["opening_event_observed"] = ensemble["opening_0p05mm_year"].notna()
    ensemble["opening_event_or_censor_h"] = np.where(
        ensemble["opening_event_observed"],
        ensemble["opening_0p05mm_year"] * 365.25 * 24.0,
        ensemble["completed_hours"],
    )
    ensemble.to_csv(OUT / "conditional_lifetime_ensemble.csv", index=False)
    damage_km = km_table(
        ensemble["damage_event_or_censor_h"].to_numpy(float),
        ensemble["damage_event_observed"].to_numpy(bool),
    )
    opening_km = km_table(
        ensemble["opening_event_or_censor_h"].to_numpy(float),
        ensemble["opening_event_observed"].to_numpy(bool),
    )
    damage_km.to_csv(OUT / "damage_event_survival.csv", index=False)
    opening_km.to_csv(OUT / "opening_event_survival.csv", index=False)

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for seed, marker in zip((880, 881, 882), ("o", "s", "^")):
        subset = grid[grid.seed == seed].sort_values("grid_mm")
        axes[0].plot(
            subset.grid_mm,
            subset.spanning_d0p20_hours,
            marker=marker,
            linewidth=1.0,
            label=f"seed {seed}",
        )
        axes[1].plot(
            subset.grid_mm,
            1.0e6 * subset.spanning_d0p20_volume_m3,
            marker=marker,
            linewidth=1.0,
        )
    axes[0].set(xlabel="Grid spacing (mm)", ylabel="Spanning-event time (h)")
    axes[1].set(xlabel="Grid spacing (mm)", ylabel=r"Spanning volume ($10^{-6}$ m$^3$)")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.invert_xaxis()
    fig.savefig(OUT / "grid_event_comparison.png", dpi=300)
    plt.close(fig)

    field_sources = [
        (5.0, RESULTS / "grid_h5_seeds880_882" / "accelerated_h5_seed880_interface1.00" / "fields.npz"),
        (2.5, RESULTS / "grid_h2p5_seeds880_882" / "accelerated_h2.5_seed880_interface1.00" / "fields.npz"),
        (1.25, RESULTS / "grid_h1p25_seeds880_882" / "accelerated_h1.25_seed880_interface1.00" / "fields.npz"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    image = None
    for axis, (grid_mm, path) in zip(axes, field_sources):
        with np.load(path) as fields:
            damage = fields["damage"]
        section = damage[:, :, damage.shape[2] // 2].T
        image = axis.imshow(
            section,
            origin="lower",
            extent=(0, 80, 0, 80),
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
            interpolation="nearest",
            aspect="equal",
        )
        axis.contour(
            np.linspace(0, 80, section.shape[1]),
            np.linspace(0, 80, section.shape[0]),
            section,
            levels=[0.20],
            colors=["cyan"],
            linewidths=0.8,
        )
        axis.set_title(f"h = {grid_mm:g} mm")
        axis.set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    fig.colorbar(image, ax=axes, label="Phase-field damage", shrink=0.82)
    fig.savefig(OUT / "damage_path_grid_comparison.png", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    axis.step(damage_km.time_h, damage_km.survival, where="post", label=r"$d\geq0.20$ spanning")
    axis.step(opening_km.time_h, opening_km.survival, where="post", label="0.05 mm opening")
    censored = ensemble.loc[~ensemble.damage_event_observed, "damage_event_or_censor_h"]
    axis.plot(censored, np.full(len(censored), damage_km.survival.iloc[-1]), "+", color="C0")
    axis.set(xlabel="Accelerated exposure time (h)", ylabel="Event-free fraction", ylim=(-0.02, 1.02))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(OUT / "conditional_event_survival.png", dpi=300)
    plt.close(fig)

    receipt = {
        "grid_seed_cases": int(len(grid)),
        "grid_spacings_mm": [5.0, 2.5, 1.25],
        "mesostructure_seeds": [880, 881, 882],
        "event_definition": "first 6-neighbour d>=0.20 component touching the steel interface and true exposed surface",
        "time_step_h": 1.5625,
        "event_time_refined_h": 25.78125,
        "event_time_refinement_steps_h": [1.5625, 0.78125, 0.390625],
        "event_time_refinement_relative_change": 0.0,
        "pressure_retention_scale": 1.1,
        "event_time_all_cases_h": sorted(grid.spanning_d0p20_hours.unique().tolist()),
        "maximum_adjacent_2p5_to_1p25_volume_relative_difference": float(
            adjacent.loc[adjacent.coarse_grid_mm == 2.5, "spanning_volume_relative_difference"].max()
        ),
        "maximum_mechanics_residual": float(grid.max_mechanics_residual.max()),
        "maximum_phasefield_residual": float(grid.max_phasefield_residual.max()),
        "opening_threshold_mm": 0.05,
        "grid_seed_opening_events": int((~grid.opening_right_censored).sum()),
        "lifetime_ensemble_definition": "equal-weight sensitivity ensemble over four pressure-retention scales and three mesostructure seeds",
        "lifetime_cases": int(len(ensemble)),
        "damage_events": int(ensemble.damage_event_observed.sum()),
        "damage_right_censored": int((~ensemble.damage_event_observed).sum()),
        "opening_events": int(ensemble.opening_event_observed.sum()),
        "opening_right_censored": int((~ensemble.opening_event_observed).sum()),
        "natural_service_life_interpretation": False,
    }
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
