"""Generate article figures from the archived residual-controlled results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def make_figures(results: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 10.0})

    acc = pd.read_csv(results / "mullard_stewart_cases" / "case_summary.csv")
    x = np.arange(len(acc))
    labels = acc["case_id"].str.replace("_", " ", regex=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    axes[0].bar(x, acc["peak_pressure_mpa"], color="#0072B2")
    axes[0].set_ylabel("Peak pressure (MPa)")
    axes[1].bar(x, acc["final_max_damage"], color="#D55E00")
    axes[1].set_ylabel("Final maximum damage")
    axes[2].semilogy(x, acc["maximum_phasefield_residual"], "o-", label="Phase field")
    axes[2].semilogy(x, acc["maximum_mechanics_residual"], "s-", label="Mechanics")
    axes[2].set_ylabel("Maximum relative residual")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right")
        ax.grid(axis="y", alpha=0.25)
    _save(fig, output / "Figure_8.png")

    axial_path = results / "axial_control_summary.csv"
    if axial_path.exists():
        axial = pd.read_csv(axial_path)
        pivot = axial.pivot(index="seed", columns="axial_transport", values="depassivation_year")
        x = np.arange(len(pivot))
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)
        axes[0].plot(x, pivot["enabled"], "o-", label="Axial transport enabled")
        axes[0].plot(x, pivot["disabled"], "s--", label="Axial transport disabled")
        axes[0].set(xticks=x, xticklabels=pivot.index, xlabel="Mesostructure seed", ylabel="First-local indicator (yr)")
        axes[0].legend(frameon=False, fontsize=8)
        delta = pivot["disabled"] - pivot["enabled"]
        axes[1].bar(x, delta, color="#0072B2")
        axes[1].axhline(0.0, color="black", lw=0.8)
        axes[1].set(xticks=x, xticklabels=pivot.index, xlabel="Mesostructure seed", ylabel="Disabled minus enabled (yr)")
        for ax in axes:
            ax.grid(axis="y", alpha=0.25)
        _save(fig, output / "Figure_10_axial.png")

    base = pd.read_csv(results / "baseline" / "paper_baseline_3d" / "history.csv")
    years = base["times"]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1), constrained_layout=True)
    axes[0].plot(years, base["elastic_energy_j"], label="Elastic energy")
    axes[0].plot(years, base["fracture_energy_j"], label="Fracture energy")
    axes[0].plot(years, base["pressure_work_j"], label="Pressure work")
    axes[0].set(xlabel="Time (yr)", ylabel="Energy (J)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(years, base["max_pressure_mpa"], color="#0072B2", label="Peak pressure")
    axes[1].plot(years, base["max_damage"], color="#D55E00", label="Maximum damage")
    axes[1].set(xlabel="Time (yr)", ylabel="Pressure (MPa) / damage")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].semilogy(years, np.maximum(np.abs(base["rust_mass_balance_error_m"]), 1e-30), color="#009E73")
    axes[2].set(xlabel="Time (yr)", ylabel="Mass-balance error (m)")
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, output / "Figure_9.png")

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(years, base["max_pressure_mpa"], color="#0072B2")
    axes[0].set(xlabel="Time (yr)", ylabel="Peak pressure (MPa)")
    axes[1].plot(years, base["max_damage"], color="#D55E00")
    axes[1].axhline(0.20, color="0.4", ls=":")
    axes[1].set(xlabel="Time (yr)", ylabel="Maximum damage")
    axes[2].plot(years, base["mechanical_energy_relative_error"], color="#CC79A7")
    axes[2].set(xlabel="Time (yr)", ylabel="|Wp - 2Eel| / max(Wp, 2Eel)")
    axes[3].semilogy(years, np.maximum(base["transport_relative_residual"], 1e-30), label="Transport")
    axes[3].semilogy(years, np.maximum(base["phasefield_relative_residual"], 1e-30), label="Phase field")
    axes[3].set(xlabel="Time (yr)", ylabel="Relative residual")
    axes[3].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, output / "Figure_11.png")

    par = pd.read_csv(results / "parameter_summary.csv")
    transport_ids = ["cover_40mm", "parameter_baseline", "cover_60mm", "chloride_2p5", "chloride_5p5", "diffusivity_0p5", "diffusivity_2p0"]
    tr = par[par.case_id.isin(transport_ids)].copy()
    transport_labels = {
        "parameter_baseline": "baseline",
        "cover_40mm": "cover\n40 mm",
        "cover_60mm": "cover\n60 mm",
        "chloride_2p5": "surface Cl\n2.5",
        "chloride_5p5": "surface Cl\n5.5",
        "diffusivity_0p5": "mortar D\n0.5x",
        "diffusivity_2p0": "mortar D\n2.0x",
    }
    fig, ax = plt.subplots(figsize=(8.8, 3.7), constrained_layout=True)
    ax.bar(np.arange(len(tr)), tr["depassivation_year"], color="#0072B2")
    ax.set_xticks(np.arange(len(tr)))
    ax.set_xticklabels([transport_labels.get(case_id, case_id) for case_id in tr["case_id"]], rotation=0)
    ax.set_ylabel("First-local indicator (yr)")
    ax.set_title("Transport and exposure sensitivity")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output / "Figure_12.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.3), constrained_layout=True)
    parameter_labels = {
        "parameter_baseline": "baseline",
        "cover_40mm": "cover 40",
        "cover_60mm": "cover 60",
        "chloride_2p5": "Cl 2.5",
        "chloride_5p5": "Cl 5.5",
        "diffusivity_0p5": "D 0.5x",
        "diffusivity_2p0": "D 2.0x",
    }
    short_labels = [parameter_labels.get(case_id, case_id) for case_id in tr["case_id"]]
    axes[0].bar(np.arange(len(tr)), tr["final_max_damage"], color="#D55E00")
    axes[0].set_ylabel("Final maximum damage")
    axes[0].set_xticks(np.arange(len(tr)), short_labels)
    axes[1].bar(np.arange(len(tr)), tr["peak_pressure_mpa"], color="#009E73")
    axes[1].set_ylabel("Peak pressure (MPa)")
    axes[1].set_xticks(np.arange(len(tr)), short_labels)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    _save(fig, output / "Figure_13.png")

    mech_ids = ["storage_4um", "parameter_baseline", "storage_25um", "expansion_2p0", "expansion_3p5"]
    mech = par[par.case_id.isin(mech_ids)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), constrained_layout=True)
    axes[0].bar(np.arange(len(mech)), mech["depassivation_year"], color="#0072B2")
    axes[0].set_ylabel("First-local indicator (yr)")
    axes[0].set_xticks(np.arange(len(mech)), mech["case_id"].str.replace("_", "\n", regex=False))
    axes[1].bar(np.arange(len(mech)), mech["final_max_damage"], color="#D55E00")
    axes[1].set_ylabel("Final maximum damage")
    axes[1].set_xticks(np.arange(len(mech)), mech["case_id"].str.replace("_", "\n", regex=False))
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    _save(fig, output / "Figure_14.png")

    frac = par[par.case_id.isin(["fracture_160", "parameter_baseline", "fracture_240"])].copy()
    fig, ax = plt.subplots(figsize=(6.4, 3.5), constrained_layout=True)
    ax.bar(np.arange(len(frac)), frac["final_max_damage"], color="#CC79A7")
    ax.set_ylabel("Final maximum damage")
    ax.set_xticks(np.arange(len(frac)), frac["case_id"].str.replace("_", "\n", regex=False))
    ax.set_title("Fracture-energy sensitivity in the pre-localization regime")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output / "Figure_15.png")

    horizon_cases = {
        "5.0 mm": results / "service_horizon_h5" / "history.csv",
        "2.5 mm": results / "service_horizon_h2p5" / "history.csv",
    }
    if all(path.exists() for path in horizon_cases.values()):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
        for label, path in horizon_cases.items():
            horizon = pd.read_csv(path)
            axes[0].plot(horizon["times"], horizon["max_damage"], label=label)
            axes[1].plot(
                horizon["times"],
                horizon["surface_crack_width_envelope_mm"],
                label=label,
            )
        axes[0].axhline(0.20, color="0.35", ls=":", label="Localization threshold")
        axes[1].axhline(0.05, color="0.35", ls=":", label="Opening threshold")
        axes[0].set(xlabel="Time (yr)", ylabel="Maximum damage")
        axes[1].set(xlabel="Time (yr)", ylabel="Resolved surface opening (mm)")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
        _save(fig, output / "Figure_17.png")

    dt_path = results / "time_step_summary.csv"
    mesh_path = results / "mesh_summary.csv"
    if dt_path.exists() and mesh_path.exists():
        dt = pd.read_csv(dt_path).sort_values("dt_year")
        mesh = pd.read_csv(mesh_path).sort_values("grid_spacing_mm")
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), constrained_layout=True)
        axes[0, 0].plot(dt["dt_year"], dt["depassivation_year"], "o-")
        axes[0, 0].set(xlabel="Time step (yr)", ylabel="First-local indicator (yr)")
        axes[0, 1].plot(dt["dt_year"], dt["maximum_mechanical_energy_relative_error"], "s-")
        axes[0, 1].set(xlabel="Time step (yr)", ylabel="Maximum relative energy mismatch")
        axes[1, 0].plot(mesh["grid_spacing_mm"], mesh["depassivation_year"], "o-")
        axes[1, 0].set(xlabel="Grid spacing (mm)", ylabel="First-local indicator (yr)")
        axes[1, 1].plot(mesh["grid_spacing_mm"], mesh["final_max_damage"], "s-")
        axes[1, 1].axhline(0.20, color="#666666", ls=":")
        axes[1, 1].set(xlabel="Grid spacing (mm)", ylabel="Final maximum damage")
        for ax in axes.ravel():
            ax.grid(alpha=0.25)
        _save(fig, output / "Figure_16.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    axes[0].bar(np.arange(len(acc)), acc["final_surface_width_mm"], color="#56B4E9")
    axes[0].set_ylabel("Resolved surface opening (mm)")
    axes[0].set_xticks(np.arange(len(acc)), labels, rotation=55, ha="right")
    axes[1].axis("off")
    axes[1].text(0.02, 0.95, "All eight cases: zero resolved surface opening", va="top", fontsize=10)
    axes[1].text(0.02, 0.75, "Residual-controlled boundary check\n5 mm grid, 25 h step, 3500 h", va="top")
    _save(fig, output / "figS1_residual_checks.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_figures(args.results, args.output)
