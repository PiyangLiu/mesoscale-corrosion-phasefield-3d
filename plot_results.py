"""Create a compact diagnostic figure from a completed simulation."""

from argparse import ArgumentParser
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results.png"))
    args = parser.parse_args()
    fields = np.load(args.results / "fields.npz")
    with (args.results / "history.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    times = np.asarray([float(row["times"]) for row in rows])
    max_current = np.asarray([float(row["max_icorr"]) for row in rows])
    damage = np.asarray([float(row["max_damage"]) for row in rows])
    pressure = np.asarray([float(row["max_pressure_mpa"]) for row in rows])
    rust_generated = 1.0e6 * np.asarray(
        [float(row["rust_generated_m"]) for row in rows]
    )
    rust_retained = 1.0e6 * np.asarray([float(row["rust_retained_m"]) for row in rows])
    rust_lost = 1.0e6 * np.asarray([float(row["rust_lost_m"]) for row in rows])
    mid = fields["damage"].shape[2] // 2

    material = np.zeros(fields["damage"][:, :, mid].shape, dtype=np.uint8)
    material[fields["itz_fraction"][:, :, mid] > 0.0] = 1
    material[fields["aggregate"][:, :, mid]] = 2
    material[fields["rebar"][:, :, mid]] = 3
    material_colors = ListedColormap(["#d9d9d9", "#f2b134", "#6c757d", "#8b1e1e"])
    material_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), material_colors.N)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    phase = axes[0, 0].imshow(
        material.T,
        origin="lower",
        cmap=material_colors,
        norm=material_norm,
        interpolation="nearest",
    )
    phase_bar = figure.colorbar(phase, ax=axes[0, 0], ticks=range(4))
    phase_bar.ax.set_yticklabels(["Mortar", "ITZ", "Aggregate", "Rebar"])
    axes[0, 0].set_title("Mid-thickness material phases")

    chloride = axes[0, 1].imshow(
        fields["chloride"][:, :, mid].T, origin="lower", cmap="viridis"
    )
    figure.colorbar(chloride, ax=axes[0, 1], label="Chloride (kg/m3)")
    axes[0, 1].set_title("Mid-thickness chloride field")

    phase_field = axes[0, 2].imshow(
        fields["damage"][:, :, mid].T,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=max(float(np.max(fields["damage"])), 0.1),
    )
    figure.colorbar(phase_field, ax=axes[0, 2], label="Damage")
    axes[0, 2].set_title("Mid-thickness phase field")

    axes[1, 0].plot(times, max_current, marker="o", color="#176b87")
    axes[1, 0].set(
        xlabel="Time (year)",
        ylabel="Maximum corrosion current (A/m2)",
        title="Corrosion-current history",
    )

    axes[1, 1].plot(times, rust_generated, color="#7b2d26", label="Generated")
    axes[1, 1].plot(times, rust_retained, color="#176b87", label="Retained")
    axes[1, 1].plot(times, rust_lost, color="#6c757d", label="Lost")
    axes[1, 1].set(
        xlabel="Time (year)",
        ylabel="Equivalent thickness (micrometre)",
        title="Corrosion-product balance",
    )
    axes[1, 1].legend(loc="upper left")

    pressure_axis = axes[1, 2]
    damage_axis = pressure_axis.twinx()
    pressure_line = pressure_axis.plot(
        times, pressure, marker="o", color="#176b87", label="Peak pressure"
    )[0]
    damage_line = damage_axis.plot(
        times, damage, marker="s", color="#b22222", label="Maximum damage"
    )[0]
    pressure_axis.set(
        xlabel="Time (year)",
        ylabel="Peak pressure (MPa)",
        title="Corrosion-cracking history",
    )
    damage_axis.set_ylabel("Maximum damage")
    damage_axis.set_ylim(0, 1.02)
    pressure_axis.legend(
        [pressure_line, damage_line],
        [pressure_line.get_label(), damage_line.get_label()],
        loc="upper left",
    )

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
