"""Build the article's three-dimensional field figures from archived results.

The script reads the residual-controlled field files and writes Figure 13 and
Figure 15. A rebar-local camera window keeps the steel bar visible across the
three-dimensional panels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


STEEL_RENDER_COLOR = "#0B5CAD"
STEEL_EDGE_COLOR = "#08233F"
PRESSURE_POINT_COLOR = "#D55E00"
PRESSURE_POINT_EDGE_COLOR = "#6B2B00"
VIEW_ELEV = 10
VIEW_AZIM = -142
REBAR_MARGIN_MM = 17.0


CASE_FILES = {
    "baseline_3d": "baseline/paper_baseline_3d/fields.npz",
    "full_cover_40mm": "parameters/cover_40mm/fields.npz",
    "full_diffusivity_high": "parameters/diffusivity_2p0/fields.npz",
    "full_gap_low": "parameters/storage_4um/fields.npz",
    "full_rust_gamma_high": "parameters/expansion_3p5/fields.npz",
    "full_fracture_high": "parameters/fracture_240/fields.npz",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.linewidth": 0.75,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel(ax, label: str) -> None:
    kwargs = {
        "transform": ax.transAxes,
        "ha": "left",
        "va": "bottom",
        "fontsize": 9,
        "fontweight": "bold",
        "clip_on": False,
        "zorder": 20,
    }
    if hasattr(ax, "text2D"):
        ax.text2D(-0.115, 1.045, label, **kwargs)
    else:
        ax.text(-0.115, 1.045, label, **kwargs)


def _load_case(data_root: Path, case_id: str) -> dict[str, np.ndarray]:
    path = data_root / CASE_FILES[case_id]
    if not path.is_file():
        raise FileNotFoundError(f"Required field data not found: {path}")
    with np.load(path) as raw:
        required = {
            "X",
            "Y",
            "Z",
            "damage",
            "rust_pressure",
            "aggregate",
            "rebar",
            "rebar_interface",
        }
        missing = required.difference(raw.files)
        if missing:
            raise KeyError(f"{path} is missing fields: {sorted(missing)}")
        return {
            "X": np.asarray(raw["X"]),
            "Y": np.asarray(raw["Y"]),
            "Z": np.asarray(raw["Z"]),
            "damage": np.asarray(raw["damage"]),
            "peak_rust_pressure": np.asarray(raw["rust_pressure"]),
            "is_agg": np.asarray(raw["aggregate"], dtype=bool),
            "is_rebar": np.asarray(raw["rebar"], dtype=bool),
            "rebar_bnd": np.asarray(raw["rebar_interface"], dtype=bool),
        }


def _display_xyz(
    d: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ymax = float(np.nanmax(d["Y"]))
    return (ymax - y) * 1000.0, z * 1000.0, x * 1000.0


def _rebar_geometry(d: dict[str, np.ndarray]) -> tuple[float, float, float]:
    x = d["X"]
    y = d["Y"]
    mask = d["is_rebar"]
    cx = float(np.nanmean(x[mask]))
    cy = float(np.nanmean(y[mask]))
    radius = np.sqrt((x[mask] - cx) ** 2 + (y[mask] - cy) ** 2)
    return cx, cy, float(np.nanpercentile(radius, 98.0))


def _clean_3d_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(False)
    ax.set_axis_off()


def _zoom_to_rebar(
    ax,
    d: dict[str, np.ndarray],
    zoom: float,
    margin_mm: float = REBAR_MARGIN_MM,
) -> None:
    """Crop the 3D view in display coordinates without clipping field points."""
    cx, cy, _ = _rebar_geometry(d)
    ymax = float(np.nanmax(d["Y"]))
    display_x = (ymax - cy) * 1000.0
    display_z = cx * 1000.0
    zmin = float(np.nanmin(d["Z"])) * 1000.0
    zmax = float(np.nanmax(d["Z"])) * 1000.0
    ax.set_xlim(display_x - margin_mm, display_x + margin_mm)
    ax.set_ylim(zmin, zmax)
    ax.set_zlim(display_z - margin_mm, display_z + margin_mm)
    ax.set_box_aspect(
        (2.0 * margin_mm, zmax - zmin, 2.0 * margin_mm),
        zoom=zoom,
    )


def _plot_rebar_cylinder(
    ax,
    d: dict[str, np.ndarray],
    alpha: float = 0.99,
    color: str = STEEL_RENDER_COLOR,
    radius_scale: float = 1.12,
    outline: bool = True,
) -> None:
    cx, cy, radius = _rebar_geometry(d)
    radius *= radius_scale
    theta = np.linspace(0.0, 2.0 * np.pi, 56)
    z_axis = np.linspace(float(np.nanmin(d["Z"])), float(np.nanmax(d["Z"])), 28)
    theta_grid, z_grid = np.meshgrid(theta, z_axis)
    ymax = float(np.nanmax(d["Y"]))
    ax.plot_surface(
        (ymax - (cy + radius * np.sin(theta_grid))) * 1000.0,
        z_grid * 1000.0,
        (cx + radius * np.cos(theta_grid)) * 1000.0,
        color=color,
        alpha=alpha,
        edgecolor=STEEL_EDGE_COLOR,
        linewidth=0.32 if outline else 0.18,
        shade=True,
    )
    radial = np.linspace(0.0, radius, 18)
    theta_cap, radial_grid = np.meshgrid(theta, radial)
    for zcap in (float(np.nanmin(d["Z"])), float(np.nanmax(d["Z"]))):
        ax.plot_surface(
            (ymax - (cy + radial_grid * np.sin(theta_cap))) * 1000.0,
            np.full_like(theta_cap, zcap * 1000.0),
            (cx + radial_grid * np.cos(theta_cap)) * 1000.0,
            color=color,
            alpha=min(1.0, alpha + 0.05),
            edgecolor=STEEL_EDGE_COLOR,
            linewidth=0.24 if outline else 0.12,
            shade=True,
        )
    if not outline:
        return
    zmin = float(np.nanmin(d["Z"])) * 1000.0
    zmax = float(np.nanmax(d["Z"])) * 1000.0
    theta_overlay = np.linspace(0.0, 2.0 * np.pi, 22, endpoint=False)
    z_overlay = np.linspace(float(np.nanmin(d["Z"])), float(np.nanmax(d["Z"])), 34)
    theta_points, z_points = np.meshgrid(theta_overlay, z_overlay)
    ax.scatter(
        ((ymax - (cy + radius * np.sin(theta_points))) * 1000.0).ravel(),
        (z_points * 1000.0).ravel(),
        ((cx + radius * np.cos(theta_points)) * 1000.0).ravel(),
        s=3.1,
        c=color,
        edgecolors=STEEL_EDGE_COLOR,
        linewidths=0.10,
        alpha=0.82,
        depthshade=False,
    )
    ax.plot(
        [(ymax - cy) * 1000.0, (ymax - cy) * 1000.0],
        [zmin, zmax],
        [cx * 1000.0, cx * 1000.0],
        color="#061A2E",
        lw=1.15,
        alpha=0.98,
    )
    for theta0 in (np.pi / 2.0, 3.0 * np.pi / 2.0):
        xline = (ymax - (cy + radius * np.sin(theta0))) * 1000.0
        zline = (cx + radius * np.cos(theta0)) * 1000.0
        ax.plot(
            [xline, xline],
            [zmin, zmax],
            [zline, zline],
            color=STEEL_EDGE_COLOR,
            lw=1.05,
            alpha=0.96,
        )


def _plot_case_crack_pressure_3d(
    ax,
    d: dict[str, np.ndarray],
    zoom: float,
) -> None:
    crack_idx = np.argwhere(
        (d["damage"] >= 0.35) & ~(d["is_rebar"] | d["is_agg"])
    )
    if len(crack_idx) > 3200:
        sample = np.linspace(0, len(crack_idx) - 1, 3200).astype(int)
        crack_idx = crack_idx[sample]
    if len(crack_idx):
        xs, ys, zs = _display_xyz(
            d,
            d["X"][tuple(crack_idx.T)],
            d["Y"][tuple(crack_idx.T)],
            d["Z"][tuple(crack_idx.T)],
        )
        ax.scatter(
            xs,
            ys,
            zs,
            s=0.95,
            c=d["damage"][tuple(crack_idx.T)],
            cmap="inferno",
            vmin=0.35,
            vmax=1.0,
            alpha=0.40,
            depthshade=False,
        )
    else:
        ax.text2D(
            0.05,
            0.88,
            r"no $d\geq0.35$ cells",
            transform=ax.transAxes,
            fontsize=6.5,
            color="#4A5568",
        )
    pressure_idx = np.argwhere(
        d["rebar_bnd"] & (d["peak_rust_pressure"] > 0.0)
    )
    if len(pressure_idx) > 900:
        sample = np.linspace(0, len(pressure_idx) - 1, 900).astype(int)
        pressure_idx = pressure_idx[sample]
    if len(pressure_idx):
        xs, ys, zs = _display_xyz(
            d,
            d["X"][tuple(pressure_idx.T)],
            d["Y"][tuple(pressure_idx.T)],
            d["Z"][tuple(pressure_idx.T)],
        )
        ax.scatter(
            xs,
            ys,
            zs,
            s=2.55,
            c=PRESSURE_POINT_COLOR,
            edgecolors=PRESSURE_POINT_EDGE_COLOR,
            linewidths=0.12,
            alpha=0.78,
            depthshade=False,
        )
    _plot_rebar_cylinder(ax, d)
    _zoom_to_rebar(ax, d, zoom=zoom)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.view_init(VIEW_ELEV, VIEW_AZIM)
    _clean_3d_axis(ax)


def _surface_damage_projection(d: dict[str, np.ndarray]) -> np.ndarray:
    damage = np.array(d["damage"], dtype=float)
    masked = np.where(d["is_rebar"], np.nan, damage)
    return np.nanmax(masked, axis=1)


def _surface_damage_range(
    data_root: Path, case_ids: list[str]
) -> tuple[float, float]:
    values = []
    for case_id in case_ids:
        projection = _surface_damage_projection(_load_case(data_root, case_id))
        finite = projection[np.isfinite(projection)]
        if finite.size:
            values.append(finite)
    if not values:
        return 0.0, 1.0
    vmax = float(np.nanmax(np.concatenate(values)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    return 0.0, vmax * 1.05


def _plot_surface_damage_projection(
    ax,
    d: dict[str, np.ndarray],
    vmax: float,
    show_ylabel: bool,
):
    x = d["X"][:, 0, :] * 1000.0
    z = d["Z"][:, 0, :] * 1000.0
    projection = _surface_damage_projection(d)
    mesh = ax.pcolormesh(
        x,
        z,
        np.ma.masked_invalid(projection),
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        shading="auto",
    )
    ax.set_aspect("auto")
    ax.set_xlabel("surface coordinate (mm)", labelpad=1.2)
    if show_ylabel:
        ax.set_ylabel("bar axis (mm)", labelpad=1.2)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.tick_params(length=2.5, pad=1.5)
    ax.grid(True, color="#E2E8F0", lw=0.35, alpha=0.55)
    return mesh


def _rebar_surface_map(
    d: dict[str, np.ndarray], field: np.ndarray, ntheta: int = 48
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy, _ = _rebar_geometry(d)
    z_values = d["Z"][0, 0, :]
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    surface = np.full((len(z_values), ntheta), np.nan)
    count = np.zeros((len(z_values), ntheta), dtype=int)
    idx = np.argwhere(d["rebar_bnd"])
    if not len(idx):
        return theta, z_values, surface
    values = field[tuple(idx.T)]
    theta_points = np.mod(
        np.arctan2(d["Y"][tuple(idx.T)] - cy, d["X"][tuple(idx.T)] - cx),
        2.0 * np.pi,
    )
    z_points = d["Z"][tuple(idx.T)]
    theta_bins = (
        np.floor(theta_points / (2.0 * np.pi) * ntheta).astype(int) % ntheta
    )
    z_bins = np.array(
        [int(np.argmin(np.abs(z_values - value))) for value in z_points],
        dtype=int,
    )
    for z_index, theta_index, value in zip(z_bins, theta_bins, values):
        if not np.isfinite(value):
            continue
        if count[z_index, theta_index] == 0:
            surface[z_index, theta_index] = float(value)
        else:
            surface[z_index, theta_index] = max(
                surface[z_index, theta_index], float(value)
            )
        count[z_index, theta_index] += 1
    return theta, z_values, surface


def _pressure_range(
    data_root: Path, case_ids: list[str]
) -> tuple[float, float]:
    values = []
    for case_id in case_ids:
        d = _load_case(data_root, case_id)
        pressure = d["peak_rust_pressure"] / 1.0e6
        active = pressure[d["rebar_bnd"] & (pressure > 0.0)]
        if active.size:
            values.append(active[np.isfinite(active)])
    if not values:
        return 0.0, 1.0
    pooled = np.concatenate(values)
    vmin = float(np.nanmin(pooled))
    vmax = float(np.nanmax(pooled))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    margin = 0.02 * (vmax - vmin)
    return max(0.0, vmin - margin), vmax + margin


def _plot_pressure_unwrapped(
    ax,
    d: dict[str, np.ndarray],
    vmin: float,
    vmax: float,
    show_ylabel: bool,
):
    theta, z_values, surface = _rebar_surface_map(
        d, d["peak_rust_pressure"] / 1.0e6
    )
    theta_degrees = np.r_[np.degrees(theta), 360.0]
    pressure_closed = np.c_[surface, surface[:, :1]]
    mesh = ax.pcolormesh(
        theta_degrees,
        z_values * 1000.0,
        np.ma.masked_invalid(pressure_closed),
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    ax.set_xlim(0.0, 360.0)
    ax.set_xlabel(r"$\theta$ (deg)", labelpad=1.2)
    if show_ylabel:
        ax.set_ylabel("bar axis (mm)", labelpad=1.2)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.set_xticks([0, 180, 360])
    ax.tick_params(length=2.5, pad=1.5)
    ax.grid(True, color="#E2E8F0", lw=0.35, alpha=0.55)
    return mesh


def _set_colorbar_title(cbar, label: str) -> None:
    cbar.set_label(label, fontsize=7.1, labelpad=2.0)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    cbar.ax.tick_params(length=2.5, pad=1.5)


def _plot_triplet(
    fig,
    grid_cells,
    data_root: Path,
    case_id: str,
    labels: list[str],
    pressure_limits: tuple[float, float],
    damage_vmax: float,
    show_ylabel: bool,
    zoom: float,
):
    d = _load_case(data_root, case_id)
    ax = fig.add_subplot(grid_cells[0], projection="3d")
    _plot_case_crack_pressure_3d(ax, d, zoom=zoom)
    panel(ax, labels[0])
    ax = fig.add_subplot(grid_cells[1])
    surface_mesh = _plot_surface_damage_projection(
        ax, d, vmax=damage_vmax, show_ylabel=show_ylabel
    )
    panel(ax, labels[1])
    ax = fig.add_subplot(grid_cells[2])
    pressure_mesh = _plot_pressure_unwrapped(
        ax,
        d,
        vmin=pressure_limits[0],
        vmax=pressure_limits[1],
        show_ylabel=show_ylabel,
    )
    panel(ax, labels[2])
    return surface_mesh, pressure_mesh


def _build_figure(
    output: Path,
    data_root: Path,
    filename: str,
    cases: list[str],
    labels: list[list[str]],
    figure_height: float,
    height_ratios: list[float],
    zoom: float,
) -> None:
    ncols = len(cases)
    fig = plt.figure(figsize=(7.35, figure_height), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        ncols,
        left=0.062,
        right=0.902,
        bottom=0.085,
        top=0.945,
        width_ratios=[1.0] * ncols,
        height_ratios=height_ratios,
        hspace=0.245,
        wspace=0.190 if ncols == 3 else 0.185,
    )
    pressure_limits = _pressure_range(data_root, cases)
    damage_limits = _surface_damage_range(data_root, cases)
    surface_mesh = pressure_mesh = None
    for index, case_id in enumerate(cases):
        surface_mesh, pressure_mesh = _plot_triplet(
            fig,
            [grid[0, index], grid[1, index], grid[2, index]],
            data_root,
            case_id,
            [labels[0][index], labels[1][index], labels[2][index]],
            pressure_limits,
            damage_limits[1],
            show_ylabel=index == 0,
            zoom=zoom,
        )
    if surface_mesh is not None:
        cax_bounds = (
            [0.930, 0.374, 0.015, 0.188]
            if height_ratios[0] != 1.0
            else [0.930, 0.395, 0.015, 0.205]
        )
        cax = fig.add_axes(cax_bounds)
        cbar = fig.colorbar(surface_mesh, cax=cax)
        _set_colorbar_title(cbar, "surface damage")
    if pressure_mesh is not None:
        cax_bounds = (
            [0.930, 0.094, 0.015, 0.188]
            if height_ratios[0] != 1.0
            else [0.930, 0.095, 0.015, 0.205]
        )
        cax = fig.add_axes(cax_bounds)
        cbar = fig.colorbar(pressure_mesh, cax=cax)
        _set_colorbar_title(cbar, r"$p_r$ (MPa)")
    fig.savefig(output / filename)
    plt.close(fig)


def rebuild(data_root: Path, output: Path) -> None:
    data_root = data_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    style()
    _build_figure(
        output,
        data_root,
        "Figure_13.png",
        ["full_cover_40mm", "baseline_3d", "full_diffusivity_high"],
        [
            ["(a)", "(d)", "(g)"],
            ["(b)", "(e)", "(h)"],
            ["(c)", "(f)", "(i)"],
        ],
        figure_height=6.34,
        height_ratios=[1.28, 1.0, 1.0],
        zoom=1.20,
    )
    _build_figure(
        output,
        data_root,
        "Figure_15.png",
        [
            "full_gap_low",
            "baseline_3d",
            "full_rust_gamma_high",
            "full_fracture_high",
        ],
        [
            ["(a)", "(d)", "(g)", "(j)"],
            ["(b)", "(e)", "(h)", "(k)"],
            ["(c)", "(f)", "(i)", "(l)"],
        ],
        figure_height=5.85,
        height_ratios=[1.0, 1.0, 1.0],
        zoom=1.25,
    )


def main() -> None:
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build article Figure 13 and Figure 15."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=package_root / "results" / "service_exposure",
        help="Service-exposure result directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=package_root / "outputs" / "field_figures",
        help="Directory for Figure_13.png and Figure_15.png.",
    )
    args = parser.parse_args()
    rebuild(args.data, args.output)
    print((args.output / "Figure_13.png").resolve())
    print((args.output / "Figure_15.png").resolve())


if __name__ == "__main__":
    main()
