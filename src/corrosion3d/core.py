"""Physical constants and deterministic three-dimensional mesostructure generation."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
FARADAY = 96485.33212
R_GAS = 8.314462618
TEMP_K = 293.15
M_FE = 0.055845
RHO_FE = 7850.0
N_FE = 2.0


def hoop_cracking_pressure(a, b, tensile_strength):
    return tensile_strength * (b**2 - a**2) / (a**2 + b**2)


@dataclass
class Microstructure3D:
    aggregate: np.ndarray
    itz: np.ndarray
    itz_fraction: np.ndarray
    rebar: np.ndarray
    records: list
    actual_aggregate_fraction: float
    actual_itz_fraction: float


def generate_microstructure_3d(
    shape=(61, 101, 25),
    size=(0.12, 0.2, 0.04),
    rebar_radius=0.01,
    cover_depth=0.05,
    target_aggregate_fraction=0.18,
    itz_width=0.0018,
    seed=88,
    max_attempts=6000,
    records=None,
):
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    lx, ly, lz = size
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    z = np.linspace(0.0, lz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    dx, dy, dz = (lx / (nx - 1), ly / (ny - 1), lz / (nz - 1))
    cx, cy = (lx / 2.0, ly - cover_depth - rebar_radius)
    rebar = (X - cx) ** 2 + (Y - cy) ** 2 <= rebar_radius**2
    available = ~rebar
    aggregate = np.zeros(shape, dtype=bool)
    target_nodes = int(target_aggregate_fraction * int(np.count_nonzero(available)))
    supplied_records = records is not None
    records = (
        []
        if records is None
        else [
            {
                "center": tuple((float(value) for value in record["center"])),
                "axes": tuple((float(value) for value in record["axes"])),
                "angle": float(record["angle"]),
            }
            for record in records
        ]
    )
    candidates = records if supplied_records else range(max_attempts)
    accepted_records = []
    for candidate in candidates:
        if not supplied_records and int(np.count_nonzero(aggregate)) >= target_nodes:
            break
        if supplied_records:
            px, py, pz = candidate["center"]
            axes = np.asarray(candidate["axes"], dtype=float)
            angle = float(candidate["angle"])
        else:
            r = rng.uniform(0.002, 0.006)
            axes = r * rng.uniform(0.65, 1.35, size=3)
            lower = axes[:2] + np.array([dx, dy])
            upper = np.array([lx, ly]) - lower
            if np.any(upper <= lower):
                continue
            px = rng.uniform(axes[0] + dx, lx - axes[0] - dx)
            py = rng.uniform(axes[1] + dy, ly - axes[1] - dy)
            pz = rng.uniform(0.0, lz)
            if (px - cx) ** 2 + (py - cy) ** 2 < (
                rebar_radius + max(axes) + itz_width
            ) ** 2:
                continue
            angle = rng.uniform(0.0, 2.0 * np.pi)
        ca, sa = (np.cos(angle), np.sin(angle))
        pad = itz_width + max(dx, dy, dz)
        ix0 = max(0, int((px - axes[0] - pad) / dx))
        ix1 = min(nx, int((px + axes[0] + pad) / dx) + 2)
        iy0 = max(0, int((py - axes[1] - pad) / dy))
        iy1 = min(ny, int((py + axes[1] + pad) / dy) + 2)
        iz0 = max(0, int((pz - axes[2] - pad) / dz))
        iz1 = min(nz, int((pz + axes[2] + pad) / dz) + 2)
        if ix1 <= ix0 or iy1 <= iy0 or iz1 <= iz0:
            continue
        sx = X[ix0:ix1, iy0:iy1, iz0:iz1] - px
        sy = Y[ix0:ix1, iy0:iy1, iz0:iz1] - py
        sz = Z[ix0:ix1, iy0:iy1, iz0:iz1] - pz
        xr = ca * sx + sa * sy
        yr = -sa * sx + ca * sy
        ellipsoid = (xr / axes[0]) ** 2 + (yr / axes[1]) ** 2 + (
            sz / axes[2]
        ) ** 2 <= 1.0
        if not np.any(ellipsoid):
            continue
        local_rebar = rebar[ix0:ix1, iy0:iy1, iz0:iz1]
        local_agg = aggregate[ix0:ix1, iy0:iy1, iz0:iz1]
        if np.any(ellipsoid & local_rebar):
            if supplied_records:
                raise ValueError(
                    "supplied aggregate records intersect the reinforcement"
                )
            continue
        if not supplied_records and np.any(ellipsoid & local_agg):
            continue
        aggregate[ix0:ix1, iy0:iy1, iz0:iz1] |= ellipsoid
        accepted_records.append(
            {"center": (px, py, pz), "axes": tuple(axes), "angle": float(angle)}
        )
    cell_size = max(dx, dy, dz)
    if np.any(aggregate):
        dist_to_agg = ndimage.distance_transform_edt(~aggregate, sampling=(dx, dy, dz))
        itz_fraction = np.clip(
            (float(itz_width) + cell_size - dist_to_agg) / cell_size, 0.0, 1.0
        )
    else:
        itz_fraction = np.zeros(shape, dtype=float)
    itz_fraction[aggregate | rebar] = 0.0
    itz = itz_fraction > 1e-12
    return Microstructure3D(
        aggregate=aggregate,
        itz=itz,
        itz_fraction=itz_fraction,
        rebar=rebar,
        records=accepted_records,
        actual_aggregate_fraction=float(
            np.count_nonzero(aggregate) / max(np.count_nonzero(available), 1)
        ),
        actual_itz_fraction=float(np.mean(itz_fraction)),
    )
