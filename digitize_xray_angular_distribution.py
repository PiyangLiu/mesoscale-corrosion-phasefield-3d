"""Digitize angular corrosion-product thickness from Xu et al. (2024), Fig. 8.

The published heatmaps use five discrete colours for 0--2.5 mm thickness.
Values are reported as bin midpoints with a +/-0.25 mm quantization interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import optimize


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "public_literature"
PALETTE = np.asarray(
    [
        (100, 172, 222),
        (210, 232, 246),
        (218, 187, 218),
        (178, 123, 181),
        (135, 61, 148),
    ],
    dtype=float,
)
BIN_MIDPOINTS_MM = np.asarray([0.25, 0.75, 1.25, 1.75, 2.25])
PANELS = {
    "left": (26, 8, 240, 104),
    "center": (26, 199, 240, 295),
    "right": (26, 389, 240, 485),
}


def classify_pixels(rgb):
    distances = np.linalg.norm(rgb[..., None, :].astype(float) - PALETTE, axis=-1)
    index = np.argmin(distances, axis=-1)
    distance = np.min(distances, axis=-1)
    values = BIN_MIDPOINTS_MM[index]
    values[distance > 24.0] = np.nan
    return values, distance


def fit_von_mises(angles_deg, thickness_mm):
    theta = np.deg2rad(np.asarray(angles_deg, dtype=float))
    observed = np.asarray(thickness_mm, dtype=float)

    def model(parameters):
        baseline, amplitude, kappa, direction = parameters
        return baseline + amplitude * np.exp(kappa * (np.cos(theta - direction) - 1.0))

    def residual(parameters):
        return model(parameters) - observed

    initial_direction = float(theta[int(np.nanargmax(observed))])
    result = optimize.least_squares(
        residual,
        x0=np.asarray([0.1, max(float(np.max(observed) - 0.1), 0.1), 1.0, initial_direction]),
        bounds=(
            np.asarray([0.0, 0.0, 0.0, -2.0 * np.pi]),
            np.asarray([3.0, 5.0, 8.0, 2.0 * np.pi]),
        ),
    )
    fitted = model(result.x)
    rmse = float(np.sqrt(np.mean((fitted - observed) ** 2)))
    return {
        "baseline_mm": float(result.x[0]),
        "amplitude_mm": float(result.x[1]),
        "kappa": float(result.x[2]),
        "direction_deg": float(np.rad2deg(result.x[3]) % 360.0),
        "rmse_mm": rmse,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Digitize Xu et al. (2024), Fig. 8 from a user-supplied image."
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        required=True,
        help="Local image of Fig. 8 obtained by the user under applicable access terms.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--overlay",
        type=Path,
        default=None,
        help="Optional path for a diagnostic overlay; omitted by default.",
    )
    args = parser.parse_args()
    source = args.source_image.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("RGB")
    array = np.asarray(image)
    rows = []
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    fits = {}
    for panel, (x0, y0, x1, y1) in PANELS.items():
        crop = array[y0:y1, x0:x1]
        values, _ = classify_pixels(crop)
        height = y1 - y0
        panel_rows = []
        for row_index in range(12):
            lower = int(round(row_index * height / 12.0))
            upper = int(round((row_index + 1) * height / 12.0))
            sample = values[lower:upper]
            finite = sample[np.isfinite(sample)]
            angle_deg = float((330 - 30 * row_index) % 360)
            if finite.size == 0:
                mean = np.nan
                lower_mean = np.nan
                upper_mean = np.nan
            else:
                mean = float(np.mean(finite))
                lower_mean = float(np.mean(np.maximum(finite - 0.25, 0.0)))
                upper_mean = float(np.mean(np.minimum(finite + 0.25, 2.5)))
            row = {
                "panel": panel,
                "panel_semantics": "separate parallel rebar in the multiple-rebar specimen",
                "angle_deg": angle_deg,
                "model_angle_from_cover_deg": float((angle_deg - 180.0) % 360.0),
                "thickness_midpoint_mean_mm": mean,
                "thickness_lower_mean_mm": lower_mean,
                "thickness_upper_mean_mm": upper_mean,
                "classified_pixel_fraction": float(finite.size / sample.size),
                "source_doi": "10.1016/j.cemconcomp.2024.105565",
                "source_figure": "Fig. 8a published heatmap",
            }
            rows.append(row)
            panel_rows.append(row)
            draw.rectangle(
                (x0, y0 + lower, x1 - 1, y0 + upper - 1),
                outline=(220, 20, 60),
                width=1,
            )
        panel_frame = pd.DataFrame(panel_rows).sort_values("angle_deg")
        fits[panel] = fit_von_mises(
            panel_frame["angle_deg"], panel_frame["thickness_midpoint_mean_mm"]
        )
        fits[panel]["classified_pixel_fraction_mean"] = float(
            panel_frame["classified_pixel_fraction"].mean()
        )
    frame = pd.DataFrame(rows)
    pooled = (
        frame.groupby("angle_deg", as_index=False)["thickness_midpoint_mean_mm"]
        .mean()
        .sort_values("angle_deg")
    )
    fits["pooled_three_rebars"] = fit_von_mises(
        pooled["angle_deg"], pooled["thickness_midpoint_mean_mm"]
    )
    fits["_coordinate_mapping"] = {
        "source_geometry": "Fig. 8 Left/Center/Right are three parallel rebars in the multiple-rebar specimen",
        "published_angle_reference": "Fig. 3 defines theta=180 degrees on the cover-facing side",
        "model_angle_reference": "model angle zero is the cover-facing direction",
        "conversion": "model_angle_deg = (published_angle_deg - 180) mod 360",
    }
    frame.to_csv(
        output / "xu2024_fig8_angular_rust_thickness_digitized.csv", index=False
    )
    (output / "xu2024_fig8_von_mises_fits.json").write_text(
        json.dumps(fits, indent=2), encoding="utf-8"
    )
    if args.overlay is not None:
        overlay_path = args.overlay.resolve()
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path)


if __name__ == "__main__":
    main()
