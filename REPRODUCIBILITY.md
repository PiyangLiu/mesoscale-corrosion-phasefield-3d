# Reproducing the calculations

## Environment

```bash
python -m venv .venv
python -m pip install -e .
python -m pip install -r requirements-dev.txt
pytest -q
```

The calculation programs use Python 3.11, NumPy, SciPy, PyAMG, pandas,
Matplotlib and Pillow.

## Service-exposure studies

`paper_studies.py` contains the baseline, time-step, parameter,
axial-transport and matched spatial-partition studies. The baseline
configuration is `configs/paper_baseline.json`.

```bash
python paper_studies.py --help
python paper_figures.py --results results/service_exposure --output outputs/service_figures
python build_field_figures.py --output outputs/field_figures
```

The eight Mullard-Stewart boundary cases are generated with:

```bash
python mullard_stewart_cases.py --output outputs/mullard_stewart_cases
```

The 100-year calculations use:

```bash
python service_exposure_horizon.py --config configs/service_horizon_h5.json --output outputs/service_horizon_h5 --progress
python service_exposure_horizon.py --config configs/service_horizon_h2p5.json --output outputs/service_horizon_h2p5 --progress
```

## Conditional accelerated localization

`accelerated_localization_study.py` generates the grid-seed and temporal
refinement cases. The released raw outputs are assembled with:

```bash
python build_localization_results.py
```

The resulting tables and figures are written to
`results/early_localization/grid_seed_summary`.

## Residual-opening calculation

The right-censored 2.5 mm state can be continued with:

```bash
python residual_opening_study.py \
  --output results/residual_opening/h2p5_right_censored \
  --case-id xu2024_profile_h2p5_adaptive_seed880 \
  --grid-mm 2.5 --dt-hours 2.5 --total-hours 600 \
  --post-opening-hours 10 --opening-threshold-mm 0.05 --unload-trigger-mm 0.051 \
  --seed 880 --rust-expansion 4.4 --compacted-fraction 0.9 \
  --wedge-fraction 1.0 --contact-stiffness 4e11 \
  --leakage-permeability 5e-4 --damage-coupling-tolerance 1e-3 \
  --adaptive-time-stepping --minimum-dt-hours 0.625 \
  --residual-stiffness 1e-5 --corrosion-profile-mode xu2024_fig8 --resume
```

The restart state is tied to the listed case identifier and parameters.

## Xu angular profile

The derived CSV and von Mises fit can be regenerated from a local copy of the
source figure:

```bash
python digitize_xray_angular_distribution.py \
  --source-image PATH_TO_FIGURE_8 \
  --output outputs/xu_digitization \
  --overlay outputs/xu_digitization_overlay.png
```
