# Three-dimensional mesoscale corrosion phase-field model

This repository contains the Python implementation and numerical data associated
with the article *A 3D mesoscopic phase-field model for chloride-induced corrosion cracking in reinforced
concrete with multi-ion transport and corrosion-product pressure evolution*.

The model resolves mortar, aggregates, aggregate interfacial transition zones
(ITZs), reinforcing steel and the steel-concrete interface on a three-dimensional
voxel grid. The coupled calculation includes moisture and ionic transport,
depassivation, corrosion current, steel loss, corrosion-product retention and
transport, interfacial pressure, mechanical equilibrium, phase-field damage and
surface-opening observables.

## Numerical studies

The repository contains the calculations reported in the article:

- the 22-year service-exposure baseline;
- time-step, parameter and axial-transport studies;
- the 5.0, 2.5 and 1.25 mm matched spatial-partition study;
- the eight Mullard-Stewart accelerated-current boundary cases;
- the 5.0 and 2.5 mm 100-year mechanical-event calculations;
- the three-grid, three-seed conditional accelerated localization study;
- the 12-case conditional event-time ensemble;
- the 5 mm current-interruption opening calculation and the right-censored
  2.5 mm trajectory.

The first-local depassivation time is the end of the first accepted step at
which any steel-interface cell reaches the critical free-chloride value. It is
reported as a comparative transport-mechanism indicator at the stated spatial
partition. The accelerated localization event is the first six-neighbour
`d >= 0.20` component connecting the steel interface to the exposed surface.
Opening events use the instantaneous displacement gauge defined in the article.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt
pytest -q
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

## Repository structure

- `src/corrosion3d/`: coupled solver, microstructure generator, transport,
  mechanics, phase field, unilateral contact and diagnostics.
- `configs/`: service-exposure baseline, axial-control and 100-year
  configurations.
- `data/`: deterministic matched geometry and literature-derived numerical
  tables.
- `results/service_exposure/`: baseline, time-step, parameter, axial-control,
  matched-grid, Mullard-Stewart and 100-year results.
- `results/early_localization/`: grid-seed, temporal-refinement and conditional
  event-time data.
- `results/residual_opening/`: current-interruption result and the restartable
  endpoint of the right-censored adjacent-grid trajectory.
- `figures/`: figures used in the article.
- `tests/`: deterministic tests for the released model equations and event
  definitions.

The correspondence between result directories and article sections is listed
in [RESULTS_INDEX.md](RESULTS_INDEX.md). Field and history variables are defined
in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Rebuilding article results

Rebuild the conditional localization tables and figures:

```bash
python build_localization_results.py
```

Rebuild the service-exposure summary figures:

```bash
python paper_figures.py --results results/service_exposure --output outputs/service_figures
```

Rebuild the three-dimensional field panels for Figures 13 and 15:

```bash
python build_field_figures.py --output outputs/field_figures
```

Further commands and the residual-opening restart command are provided in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Literature-derived data

`data/public_literature/` contains the digitized Mullard-Stewart crack curves
and the angular corrosion-product thickness values derived from Xu et al.
(2024). Source references, DOI where available, and coordinate conventions are
stored with the tables.
Publisher images are not redistributed. The Xu digitization utility accepts a
user-provided local copy of the source figure.

## Citation and license

Software citation metadata are provided in [CITATION.cff](CITATION.cff). The
source code is distributed under the MIT License.
