# Data dictionary

All dimensional model inputs use SI units unless a field name explicitly gives
another unit. Service-exposure time is in years; accelerated histories include
both `times_h` (hours) and `times` (years).

## Common files

- `summary.json`: configuration, endpoint events, solver residuals, mass balance,
  and interpretation flags for one case.
- `history.csv`: accepted-step scalar histories. Important fields include
  pressure, damage, current, crack opening, rust inventories, numerical
  residuals, event times, and connected-component measures.
- `fields.npz`: compressed NumPy arrays containing final fields and masks.
- `aggregate_geometry.json`: deterministic aggregate records used to reproduce a
  mesostructure.
- `receipt.json` or `RUN_RECEIPT.json`: case identity, source, settings, and
  aggregation provenance.

## Important history fields

| Field family | Meaning | Unit |
|---|---|---|
| `max_pressure_mpa` | Accepted end-of-step maximum rust pressure | MPa |
| `trial_pressure_max_mpa` | Maximum pressure before within-step damage release | MPa |
| `max_damage` | Maximum irreversible phase-field damage | 1 |
| `surface_crack_width_mm` | Current virtual-gauge opening | mm |
| `surface_crack_width_envelope_mm` | Historical maximum gauge opening | mm |
| `rust_generated_m`, `rust_retained_m`, `rust_lost_m` | Area-normalized product inventories | m |
| `rust_*_volume_m3` | Integrated product inventories | m3 |
| `mechanics_relative_residual` | Relative force residual | 1 |
| `phasefield_relative_residual` | Relative phase-field residual | 1 |
| `coupling_pressure_relative_residual` | Pressure/contact compatibility residual | 1 |
| `pressure_cap_trigger_count` | Cells exceeding the pressure-release criterion in a step | count |
| `pressure_overflow_volume_m3` | Product removed by the recorded release rule | m3 |
| `max_tensile_strain` | Maximum tensile principal strain diagnostic | 1 |
| `max_phasefield_driving_energy` | Maximum tensile phase-field driving energy density | J/m3 |

## Event definitions

- `depassivation_year`: end of the first accepted step in which any
  steel-interface cell reaches the critical free-chloride value.
- `spanning_d0p20_year`: first accepted step with a six-neighbour `d >= 0.20`
  component touching both the steel-interface band and true exposed surface.
- `opening_0p05mm_year`: first accepted step whose current displacement gauge is
  at least 0.05 mm. A blank or null value is a right-censored outcome.

## Right-censored restart state

`restart_state.pkl` stores the solver state at the 145.625 h right-censoring
time. The corresponding scalar history and endpoint summary are provided as
`history.csv` and `summary.json`. Continue the trajectory with release version
1.2.0 and the exact case settings in `REPRODUCIBILITY.md`.
