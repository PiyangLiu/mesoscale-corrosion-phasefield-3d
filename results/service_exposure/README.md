# Service-exposure calculation data

This directory contains the archived service-exposure and Mullard-Stewart
accelerated-current calculations reported in the article and supplementary
material.

- `baseline/`: 22-year reference calculation at 2.5 mm spacing.
- `time_step/` and `time_step_summary.csv`: 1.0, 0.5 and 0.25 year steps.
- `mesh/` and `mesh_summary.csv`: matched 5.0, 2.5 and 1.25 mm spatial
  partitions through 35 years.
- `parameters/` and `parameter_summary.csv`: one-factor parameter study on the
  common 5 mm discretization.
- `axial_control/` and `axial_control_summary.csv`: axial-transport enabled and
  disabled calculations for seeds 88-92.
- `mullard_stewart_cases/`: eight Mullard-Stewart boundary calculations.
- `service_horizon_h5/` and `service_horizon_h2p5/`: 100-year mechanical-event
  calculations with 2.5 mm steel-interface width and 10 mm surface-observation
  depth.
- `RUN_RECEIPT.json`: solver identity, source hashes and event definitions.

The first-local depassivation time is recorded at the end of the first accepted
transport step whose steel-interface free chloride reaches the critical value.
The 100-year calculations reach neither `d_max = 0.20` nor a 0.05 mm surface
opening. Their endpoint damage values are 0.013651 at 5.0 mm and 0.015572 at
2.5 mm.
