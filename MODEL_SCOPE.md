# Model equations and reported observables

The service-exposure solver advances saturation, chloride, sodium and hydroxyl
fields, reconstructs the electroneutral electric-field contribution used by the
reduced corrosion kinetics, and evaluates the local depassivation state and
corrosion current. Faraday steel loss is partitioned into retained and
transported corrosion product. The retained excess produces interface pressure,
which is coupled to three-dimensional displacement equilibrium and tensile
phase-field damage.

The residual-opening formulation separates mobile and compacted corrosion
product. The compacted state contributes an irreversible wedge, and a
primal-dual active-set contact solve enforces non-negative pressure,
non-negative separation and pressure-separation complementarity.

The article reports three distinct event classes:

1. first-local chloride-threshold crossing at the steel interface;
2. damage initiation and a face-connected steel-to-surface localization band;
3. displacement-based surface opening, including the current-interruption
   response of the compacted-rust contact formulation.

Service-exposure times and accelerated-current times describe their respective
boundary conditions. Conditional accelerated event times use the imposed
current, geometry and material parameters listed in the article and
supplementary material.
