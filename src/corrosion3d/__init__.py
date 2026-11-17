"""Reference implementation of the three-dimensional corrosion-cracking model."""

from .solver import EngineeringParameters, SolverSummary, run_full_3d

__all__ = ["EngineeringParameters", "SolverSummary", "run_full_3d"]
__version__ = "1.2.0"
