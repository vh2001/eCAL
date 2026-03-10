"""Backward-compatibility shim — use ecal.calculators.training instead."""
from ecal.calculators.training import Training  # noqa: F401

__all__ = ["Training"]
