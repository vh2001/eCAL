"""Backward-compatibility shim — use ecal.calculators.transmission instead."""
from ecal.calculators.transmission import Transmission  # noqa: F401

__all__ = ["Transmission"]
