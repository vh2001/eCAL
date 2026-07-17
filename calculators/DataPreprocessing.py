"""Backward-compatibility shim — use ecal.calculators.preprocessing instead."""
from ecal.calculators.preprocessing import DataPreprocessing  # noqa: F401

__all__ = ["DataPreprocessing"]
