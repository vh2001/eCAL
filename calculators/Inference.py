"""Backward-compatibility shim — use ecal.calculators.inference instead."""
from ecal.calculators.inference import Inference  # noqa: F401

__all__ = ["Inference"]
