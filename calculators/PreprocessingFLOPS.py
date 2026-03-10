"""Backward-compatibility shim — use ecal.calculators.preprocessing_flops instead."""
from ecal.calculators.preprocessing_flops import (  # noqa: F401
    PreprocessingFLOPCalculator,
    NormalizationCalculator,
    MinMaxScalingCalculator,
    GramianDifferenceFieldCalculator,
)

__all__ = [
    "PreprocessingFLOPCalculator",
    "NormalizationCalculator",
    "MinMaxScalingCalculator",
    "GramianDifferenceFieldCalculator",
]
