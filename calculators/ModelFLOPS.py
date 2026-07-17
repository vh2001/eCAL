"""Backward-compatibility shim — use ecal.calculators.model_flops instead."""
from ecal.calculators.model_flops import (  # noqa: F401
    FLOPCalculator,
    FlopsCalculatorFactory,
    CalFlopsCalculatorHF,
    CalFlopsCalculatorPT,
    MLPCalculator,
    CNNCalculator,
    KANCalculator,
    TransformerCalculator,
)

__all__ = [
    "FLOPCalculator",
    "FlopsCalculatorFactory",
    "CalFlopsCalculatorHF",
    "CalFlopsCalculatorPT",
    "MLPCalculator",
    "CNNCalculator",
    "KANCalculator",
    "TransformerCalculator",
]
