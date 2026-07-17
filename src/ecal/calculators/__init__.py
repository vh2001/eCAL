"""eCAL calculator modules for energy estimation across the AI lifecycle."""

from ecal.calculators.transmission import Transmission
from ecal.calculators.preprocessing import DataPreprocessing
from ecal.calculators.training import Training
from ecal.calculators.inference import Inference
from ecal.calculators.model_flops import (
    FLOPCalculator,
    FlopsCalculatorFactory,
    CalFlopsCalculatorHF,
    CalFlopsCalculatorPT,
    MLPCalculator,
    CNNCalculator,
    KANCalculator,
    TransformerCalculator,
)
from ecal.calculators.preprocessing_flops import (
    PreprocessingFLOPCalculator,
    NormalizationCalculator,
    MinMaxScalingCalculator,
    GramianDifferenceFieldCalculator,
)

__all__ = [
    "Transmission",
    "DataPreprocessing",
    "Training",
    "Inference",
    "FLOPCalculator",
    "FlopsCalculatorFactory",
    "CalFlopsCalculatorHF",
    "CalFlopsCalculatorPT",
    "MLPCalculator",
    "CNNCalculator",
    "KANCalculator",
    "TransformerCalculator",
    "PreprocessingFLOPCalculator",
    "NormalizationCalculator",
    "MinMaxScalingCalculator",
    "GramianDifferenceFieldCalculator",
]
