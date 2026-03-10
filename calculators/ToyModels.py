"""Backward-compatibility shim — use ecal.calculators.toy_models instead."""
from ecal.calculators.toy_models import (  # noqa: F401
    SimpleMLP,
    SimpleCNN,
    SimpleMLP_practical,
    SimpleCNN_practical,
    KANLikeRegressor,
    MultiHeadSelfAttention,
    TransformerBlock,
    LossFun,
    Transformer,
)

__all__ = [
    "SimpleMLP",
    "SimpleCNN",
    "SimpleMLP_practical",
    "SimpleCNN_practical",
    "KANLikeRegressor",
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "LossFun",
    "Transformer",
]
