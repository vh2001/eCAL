"""Shared fixtures for eCAL tests."""

import pytest

from ecal.calculators.transmission import Transmission
from ecal.calculators.preprocessing import DataPreprocessing
from ecal.calculators.model_flops import (
    MLPCalculator,
    CNNCalculator,
    KANCalculator,
    TransformerCalculator,
)


@pytest.fixture
def mlp_calculator():
    return MLPCalculator(num_layers=3, din=10, dout=2)


@pytest.fixture
def cnn_calculator():
    return CNNCalculator(num_cnv_layers=3, num_pool_layers=3, i_r=10, i_c=1, k_r=3, k_c=1, c_in=1)


@pytest.fixture
def kan_calculator():
    return KANCalculator(num_layers=3, grid_size=10, din=10, dout=2)


@pytest.fixture
def transformer_calculator():
    return TransformerCalculator(
        context_length=10, embedding_size=16, num_heads=2,
        num_decoder_blocks=3, feed_forward_size=32, vocab_size=2,
    )


@pytest.fixture
def generic_transmission():
    return Transmission(
        application="Generic_application",
        presentation="Generic_presentation",
        session="Generic_session",
        transport="Generic_transport",
        network="Generic_network",
        datalink="Generic_datalink",
        physical="Generic_physical",
        failure_rate=0.0,
    )


@pytest.fixture
def normalization_preprocessing():
    return DataPreprocessing(
        preprocessing_type="normalization",
        processor_flops_per_second=1e10,
        processor_max_power=100,
    )
