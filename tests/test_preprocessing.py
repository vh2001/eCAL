"""Tests for the DataPreprocessing calculator."""

import pytest
from ecal.calculators.preprocessing import DataPreprocessing
from ecal.calculators.preprocessing_flops import (
    NormalizationCalculator,
    MinMaxScalingCalculator,
    GramianDifferenceFieldCalculator,
)


class TestPreprocessingFlops:
    def test_normalization_flops(self):
        calc = NormalizationCalculator()
        result = calc.calculate_flops(100)
        assert result["total_flops"] == 6 * 100 + 1

    def test_min_max_scaling_flops(self):
        calc = MinMaxScalingCalculator()
        result = calc.calculate_flops(100)
        assert result["total_flops"] == 2 * 100 + 1

    def test_gadf_flops(self):
        calc = GramianDifferenceFieldCalculator()
        result = calc.calculate_flops(data_size=10, time_steps=5)
        assert result["total_flops"] > 0
        assert result["data_shape"] == (10, 5, 5)

    def test_normalization_more_data_more_flops(self):
        calc = NormalizationCalculator()
        f1 = calc.calculate_flops(100)["total_flops"]
        f2 = calc.calculate_flops(1000)["total_flops"]
        assert f2 > f1


class TestDataPreprocessing:
    def test_normalization_energy(self, normalization_preprocessing):
        result = normalization_preprocessing.calculate_energy(1000, 10)
        assert result["total_energy"] > 0

    def test_min_max_energy(self):
        pp = DataPreprocessing(
            preprocessing_type="min_max_scaling",
            processor_flops_per_second=1e10,
            processor_max_power=100,
        )
        result = pp.calculate_energy(1000, 10)
        assert result["total_energy"] > 0

    def test_gadf_energy(self):
        pp = DataPreprocessing(
            preprocessing_type="GADF",
            processor_flops_per_second=1e10,
            processor_max_power=100,
        )
        result = pp.calculate_energy(100, 10)
        assert result["total_energy"] > 0

    def test_invalid_preprocessing_type(self):
        with pytest.raises(ValueError, match="Unsupported preprocessing type"):
            DataPreprocessing(preprocessing_type="nonexistent")
