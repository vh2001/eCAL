"""Tests for FLOP calculators."""

import pytest


class TestMLPCalculator:
    def test_positive_flops(self, mlp_calculator):
        result = mlp_calculator.calculate(None, (1, 10))
        assert result["total_flops"] > 0

    def test_more_layers_more_flops(self):
        from ecal.calculators.model_flops import MLPCalculator
        c3 = MLPCalculator(num_layers=3, din=10, dout=2)
        c6 = MLPCalculator(num_layers=6, din=10, dout=2)
        f3 = c3.calculate(None, (1, 10))["total_flops"]
        f6 = c6.calculate(None, (1, 10))["total_flops"]
        assert f6 > f3


class TestCNNCalculator:
    def test_positive_flops(self, cnn_calculator):
        result = cnn_calculator.calculate(None, (1, 1, 10))
        assert result["total_flops"] > 0


class TestKANCalculator:
    def test_positive_flops(self, kan_calculator):
        result = kan_calculator.calculate(None, (1, 10))
        assert result["total_flops"] > 0

    def test_more_layers_more_flops(self):
        from ecal.calculators.model_flops import KANCalculator
        c2 = KANCalculator(num_layers=2, grid_size=10, din=10, dout=2)
        c5 = KANCalculator(num_layers=5, grid_size=10, din=10, dout=2)
        f2 = c2.calculate(None, (1, 10))["total_flops"]
        f5 = c5.calculate(None, (1, 10))["total_flops"]
        assert f5 > f2


class TestTransformerCalculator:
    def test_positive_flops(self, transformer_calculator):
        result = transformer_calculator.calculate(None, (1, 10))
        assert result["total_flops"] > 0
        assert "breakdown" in result

    def test_breakdown_structure(self, transformer_calculator):
        result = transformer_calculator.calculate(None, (1, 10))
        breakdown = result["breakdown"]
        assert "attention" in breakdown
        assert "mlp_blocks_flops" in breakdown
        assert "per_block_flops" in breakdown
