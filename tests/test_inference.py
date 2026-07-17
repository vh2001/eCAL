"""Tests for the Inference calculator."""

import pytest
from ecal.calculators.inference import Inference


class TestInference:
    def test_inference_energy_positive(self, mlp_calculator):
        inf = Inference(
            model_name="MLP",
            input_size=(1, 10),
            num_samples=1000,
            processor_flops_per_second=1e13,
            processor_max_power=100,
            calculator=mlp_calculator,
        )
        energy = inf.calculate_energy()
        assert energy > 0

    def test_inference_flops_positive(self, mlp_calculator):
        inf = Inference(
            model_name="MLP",
            input_size=(1, 10),
            num_samples=1000,
            processor_flops_per_second=1e13,
            processor_max_power=100,
            calculator=mlp_calculator,
        )
        flops = inf.calculate_flops()
        assert flops > 0

    def test_more_inferences_more_energy(self, mlp_calculator):
        def make_inf(n):
            return Inference(
                model_name="MLP",
                input_size=(1, 10),
                num_samples=n,
                processor_flops_per_second=1e13,
                processor_max_power=100,
                calculator=mlp_calculator,
            )

        e1 = make_inf(100).calculate_energy()
        e2 = make_inf(10000).calculate_energy()
        assert e2 > e1
