"""Tests for the Training calculator."""

import pytest
from ecal.calculators.training import Training
from ecal.calculators.model_flops import MLPCalculator


class TestTraining:
    @pytest.fixture
    def mlp_training(self, mlp_calculator):
        return Training(
            model_name="MLP",
            num_epochs=10,
            batch_size=32,
            processor_flops_per_second=1e13,
            processor_max_power=100,
            num_samples=1000,
            input_size=(1, 10),
            evaluation_strategy="cross_validation",
            k_folds=5,
            split_ratio=0.8,
            calculator=mlp_calculator,
        )

    def test_training_energy_positive(self, mlp_training):
        result = mlp_training.calculate_energy()
        assert result["training_energy"] > 0
        assert result["evaluation_energy"] > 0
        assert result["total_energy"] > 0

    def test_training_flops_positive(self, mlp_training):
        flops = mlp_training.calculate_flops_training()
        assert flops > 0

    def test_evaluation_flops_positive(self, mlp_training):
        flops = mlp_training.calculate_flops_evaluation()
        assert flops > 0

    def test_time_inversion_regression(self, mlp_calculator):
        """Regression test: train_time should be flops/flops_per_sec, not inverted."""
        training = Training(
            model_name="MLP",
            num_epochs=10,
            batch_size=32,
            processor_flops_per_second=1e13,
            processor_max_power=100,
            num_samples=1000,
            input_size=(1, 10),
            evaluation_strategy="cross_validation",
            k_folds=5,
            split_ratio=0.8,
            calculator=mlp_calculator,
        )
        result = training.calculate_energy()
        # train_time = training_flops / processor_flops_per_second
        # For small models, train_time should be small (< 1 sec), not huge
        assert result["train_time"] < 100  # Should be tiny, not 1e13/flops
        assert result["train_time"] == result["training_flops"] / 1e13

    def test_train_test_split_strategy(self, mlp_calculator):
        training = Training(
            model_name="MLP",
            num_epochs=10,
            batch_size=32,
            processor_flops_per_second=1e13,
            processor_max_power=100,
            num_samples=1000,
            input_size=(1, 10),
            evaluation_strategy="train_test_split",
            k_folds=5,
            split_ratio=0.8,
            calculator=mlp_calculator,
        )
        result = training.calculate_energy()
        assert result["total_energy"] > 0

    def test_invalid_evaluation_strategy(self, mlp_calculator):
        with pytest.raises(ValueError, match="Unsupported evaluation strategy"):
            Training(
                model_name="MLP",
                num_epochs=10,
                batch_size=32,
                processor_flops_per_second=1e13,
                processor_max_power=100,
                num_samples=1000,
                input_size=(1, 10),
                evaluation_strategy="bogus",
                k_folds=5,
                split_ratio=0.8,
                calculator=mlp_calculator,
            )

    def test_more_epochs_more_energy(self, mlp_calculator):
        def make_training(epochs):
            return Training(
                model_name="MLP",
                num_epochs=epochs,
                batch_size=32,
                processor_flops_per_second=1e13,
                processor_max_power=100,
                num_samples=1000,
                input_size=(1, 10),
                evaluation_strategy="cross_validation",
                k_folds=5,
                split_ratio=0.8,
                calculator=mlp_calculator,
            )

        e10 = make_training(10).calculate_energy()["training_energy"]
        e100 = make_training(100).calculate_energy()["training_energy"]
        assert e100 > e10
