"""Integration tests for ecal.estimate()."""

import pytest
from ecal.api import estimate


class TestEstimate:
    def test_mlp_estimate(self):
        result = estimate(
            model_type="MLP",
            model_params={"num_layers": 3, "din": 10, "dout": 2},
            num_samples=100,
            num_epochs=5,
            num_inferences=100,
        )
        assert result["total"] > 0
        assert result["ecal_j_per_bit"] > 0
        assert result["training"] > 0
        assert result["inference"] > 0

    def test_cnn_estimate(self):
        result = estimate(
            model_type="CNN",
            model_params={"num_cnv_layers": 3, "num_pool_layers": 3},
            num_samples=100,
            num_epochs=5,
            num_inferences=100,
        )
        assert result["total"] > 0

    def test_kan_estimate(self):
        result = estimate(
            model_type="KAN",
            model_params={"num_layers": 3, "grid_size": 10, "din": 10, "dout": 2},
            num_samples=100,
            num_epochs=5,
            num_inferences=100,
        )
        assert result["total"] > 0

    def test_transformer_estimate(self):
        result = estimate(
            model_type="Transformer",
            model_params={
                "context_length": 10,
                "embedding_size": 16,
                "num_heads": 2,
                "num_decoder_blocks": 3,
                "feed_forward_size": 32,
                "vocab_size": 2,
            },
            num_samples=100,
            num_epochs=5,
            num_inferences=100,
        )
        assert result["total"] > 0

    def test_hardware_profile(self):
        result = estimate(
            model_type="MLP",
            model_params={"num_layers": 3, "din": 10, "dout": 2},
            hardware="generic_cpu",
            num_samples=100,
            num_epochs=5,
            num_inferences=100,
        )
        assert result["total"] > 0

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported model type"):
            estimate(model_type="LSTM")

    def test_result_keys(self):
        result = estimate(
            model_type="MLP",
            model_params={"num_layers": 3, "din": 10, "dout": 2},
            num_samples=100,
            num_epochs=5,
        )
        expected_keys = {
            "transmission", "preprocessing", "training", "evaluation",
            "inference", "inference_process", "total", "ecal_j_per_bit",
            "Ed bits", "inf_proc_bits", "total_bits",
        }
        assert expected_keys == set(result.keys())

    def test_virtualization_overhead(self):
        r1 = estimate(
            model_type="MLP",
            model_params={"num_layers": 3, "din": 10, "dout": 2},
            num_samples=100, num_epochs=5,
            virtualization_overhead=0.0,
        )
        r2 = estimate(
            model_type="MLP",
            model_params={"num_layers": 3, "din": 10, "dout": 2},
            num_samples=100, num_epochs=5,
            virtualization_overhead=0.5,
        )
        assert r2["total"] > r1["total"]

    def test_default_params(self):
        """Test that estimate works with minimal parameters."""
        result = estimate(model_type="MLP")
        assert result["total"] > 0
