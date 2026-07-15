"""Public API for eCAL energy estimation."""

import logging
from typing import Any, Dict, Optional

from ecal.calculators.transmission import Transmission
from ecal.calculators.preprocessing import DataPreprocessing
from ecal.calculators.training import Training
from ecal.calculators.inference import Inference
from ecal.calculators.model_flops import (
    MLPCalculator,
    CNNCalculator,
    KANCalculator,
    TransformerCalculator,
)
from ecal.hardware.profiles import get_profile

logger = logging.getLogger(__name__)

# Default protocol stack (generic for all layers)
_DEFAULT_PROTOCOLS = {
    "application": "Generic_application",
    "presentation": "Generic_presentation",
    "session": "Generic_session",
    "transport": "Generic_transport",
    "network": "Generic_network",
    "datalink": "Generic_datalink",
    "physical": "Generic_physical",
}


def estimate(
    model_type: str,
    model_params: Optional[Dict[str, Any]] = None,
    num_samples: int = 1000,
    sample_size: int = 10,
    num_epochs: int = 50,
    batch_size: int = 64,
    num_inferences: int = 10000,
    float_precision: int = 64,
    hardware: Optional[str] = None,
    processor_flops_per_second: Optional[float] = None,
    processor_max_power: Optional[float] = None,
    preprocessing_type: str = "normalization",
    evaluation_strategy: str = "cross_validation",
    k_folds: int = 5,
    split_ratio: float = 0.8,
    transmission_hops: Optional[list] = None,
    virtualization_overhead: float = 0.0,
) -> Dict[str, Any]:
    """Estimate the total energy cost of an AI model lifecycle.

    Args:
        model_type: Model architecture ("MLP", "CNN", "KAN", "Transformer")
        model_params: Architecture-specific parameters. Keys depend on model_type:
            MLP needs num_layers, din, dout; CNN needs num_cnv_layers,
            num_pool_layers, i_r, i_c, k_r, k_c, c_in; KAN needs num_layers,
            grid_size, din, dout; Transformer needs context_length,
            embedding_size, num_heads, num_decoder_blocks, feed_forward_size,
            vocab_size.
        num_samples: Number of training samples
        sample_size: Size of each sample (e.g., number of features)
        num_epochs: Number of training epochs
        batch_size: Training batch size
        num_inferences: Number of inference runs
        float_precision: Bits per float (default 64)
        hardware: Hardware profile name (e.g., "generic_cpu", "apple_m2")
        processor_flops_per_second: Override FLOPS (used if hardware is None)
        processor_max_power: Override power in watts (used if hardware is None)
        preprocessing_type: "normalization", "min_max_scaling", or "GADF"
        evaluation_strategy: "cross_validation" or "train_test_split"
        k_folds: Number of folds for cross-validation
        split_ratio: Train/test split ratio
        transmission_hops: List of dicts with protocol/failure_rate config per hop.
            Each dict may contain keys: application, presentation, session,
            transport, network, datalink, physical, failure_rate.
            Defaults to a single hop with generic protocols and 0% failure.
        virtualization_overhead: Energy overhead fraction [0, 1]

    Returns:
        Dictionary with energy breakdown (Joules) and eCAL metric (J/bit).
        Keys: "transmission", "preprocessing", "training", "evaluation",
        "inference", "inference_process", "total" (all in Joules),
        "ecal_j_per_bit" (J/bit), and bit counts "Ed bits", "inf_proc_bits",
        "total_bits".
    """
    if model_params is None:
        model_params = {}

    # Resolve hardware
    if hardware is not None:
        profile = get_profile(hardware)
        flops_ps = profile.flops_per_second_fp32
        max_power = profile.tdp_watts
    else:
        flops_ps = processor_flops_per_second or 1e13
        max_power = processor_max_power or 100

    # Build FLOP calculator
    calculator = _build_calculator(model_type, model_params, sample_size)
    input_size = _get_input_size(model_type, sample_size)

    # Transmission
    if transmission_hops is None:
        transmission_hops = [{"failure_rate": 0.0}]

    data_bits = num_samples * float_precision * sample_size
    transmission_energy = 0.0
    last_transmission = None
    for hop_cfg in transmission_hops:
        hop_protocols = {k: hop_cfg.get(k, v) for k, v in _DEFAULT_PROTOCOLS.items()}
        transmission = Transmission(
            failure_rate=hop_cfg.get("failure_rate", 0.0),
            **hop_protocols,
        )
        result = transmission.calculate_energy(data_bits)
        transmission_energy += result["total_energy"]
        last_transmission = transmission

    # Preprocessing
    dp_flops_ps = flops_ps if hardware else (processor_flops_per_second or 1e10)
    dp_max_power = max_power if hardware else (processor_max_power or 100)
    preprocessing = DataPreprocessing(
        preprocessing_type=preprocessing_type,
        processor_flops_per_second=dp_flops_ps,
        processor_max_power=dp_max_power,
        time_steps=sample_size,
    )
    preprocessing_calc = preprocessing.calculate_energy(num_samples, sample_size)
    preprocessing_energy = preprocessing_calc["total_energy"]

    # Training
    training = Training(
        model_name=model_type,
        num_epochs=num_epochs,
        batch_size=batch_size,
        processor_flops_per_second=flops_ps,
        processor_max_power=max_power,
        num_samples=num_samples,
        input_size=input_size,
        evaluation_strategy=evaluation_strategy,
        k_folds=k_folds,
        split_ratio=split_ratio,
        calculator=calculator,
    )
    training_calc = training.calculate_energy()
    training_energy = training_calc["training_energy"]
    evaluation_energy = training_calc["evaluation_energy"]

    # Inference
    inference = Inference(
        model_name=model_type,
        input_size=input_size,
        num_samples=num_inferences,
        processor_flops_per_second=flops_ps,
        processor_max_power=max_power,
        calculator=calculator,
    )
    inference_energy = inference.calculate_energy()

    # Inference transmission + preprocessing
    inference_transmission_energy = 0.0
    if last_transmission is not None:
        inf_bits = num_inferences * float_precision * sample_size
        inf_tx = last_transmission.calculate_energy(inf_bits)
        inference_transmission_energy = inf_tx["total_energy"]

    inference_preprocessing_calc = preprocessing.calculate_energy(num_inferences, sample_size)
    inference_preprocessing_energy = inference_preprocessing_calc["total_energy"]

    inference_process = inference_energy + inference_transmission_energy + inference_preprocessing_energy

    # Total
    total_energy = (
        transmission_energy
        + preprocessing_energy
        + training_energy
        + evaluation_energy
        + inference_process
    )
    total_energy *= 1 + virtualization_overhead

    ed_bits = float_precision * sample_size * num_samples
    inf_proc_bits = float_precision * sample_size * num_inferences
    total_bits = ed_bits + inf_proc_bits
    ecal_j_per_bit = total_energy / total_bits if total_bits > 0 else 0.0

    return {
        "transmission": transmission_energy,
        "preprocessing": preprocessing_energy,
        "training": training_energy,
        "evaluation": evaluation_energy,
        "inference": inference_energy,
        "inference_process": inference_process,
        "total": total_energy,
        "ecal_j_per_bit": ecal_j_per_bit,
        "Ed bits": ed_bits,
        "inf_proc_bits": inf_proc_bits,
        "total_bits": total_bits,
    }


def _build_calculator(model_type: str, model_params: dict, sample_size: int):
    """Build a FLOP calculator for the given model type."""
    mt = model_type.upper()
    if mt == "MLP":
        return MLPCalculator(
            num_layers=model_params.get("num_layers", 3),
            din=model_params.get("din", 10),
            dout=model_params.get("dout", 2),
            num_samples=sample_size,
            num_classes=model_params.get("num_classes", 2),
        )
    elif mt == "CNN":
        return CNNCalculator(
            num_cnv_layers=model_params.get("num_cnv_layers", 3),
            num_pool_layers=model_params.get("num_pool_layers", 3),
            i_r=model_params.get("i_r", 10),
            i_c=model_params.get("i_c", 1),
            k_r=model_params.get("k_r", 3),
            k_c=model_params.get("k_c", 1),
            c_in=model_params.get("c_in", 1),
            num_samples=sample_size,
            num_classes=model_params.get("num_classes", 2),
        )
    elif mt == "KAN":
        return KANCalculator(
            num_layers=model_params.get("num_layers", 3),
            grid_size=model_params.get("grid_size", 10),
            din=model_params.get("din", 10),
            dout=model_params.get("dout", 2),
            num_samples=sample_size,
            num_classes=model_params.get("num_classes", 2),
        )
    elif mt == "TRANSFORMER":
        return TransformerCalculator(
            context_length=model_params.get("context_length", 10),
            embedding_size=model_params.get("embedding_size", 16),
            num_heads=model_params.get("num_heads", 2),
            num_decoder_blocks=model_params.get("num_decoder_blocks", 3),
            feed_forward_size=model_params.get("feed_forward_size", 32),
            vocab_size=model_params.get("vocab_size", 2),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use MLP, CNN, KAN, or Transformer.")


def _get_input_size(model_type: str, sample_size: int):
    """Get the default input size tuple for a model type."""
    return (1, sample_size)
