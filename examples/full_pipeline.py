"""Example: build an eCAL estimate from its individual pipeline stages.

`ecal.estimate()` (see examples/basic_estimate.py) bundles all of this into
one call. This script instead drives each stage directly through the
underlying classes, which is useful when you need more control than
`estimate()` exposes -- e.g. a custom multi-hop transmission path, or
reusing one FLOP calculator across several estimates.

Pipeline modeled here: an edge device collects samples, sends them over two
network hops (WiFi -> Ethernet) to a server, which preprocesses, trains, and
serves inference -- then ships inference results back over the same path.

Run with:
    python examples/full_pipeline.py
"""

from ecal.calculators.inference import Inference
from ecal.calculators.model_flops import TransformerCalculator
from ecal.calculators.preprocessing import DataPreprocessing
from ecal.calculators.training import Training
from ecal.calculators.transmission import Transmission
from ecal.hardware.profiles import get_profile

# --- Configuration -------------------------------------------------------

hardware = get_profile("nvidia_h100_sxm")
flops_per_second = hardware.flops_per_second_fp32
processor_power = hardware.tdp_watts

num_train_samples = 5000
num_inferences = 20000
sample_size = 32       # sequence length, used as the model's input dimension
float_precision = 32    # bits per value transmitted/processed

# A small decoder-only Transformer; reused for both training and inference
# so its FLOP profile is only computed once per input shape.
calculator = TransformerCalculator(
    context_length=sample_size,
    embedding_size=64,
    num_heads=4,
    num_decoder_blocks=4,
    feed_forward_size=128,
    vocab_size=5000,
)
input_size = (1, sample_size)

# --- 1. Transmission: raw samples travel edge device -> gateway -> server,
#        each hop with its own protocol stack. ---------------------------

edge_to_gateway = Transmission(datalink="WIFI_MAC", physical="WIFI_PHY", failure_rate=0.01)
gateway_to_server = Transmission(datalink="ETHERNET", physical="Generic_physical", failure_rate=0.0)

train_bits = float_precision * sample_size * num_train_samples
hop1 = edge_to_gateway.calculate_energy(train_bits)
hop2 = gateway_to_server.calculate_energy(hop1["total_bits"])
transmission_energy = hop1["total_energy"] + hop2["total_energy"]

# --- 2. Preprocessing: normalize the data once it reaches the server. ----

preprocessing = DataPreprocessing(
    preprocessing_type="normalization",
    processor_flops_per_second=flops_per_second,
    processor_max_power=processor_power,
)
preprocessing_result = preprocessing.calculate_energy(num_train_samples, sample_size)
preprocessing_energy = preprocessing_result["total_energy"]

# --- 3. Training: 5-fold cross-validation over the preprocessed data. ----

training = Training(
    model_name="Transformer",
    batch_size=32,
    num_epochs=20,
    num_samples=num_train_samples,
    processor_flops_per_second=flops_per_second,
    processor_max_power=processor_power,
    input_size=input_size,
    evaluation_strategy="cross_validation",
    k_folds=5,
    split_ratio=0.8,
    calculator=calculator,
)
training_result = training.calculate_energy()

# --- 4. Inference: serve num_inferences requests with the trained model. -

inference = Inference(
    model_name="Transformer",
    input_size=input_size,
    num_samples=num_inferences,
    processor_flops_per_second=flops_per_second,
    processor_max_power=processor_power,
    calculator=calculator,
)
inference_energy = inference.calculate_energy()

# --- 5. Ship inference results back over the same two-hop network. ------

inference_bits = float_precision * sample_size * num_inferences
inf_hop1 = edge_to_gateway.calculate_energy(inference_bits)
inf_hop2 = gateway_to_server.calculate_energy(inf_hop1["total_bits"])
inference_transmission_energy = inf_hop1["total_energy"] + inf_hop2["total_energy"]

# --- Summary --------------------------------------------------------------

total_energy = (
    transmission_energy
    + preprocessing_energy
    + training_result["total_energy"]
    + inference_energy
    + inference_transmission_energy
)
total_bits = train_bits + inference_bits
ecal_j_per_bit = total_energy / total_bits

print("Energy breakdown (Joules):")
print(f"  transmission (data upload) : {transmission_energy:.6f} J")
print(f"  preprocessing              : {preprocessing_energy:.6f} J")
print(f"  training                   : {training_result['training_energy']:.6f} J")
print(f"  evaluation                 : {training_result['evaluation_energy']:.6f} J")
print(f"  inference                  : {inference_energy:.6f} J")
print(f"  transmission (results)     : {inference_transmission_energy:.6f} J")
print(f"  {'total':27s}: {total_energy:.6f} J")
print(f"\neCAL metric: {ecal_j_per_bit:.3e} J/bit")
