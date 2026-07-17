"""Example: estimate the energy cost of an MLP's full AI lifecycle with eCAL.

Run with:
    python examples/basic_estimate.py
"""

import ecal

# Model architecture: a 3-layer MLP with 10 input features and 2 output classes.
model_params = {
    "num_layers": 3,
    "din": 10,
    "dout": 2,
}

result = ecal.estimate(
    model_type="MLP",
    model_params=model_params,
    num_samples=1000,          # training samples
    sample_size=10,             # features per sample
    num_epochs=50,
    num_inferences=10000,       # inference calls to amortize the cost over
    hardware="apple_m2",        # pick a profile from `ecal profiles`, or pass
                                 # processor_flops_per_second / processor_max_power directly
)

print("Energy breakdown (Joules):")
for stage in ("transmission", "preprocessing", "training", "evaluation", "inference_process"):
    print(f"  {stage:18s}: {result[stage]:.6f} J")
print(f"  {'total':18s}: {result['total']:.6f} J")
print(f"\neCAL metric: {result['ecal_j_per_bit']:.3e} J/bit")
