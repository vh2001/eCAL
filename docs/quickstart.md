# Quickstart

## Python API

```python
import ecal

result = ecal.estimate(
    model_type="MLP",
    model_params={"num_layers": 3, "din": 10, "dout": 2},
    num_samples=1000,
    num_epochs=50,
    hardware="apple_m2",
)

print(f"Total energy: {result['total']:.4f} J")
print(f"eCAL: {result['ecal_j_per_bit']:.2e} J/bit")
```

See {py:func}`ecal.api.estimate` for the full parameter reference.

## CLI

```bash
# Estimate energy for an MLP
ecal estimate --model MLP --layers 3 --epochs 50 --hardware apple_m2

# JSON output
ecal estimate --model Transformer --layers 6 --hardware nvidia_h100_sxm --json

# List available hardware profiles
ecal profiles

# Version
ecal --version
```

See [CLI reference](cli.md) for the full list of flags.

## Supported Models

| Model       | FLOP Calculator        | Key Parameters                                      |
|-------------|------------------------|-----------------------------------------------------|
| MLP         | `MLPCalculator`        | `num_layers`, `din`, `dout`                         |
| CNN         | `CNNCalculator`        | `num_cnv_layers`, `num_pool_layers`, `i_r`, `k_r`   |
| KAN         | `KANCalculator`        | `num_layers`, `grid_size`, `din`, `dout`            |
| Transformer | `TransformerCalculator`| `context_length`, `embedding_size`, `num_heads`, `num_decoder_blocks` |
