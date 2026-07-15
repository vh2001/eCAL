# eCAL

Analytical estimation of the energy cost of the AI lifecycle (J/bit).

eCAL computes the total energy consumed across the full AI model lifecycle — data transmission, preprocessing, training, evaluation, and inference — using closed-form FLOP formulas and hardware power profiles.

Published in **IEEE Journal on Selected Areas in Communications (JSAC), 2026**.

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/cfortuna/eCAL.git
cd eCAL
pip install -e ".[dev]"
```

### From PyPI

```bash
pip install ecal-energy
```

### Dependencies only (legacy)

```bash
pip install -r requirements.txt
```

## Quickstart

### Python API

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

### CLI

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

### Legacy (RunCalculator.py)

```bash
# Edit configs/CalculatorConfig.py, then:
python RunCalculator.py
```

## Supported Models

| Model       | FLOP Calculator        | Key Parameters                                      |
|-------------|------------------------|-----------------------------------------------------|
| MLP         | `MLPCalculator`        | `num_layers`, `din`, `dout`                         |
| CNN         | `CNNCalculator`        | `num_cnv_layers`, `num_pool_layers`, `i_r`, `k_r`   |
| KAN         | `KANCalculator`        | `num_layers`, `grid_size`, `din`, `dout`            |
| Transformer | `TransformerCalculator`| `context_length`, `embedding_size`, `num_heads`, `num_decoder_blocks` |

## Hardware Profiles

| Profile            | FP32 FLOPS   | TDP (W) | Device |
|--------------------|-------------|---------|--------|
| `apple_m2`         | 3.6 TFLOPS  | 22      | mps    |
| `nvidia_a100_80gb` | 19.5 TFLOPS | 400     | cuda   |
| `nvidia_h100_sxm`  | 67 TFLOPS   | 700     | cuda   |
| `generic_cpu`      | 1 TFLOPS    | 100     | cpu    |
| `generic_edge`     | 0.01 TFLOPS | 15      | cpu    |

## Architecture

```
                    ecal.estimate()
                         |
        +--------+-------+-------+--------+
        |        |       |       |        |
  Transmission  Preproc  Training  Eval  Inference
        |        |       |       |        |
        v        v       v       v        v
   Protocol   FLOP    FLOP     FLOP    FLOP
   Configs   Calcs   Calcs    Calcs   Calcs
                     (per model type)
                         |
                    Hardware Profile
                   (FLOPS, power, TDP)
                         |
                   Energy = time * power
                         |
                  eCAL = total_E / total_bits
```

## Configuration

Protocol configs are in `configs/ProtocolConfigs.py` — supports 7 OSI layers with multiple protocol options (HTTP, TCP, IPv4, WiFi, Bluetooth, etc.).

Calculator parameters are in `configs/CalculatorConfig.py` for the legacy `RunCalculator.py` interface.

## Development

```bash
pip install -e ".[dev]"
pytest                    # run tests
ruff check src/ tests/    # lint
mypy src/ecal/            # type check
```

## Citation

If you use this tool please cite our [paper](https://ieeexplore.ieee.org/abstract/document/11298182):
```
@ARTICLE{11298182,
  author={Chou, Shih-Kai and Hribar, Jernej and Hanžel, Vid and Mohorčič, Mihael and Fortuna, Carolina},
  journal={IEEE Journal on Selected Areas in Communications},
  title={The Energy Cost of Artificial Intelligence Lifecycle in Communication Networks},
  year={2026},
  volume={44},
  number={},
  pages={2427-2443},
  keywords={Artificial intelligence;Measurement;Costs;Energy consumption;Carbon dioxide;Training;Standards;Data centers;Open systems;Energy efficiency;AI model lifecycle;energy consumption;carbon footprint;metric;methodology},
  doi={10.1109/JSAC.2025.3642835}}
```

Other related work:
```
@INPROCEEDINGS{11349371,
  author={Chou, Shih-Kai and Hribar, Jernej and Bertalanič, Blaž and Mohorčič, Mihael and Lagkas, Thomas and Sarigiannidis, Panagiotis and Fortuna, Carolina},
  booktitle={2025 IEEE Conference on Network Function Virtualization and Software-Defined Networking (NFV-SDN)},
  title={Energy Cost of the AI/ML Workflow in O-RAN},
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Training;Measurement;Adaptation models;Costs;Open RAN;Hardware;Energy efficiency;Complexity theory;Artificial intelligence;Optimization;sustainable 6G networks;O-RAN;AI/ML Workflow;eCAL;lifecycle;energy;Carbon Footprint},
  doi={10.1109/NFV-SDN66355.2025.11349371}}
```

```
@INPROCEEDINGS{10849732,
  author={Chou, Shih-Kai and Hribar, Jernej and Mohorčič, Mihael and Fortuna, Carolina},
  booktitle={2024 IEEE Conference on Standards for Communications and Networking (CSCN)},
  title={Towards the Standardization of Energy Efficiency Metrics of the AI Lifecycle in 6G and Beyond},
  year={2024},
  volume={},
  number={},
  pages={187-190},
  keywords={Measurement;6G mobile communication;Energy consumption;Costs;Energy measurement;Energy efficiency;Computational efficiency;Quality of experience;Artificial intelligence;Standards;6G;AI-native network;energy efficiency},
  doi={10.1109/CSCN63874.2024.10849732}}
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
