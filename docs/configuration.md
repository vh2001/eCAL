# Configuration

## Protocol configuration

The OSI-layer protocol parameters used by
{py:class}`ecal.calculators.transmission.Transmission` are defined in
{py:mod}`ecal.configs.protocol_configs`. Each layer (application,
presentation, session, transport, network, data link, physical) has a
dictionary mapping protocol name to a
{py:class}`ecal.configs.protocol_configs.LayerProtocol`, supporting protocols
such as HTTP, TCP, IPv4, WiFi, and Bluetooth.

## Hardware profiles

Hardware profiles supply the FLOPS-per-second throughput and power draw (TDP)
used to convert FLOP estimates into energy. They are managed by
{py:mod}`ecal.hardware.profiles` and loaded from a bundled YAML file
(`src/ecal/hardware/data/profiles.yaml`).

| Profile            | FP32 FLOPS   | TDP (W) | Device |
|--------------------|-------------|---------|--------|
| `apple_m2`         | 3.6 TFLOPS  | 22      | mps    |
| `nvidia_a100_80gb` | 19.5 TFLOPS | 300     | cuda   |
| `nvidia_h100_sxm`  | 67 TFLOPS   | 700     | cuda   |
| `generic_cpu`      | 1 TFLOPS    | 100     | cpu    |
| `generic_edge`     | 0.01 TFLOPS | 15      | cpu    |

List them at any time with:

```bash
ecal profiles
```

Or programmatically with {py:func}`ecal.hardware.profiles.list_profiles`.

```{note}
This page covers the configuration mechanism used by the installable
`ecal` package (`src/ecal/`). The repository's top-level `configs/` and
`RunCalculator.py` are a separate, pre-packaging interface and are not
covered by these docs.
```
