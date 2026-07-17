# Architecture

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

{py:func}`ecal.api.estimate` orchestrates the full lifecycle by composing five
stages, each backed by its own calculator class:

- **Transmission** — {py:class}`ecal.calculators.transmission.Transmission`
  estimates the energy of moving data across a configurable OSI protocol
  stack, using per-layer parameters from
  {py:mod}`ecal.configs.protocol_configs`.
- **Preprocessing** — {py:class}`ecal.calculators.preprocessing.DataPreprocessing`
  estimates the FLOPs (and resulting energy) of normalizing, scaling, or
  encoding input data, via one of the calculators in
  {py:mod}`ecal.calculators.preprocessing_flops`.
- **Training** — {py:class}`ecal.calculators.training.Training` estimates
  FLOPs for the training and evaluation passes over the dataset, using a
  model-specific FLOP calculator from
  {py:mod}`ecal.calculators.model_flops`.
- **Inference** — {py:class}`ecal.calculators.inference.Inference` estimates
  FLOPs for a batch of inference calls, reusing the same model-specific FLOP
  calculator as training.

Each stage converts its FLOP estimate into energy using a hardware profile
({py:class}`ecal.hardware.profiles.HardwareProfile`) — a FLOPS-per-second
throughput figure and a power draw in watts. `estimate()` sums the energy
across all stages and divides by the total number of bits processed to
produce the **eCAL** metric (J/bit).
