import os
from dataclasses import dataclass
from typing import Dict

import yaml


@dataclass
class HardwareProfile:
    """Hardware profile for energy estimation.

    Args:
        name: Human-readable device name (e.g. "NVIDIA H100 SXM").
        flops_per_second_fp32: Peak throughput in FLOPS at FP32 precision.
        flops_per_second_fp16: Peak throughput in FLOPS at FP16 precision.
        tdp_watts: Thermal design power in watts; used as the processor's
            power draw when converting FLOP-derived time into energy.
        gpu_power_watts: GPU power draw in watts under load (0 for CPU-only
            profiles).
        idle_power_watts: Idle power draw in watts.
        device: Backend device identifier (e.g. "cpu", "cuda", "mps").
    """
    name: str
    flops_per_second_fp32: float
    flops_per_second_fp16: float
    tdp_watts: float
    gpu_power_watts: float
    idle_power_watts: float
    device: str


_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_PROFILES_CACHE: Dict[str, HardwareProfile] = {}


def _load_profiles() -> Dict[str, HardwareProfile]:
    """Load hardware profiles from YAML file."""
    if _PROFILES_CACHE:
        return _PROFILES_CACHE

    profiles_path = os.path.join(_DATA_DIR, "profiles.yaml")
    with open(profiles_path, "r") as f:
        data = yaml.safe_load(f)

    for key, vals in data.items():
        _PROFILES_CACHE[key] = HardwareProfile(
            name=vals["name"],
            flops_per_second_fp32=float(vals["flops_per_second_fp32"]),
            flops_per_second_fp16=float(vals["flops_per_second_fp16"]),
            tdp_watts=float(vals["tdp_watts"]),
            gpu_power_watts=float(vals["gpu_power_watts"]),
            idle_power_watts=float(vals["idle_power_watts"]),
            device=vals["device"],
        )

    return _PROFILES_CACHE


def get_profile(name: str) -> HardwareProfile:
    """Get a hardware profile by name.

    Args:
        name: Profile identifier (e.g., 'apple_m2', 'nvidia_h100_sxm', 'generic_cpu')

    Returns:
        HardwareProfile dataclass

    Raises:
        KeyError: If profile name is not found
    """
    profiles = _load_profiles()
    if name not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise KeyError(f"Unknown hardware profile '{name}'. Available: {available}")
    return profiles[name]


def list_profiles() -> Dict[str, HardwareProfile]:
    """List all available hardware profiles.

    Returns:
        Dictionary mapping profile names to HardwareProfile objects
    """
    return dict(_load_profiles())
