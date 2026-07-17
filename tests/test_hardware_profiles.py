"""Tests for hardware profiles."""

import pytest
from ecal.hardware.profiles import get_profile, list_profiles, HardwareProfile


class TestHardwareProfiles:
    def test_list_profiles(self):
        profiles = list_profiles()
        assert len(profiles) >= 5
        assert "generic_cpu" in profiles
        assert "apple_m2" in profiles
        assert "nvidia_h100_sxm" in profiles

    def test_get_profile(self):
        profile = get_profile("generic_cpu")
        assert isinstance(profile, HardwareProfile)
        assert profile.name == "Generic CPU"
        assert profile.flops_per_second_fp32 > 0
        assert profile.tdp_watts > 0
        assert profile.device == "cpu"

    def test_get_apple_m2(self):
        profile = get_profile("apple_m2")
        assert profile.device == "mps"
        assert profile.flops_per_second_fp32 > 1e12

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError, match="Unknown hardware profile"):
            get_profile("nonexistent_gpu_9000")

    def test_all_profiles_have_required_fields(self):
        for name, profile in list_profiles().items():
            assert profile.name
            assert profile.flops_per_second_fp32 > 0
            assert profile.flops_per_second_fp16 > 0
            assert profile.tdp_watts > 0
            assert profile.device in ("cpu", "cuda", "mps")
