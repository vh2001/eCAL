"""Tests for the Transmission calculator."""

import pytest
from ecal.calculators.transmission import Transmission


class TestTransmission:
    def test_zero_failure_rate(self, generic_transmission):
        result = generic_transmission.calculate_energy(1000)
        assert result["total_energy"] > 0
        assert result["expected_transmissions"] == 1.0
        assert result["failure_rate"] == 0.0

    def test_increasing_failure_increases_energy(self):
        tx_0 = Transmission(
            application="Generic_application", presentation="Generic_presentation",
            session="Generic_session", transport="Generic_transport",
            network="Generic_network", datalink="Generic_datalink",
            physical="Generic_physical", failure_rate=0.0,
        )
        tx_50 = Transmission(
            application="Generic_application", presentation="Generic_presentation",
            session="Generic_session", transport="Generic_transport",
            network="Generic_network", datalink="Generic_datalink",
            physical="Generic_physical", failure_rate=0.5,
        )
        e0 = tx_0.calculate_energy(1000)["total_energy"]
        e50 = tx_50.calculate_energy(1000)["total_energy"]
        assert e50 > e0

    def test_invalid_failure_rate(self):
        with pytest.raises(ValueError, match="Failure rate must be between 0 and 1"):
            Transmission(failure_rate=1.5)

    def test_negative_failure_rate(self):
        with pytest.raises(ValueError, match="Failure rate must be between 0 and 1"):
            Transmission(failure_rate=-0.1)

    def test_layer_breakdown_present(self, generic_transmission):
        result = generic_transmission.calculate_energy(640000)
        assert "layer_breakdown" in result
        assert "application" in result["layer_breakdown"]
        assert "physical" in result["layer_breakdown"]

    def test_more_bits_more_energy(self, generic_transmission):
        e_small = generic_transmission.calculate_energy(100)["total_energy"]
        e_large = generic_transmission.calculate_energy(10000)["total_energy"]
        assert e_large > e_small

    def test_specific_protocols(self):
        tx = Transmission(
            application="HTTP", presentation="TLS", session="RPC",
            transport="TCP", network="IPv4", datalink="ETHERNET",
            physical="WIFI_PHY", failure_rate=0.0,
        )
        result = tx.calculate_energy(1000)
        assert result["total_energy"] > 0
