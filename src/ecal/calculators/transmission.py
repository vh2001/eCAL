from typing import Dict, Union
from ecal.configs.protocol_configs import (
    LayerProtocol,
    APPLICATION_PROTOCOLS,
    PRESENTATION_PROTOCOLS,
    SESSION_PROTOCOLS,
    TRANSPORT_PROTOCOLS,
    NETWORK_PROTOCOLS,
    DATALINK_PROTOCOLS,
    PHYSICAL_PROTOCOLS,
)


class Transmission:
    """
    Simplified calculator for network energy consumption that allows protocol selection
    for each OSI layer, focusing only on data and control plane overheads
    """

    def __init__(self,
                 application: str = 'HTTP',
                 presentation: str = 'TLS',
                 session: str = 'RPC',
                 transport: str = 'TCP',
                 network: str = 'IPv4',
                 datalink: str = 'WIFI_MAC',
                 physical: str = 'WIFI_PHY',
                 failure_rate: float = 0.0):
        """
        Initialize calculator with specific protocols for each layer

        Args:
            failure_rate: Probability of transmission failure (0.0 to 1.0)
        """
        self.protocols = {
            'application': APPLICATION_PROTOCOLS[application],
            'presentation': PRESENTATION_PROTOCOLS[presentation],
            'session': SESSION_PROTOCOLS[session],
            'transport': TRANSPORT_PROTOCOLS[transport],
            'network': NETWORK_PROTOCOLS[network],
            'datalink': DATALINK_PROTOCOLS[datalink],
            'physical': PHYSICAL_PROTOCOLS[physical]
        }
        if not 0 <= failure_rate <= 1:
            raise ValueError("Failure rate must be between 0 and 1")
        self.failure_rate = failure_rate

    def calculate_layer_energy(self, protocol: LayerProtocol, input_bits: int) -> Dict[str, Union[float, int]]:
        """Calculate energy consumption for a single layer"""

        # Calculate overhead bits
        data_plane_bits = int(input_bits * protocol.data_plane_overhead)
        control_plane_bits = int(input_bits * protocol.control_plane_overhead)

        # Total bits at this layer
        total_bits = input_bits + data_plane_bits + control_plane_bits
        first_term = total_bits * protocol.base_energy_per_bit_sender
        second_term = total_bits * protocol.base_energy_per_bit_receiver
        third_term = total_bits * protocol.Niot * protocol.Piot  # Niot
        fourth_term = total_bits * protocol.Ngateway * protocol.Pgateway  # Ngateway

        total_energy = first_term + second_term + third_term + fourth_term

        return {
            'total_bits': total_bits,
            'total_energy': total_energy,
            'breakdown': {
                'first_term': first_term,
                'second_term': second_term,
                'third_term': third_term,
                'fourth_term': fourth_term
            }
        }

    def calculate_energy(self, data_bits: int) -> Dict[str, Union[float, Dict]]:
        """Calculate energy consumption with retransmission consideration"""
        base_result = self._calculate_single_transmission(data_bits)

        # Calculate expected number of transmissions using geometric distribution
        # E[X] = 1/(1-p) where p is failure rate
        expected_transmissions = 1 / (1 - self.failure_rate)

        total_energy = base_result['total_energy'] * expected_transmissions
        total_bits = base_result['total_bits'] * expected_transmissions

        return {
            'total_energy': total_energy,
            'total_bits': total_bits,
            'original_bits': data_bits,
            'expected_transmissions': expected_transmissions,
            'failure_rate': self.failure_rate,
            'single_transmission': base_result,
            'layer_breakdown': base_result['layer_breakdown']
        }

    def _calculate_single_transmission(self, data_bits: int) -> Dict[str, Union[float, Dict]]:
        """Original calculation logic for a single transmission"""
        current_bits = data_bits
        total_energy = 0
        layer_results = {}

        for layer_name, protocol in self.protocols.items():
            curr_layer_result = self.calculate_layer_energy(protocol, current_bits)
            layer_results[layer_name] = {
                'protocol': protocol.name,
                'energy': curr_layer_result['total_energy'],
                'breakdown': curr_layer_result['breakdown']
            }
            total_energy += curr_layer_result['total_energy']
            current_bits = curr_layer_result['total_bits']

        return {
            'total_energy': total_energy,
            'total_bits': current_bits,
            'original_bits': data_bits,
            'layer_breakdown': layer_results
        }
