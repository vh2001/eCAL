"""Backward-compatibility shim — use ecal.configs.protocol_configs instead."""
from ecal.configs.protocol_configs import (  # noqa: F401
    LayerProtocol,
    APPLICATION_PROTOCOLS,
    PRESENTATION_PROTOCOLS,
    SESSION_PROTOCOLS,
    TRANSPORT_PROTOCOLS,
    NETWORK_PROTOCOLS,
    DATALINK_PROTOCOLS,
    PHYSICAL_PROTOCOLS,
)
