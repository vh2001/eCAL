"""eCAL: Analytical estimation of the energy cost of the AI lifecycle (J/bit)."""

from ecal._version import __version__
from ecal.api import estimate

__all__ = ["estimate", "__version__"]
