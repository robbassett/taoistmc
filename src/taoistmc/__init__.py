from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("taoistmc")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

from taoistmc.core import SightlineSampler, OpticalDepthCalculator
from taoistmc.taoist import TaoistMc
from taoistmc import config

__all__ = [
    "TaoistMc",
    "SightlineSampler",
    "OpticalDepthCalculator",
    "config"
]