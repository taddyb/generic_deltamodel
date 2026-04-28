# src/dmg/core/data/loaders/__init__.py
from .base import BaseLoader
from .hydro_loader import HydroLoader
from .ms_hydro_loader import MsHydroLoader
from .mts_hydro_loader import MtsHydroLoader

try:
    from .uh_hydro_loader import UhHydroLoader
except ImportError:
    UhHydroLoader = None

__all__ = [
    'BaseLoader',
    'HydroLoader',
    'MsHydroLoader',
    'MtsHydroLoader',
    'UhHydroLoader',
]
