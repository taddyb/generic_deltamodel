# src/dmg/trainers/__init__.py
from .base import BaseTrainer
from .ms_trainer import MsTrainer
from .trainer import Trainer
try:
    from .uh_trainer import UhTrainer
except ImportError:
    UhTrainer = None

__all__ = [
    'BaseTrainer',
    'Trainer',
    'MsTrainer',
    'UhTrainer',
]
