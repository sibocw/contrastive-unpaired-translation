"""Training module package."""

from .train import main as train_main
from .test import main as test_main

__all__ = ['train_main', 'test_main']
